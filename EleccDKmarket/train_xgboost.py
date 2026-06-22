#!/usr/bin/env python3
"""
=============================================================================
 train_xgboost.py — Baseline XGBoost pour la prédiction du prix spot Day-Ahead
                     de l'électricité au Danemark (DK1 / DK2)
=============================================================================
 Train : janvier 2022 → décembre 2023
 Test  : janvier 2024 → décembre 2024

 Hyperparamètres : Optuna (TPE bayésien + MedianPruner)
 Explicabilité   : SHAP (TreeExplainer)
 Réf. scientifique : Wijaya et al. (2024) — plages de recherche ciblées
=============================================================================
 Dépendances :
   pip install pandas numpy scikit-learn xgboost optuna shap matplotlib joblib
=============================================================================
 Arborescence attendue :
   ./final_data/train_raw.csv
   ./final_data/test_raw.csv
   ./results/                    ← créé automatiquement
=============================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR     = "./final_data"
OUTPUT_DIR   = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXCLUDE_COLS = ["HourUTC", "PriceArea"]
TARGET       = "SpotPriceEUR"

# Optuna
N_TRIALS     = 150        # essais bayésiens (convergence typique autour de 80-100)
N_SPLITS     = 3          # folds TimeSeriesSplit pour le CV
RANDOM_STATE = 42
N_JOBS       = -1         # tous les cœurs disponibles


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHARGEMENT & PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    """Charge train_raw.csv / test_raw.csv et sépare X / y."""
    train_path = os.path.join(DATA_DIR, "train_raw.csv")
    test_path  = os.path.join(DATA_DIR, "test_raw.csv")

    df_train = pd.read_csv(train_path, parse_dates=["HourUTC"])
    df_test  = pd.read_csv(test_path,  parse_dates=["HourUTC"])

    print(f"[DATA] Train : {df_train.shape[0]:,} lignes × {df_train.shape[1]} cols")
    print(f"[DATA] Test  : {df_test.shape[0]:,}  lignes × {df_test.shape[1]} cols")
    print(f"[DATA] Plage train : {df_train['HourUTC'].min()} → {df_train['HourUTC'].max()}")
    print(f"[DATA] Plage test  : {df_test['HourUTC'].min()} → {df_test['HourUTC'].max()}")

    # Colonnes features (tout sauf index/suivi et cible)
    feature_cols = [c for c in df_train.columns if c not in EXCLUDE_COLS + [TARGET]]

    X_train = df_train[feature_cols].copy()
    y_train = df_train[TARGET].copy()
    X_test  = df_test[feature_cols].copy()
    y_test  = df_test[TARGET].copy()

    # Drop NaN résiduel (sécurité — normalement 0%)
    mask_tr = X_train.isna().any(axis=1) | y_train.isna()
    mask_te = X_test.isna().any(axis=1)  | y_test.isna()
    if mask_tr.sum():
        X_train, y_train = X_train[~mask_tr], y_train[~mask_tr]
        print(f"[WARN] {mask_tr.sum()} lignes train droppées (NaN)")
    if mask_te.sum():
        X_test, y_test = X_test[~mask_te], y_test[~mask_te]
        print(f"[WARN] {mask_te.sum()} lignes test droppées (NaN)")

    print(f"[DATA] {len(feature_cols)} features retenues :")
    for i, c in enumerate(feature_cols, 1):
        print(f"       {i:2d}. {c}")
    print(f"[DATA] NaN → train: {X_train.isna().sum().sum()} | test: {X_test.isna().sum().sum()}")

    return X_train, y_train, X_test, y_test, df_test, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTIMISATION BAYÉSIENNE — OPTUNA (TPE)
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial, X_train, y_train):
    """
    Fonction objectif Optuna.
    Espace de recherche élargi par rapport à Wijaya et al. (2024) :
      - learning_rate centré sur [0.05, 0.25] (les auteurs recommandent 0.1-0.2)
      - max_depth [5, 12] (recommandé 7-10, mais on laisse Optuna explorer)
      - n_estimators [150, 500] (recommandé 200-300)
      + régularisation L1/L2 et gamma pour contrôler l'overfitting
    """
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 0.05, 0.25, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 150, 500),
        "max_depth":        trial.suggest_int("max_depth", 5, 12),
        "subsample":        trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    model = XGBRegressor(
        **params,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=N_JOBS,
    )

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    return -scores.mean()


def run_optuna(X_train, y_train):
    """Lance l'étude Optuna et réentraîne le modèle final."""
    print("\n" + "=" * 72)
    print("  OPTIMISATION BAYÉSIENNE — Optuna (TPE + MedianPruner)")
    print("=" * 72)
    print(f"[OPTUNA] {N_TRIALS} essais | {N_SPLITS} folds TimeSeriesSplit\n")

    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name="xgb_spot_price_dk",
    )

    t0 = time.time()
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0

    print(f"\n[OPTUNA] Terminé en {elapsed / 60:.1f} min")
    print(f"[OPTUNA] Meilleur MAE CV : {study.best_value:.4f} €/MWh")
    print(f"[OPTUNA] Meilleurs paramètres :")
    for k, v in study.best_params.items():
        print(f"         {k}: {round(v, 6) if isinstance(v, float) else v}")

    # Réentraînement sur l'intégralité du train
    best_model = XGBRegressor(
        **study.best_params,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=N_JOBS,
    )
    best_model.fit(X_train, y_train)

    return best_model, study


# ─────────────────────────────────────────────────────────────────────────────
# 4. ÉVALUATION SUR L'ANNÉE 2024 (TEST)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    """MAE, RMSE, R² sur le jeu de test 2024."""
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n" + "=" * 72)
    print("  PERFORMANCE SUR LE TEST 2024")
    print("=" * 72)
    print(f"  MAE  : {mae:>10.4f}  €/MWh")
    print(f"  RMSE : {rmse:>10.4f}  €/MWh")
    print(f"  R²   : {r2:>10.4f}")
    print("=" * 72)

    return y_pred, {"MAE": mae, "RMSE": rmse, "R2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPLICABILITÉ SHAP
# ─────────────────────────────────────────────────────────────────────────────
def run_shap_analysis(model, X_train, X_test, feature_cols, n_background=500):
    """
    Analyse SHAP complète via TreeExplainer :
      1. Beeswarm plot  — direction + magnitude de l'impact par feature
      2. Bar plot        — importance moyenne |SHAP| (comparable au gain XGB)
      3. Dependence plots — interaction non-linéaire des 4 top features
      4. Waterfall        — explication d'une prédiction individuelle (1ère obs test)
    """
    print("\n" + "=" * 72)
    print("  EXPLICABILITÉ SHAP — TreeExplainer")
    print("=" * 72)

    # Background pour l'explainer (sous-échantillon train)
    explainer   = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_test)
    print(f"[SHAP] {X_test.shape[0]:,} observations test expliquées")


    # ── 5a. Beeswarm (global) ──
    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_cols,
        show=False, max_display=25,
    )
    plt.title("SHAP Beeswarm — Impact directionnel des features", fontsize=13, pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_beeswarm.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SHAP] Beeswarm → {path}")

    # ── 5b. Bar plot (global) ──
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_cols,
        plot_type="bar", show=False, max_display=25,
    )
    plt.title("SHAP — Importance moyenne |SHAP value|", fontsize=13, pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SHAP] Bar → {path}")

    # ── 5c. Dependence plots — Top 4 ──
    mean_abs = np.abs(shap_values).mean(axis=0)
    top4_idx = np.argsort(mean_abs)[::-1][:4]
    top4     = [feature_cols[i] for i in top4_idx]
    print(f"[SHAP] Dependence plots pour : {top4}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for feat, ax in zip(top4, axes.flatten()):
        fi = feature_cols.index(feat)
        shap.dependence_plot(fi, shap_values, X_test,
                             feature_names=feature_cols, ax=ax, show=False)
        ax.set_title(f"Dependence : {feat}", fontsize=11)
    plt.suptitle("SHAP Dependence — Top 4 Features", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_dependence_top4.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SHAP] Dependence → {path}")

    # ── 5d. Waterfall — 1ère observation test ──
    try:
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test.iloc[0].values,
            feature_names=feature_cols,
        )
        fig = plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title("SHAP Waterfall — Explication de la 1ère prédiction test", fontsize=12)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "shap_waterfall_sample.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[SHAP] Waterfall → {path}")
    except Exception as e:
        print(f"[SHAP] Waterfall ignoré (erreur mineure) : {e}")

    return shap_values


# ─────────────────────────────────────────────────────────────────────────────
# 6. GRAPHIQUES DE PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def plot_predictions_vs_actual(df_test, y_test, y_pred, metrics):
    """Série temporelle réel vs prédit + scatter."""
    timestamps = pd.to_datetime(df_test["HourUTC"].values[:len(y_test)])

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Vue globale 2024
    axes[0].plot(timestamps, y_test.values, label="Réel",  alpha=0.7, lw=0.5, color="#1E40AF")
    axes[0].plot(timestamps, y_pred,        label="Prédit", alpha=0.7, lw=0.5, color="#DC2626")
    axes[0].set_title("Prix Spot DK1/DK2 — Réel vs Prédit (2024)", fontsize=14)
    axes[0].set_ylabel("€/MWh")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    # Zoom 2 semaines de janvier
    zoom_end = timestamps.min() + pd.Timedelta(days=14)
    mask_z   = timestamps <= zoom_end
    ax_ins = axes[0].inset_axes([0.55, 0.55, 0.42, 0.40])
    ax_ins.plot(timestamps[mask_z], y_test.values[mask_z], lw=1.2, color="#1E40AF")
    ax_ins.plot(timestamps[mask_z], y_pred[mask_z],        lw=1.2, color="#DC2626")
    ax_ins.set_title("Zoom 2 semaines", fontsize=9)
    ax_ins.tick_params(labelsize=7)
    ax_ins.grid(alpha=0.3)

    # Scatter
    axes[1].scatter(y_test, y_pred, alpha=0.15, s=4, color="#6366F1")
    lo = min(y_test.min(), y_pred.min()) - 5
    hi = max(y_test.max(), y_pred.max()) + 5
    axes[1].plot([lo, hi], [lo, hi], "--", color="#DC2626", lw=1, label="Prédiction parfaite")
    axes[1].set_xlim(lo, hi); axes[1].set_ylim(lo, hi)
    axes[1].set_xlabel("Prix réel (€/MWh)")
    axes[1].set_ylabel("Prix prédit (€/MWh)")
    axes[1].set_title(
        f"Scatter — MAE={metrics['MAE']:.2f} | RMSE={metrics['RMSE']:.2f} | R²={metrics['R2']:.4f}",
        fontsize=13,
    )
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "predictions_vs_actual.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Prédictions vs Réel → {path}")


def plot_residuals(y_test, y_pred, timestamps):
    """Histogramme + scatter temporel des résidus."""
    resid = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(resid, bins=100, color="#6366F1", edgecolor="none", alpha=0.8)
    axes[0].axvline(0, color="#DC2626", ls="--", lw=1)
    axes[0].set_xlabel("Résidu (€/MWh)"); axes[0].set_ylabel("Fréquence")
    axes[0].set_title(f"Distribution des résidus (μ={resid.mean():.2f}, σ={resid.std():.2f})")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(timestamps, resid, s=1, alpha=0.2, color="#6366F1")
    axes[1].axhline(0, color="#DC2626", ls="--", lw=1)
    axes[1].set_xlabel("Date"); axes[1].set_ylabel("Résidu (€/MWh)")
    axes[1].set_title("Résidus temporels (2024)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "residuals.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Résidus → {path}")


def plot_optuna_convergence(study):
    """Courbe de convergence de l'optimisation bayésienne."""
    vals = [t.value for t in study.trials if t.value is not None]
    best = [min(vals[:i+1]) for i in range(len(vals))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(vals, "o", ms=3, alpha=0.4, color="#6366F1", label="MAE par essai")
    ax.plot(best, "-", lw=2, color="#DC2626", label="Meilleur cumulé")
    ax.set_xlabel("Essai Optuna"); ax.set_ylabel("MAE CV (€/MWh)")
    ax.set_title("Convergence Optuna — Optimisation bayésienne TPE")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "optuna_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Convergence → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(model, study, metrics):
    """Sauvegarde modèle .joblib, historique Optuna .csv, métriques .txt."""
    model_path = os.path.join(OUTPUT_DIR, "xgb_best_model.joblib")
    joblib.dump(model, model_path)
    print(f"[SAVE] Modèle → {model_path}")

    trials_path = os.path.join(OUTPUT_DIR, "optuna_trials.csv")
    study.trials_dataframe().to_csv(trials_path, index=False)
    print(f"[SAVE] Optuna trials → {trials_path}")

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=" * 55 + "\n")
        f.write("  XGBoost Baseline — Métriques Test 2024\n")
        f.write("=" * 55 + "\n\n")
        for k, v in metrics.items():
            f.write(f"  {k:6s} : {v:.6f}\n")
        f.write("\n" + "-" * 55 + "\n")
        f.write("  Hyperparamètres optimaux (Optuna TPE)\n")
        f.write("-" * 55 + "\n\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}: {round(v, 6) if isinstance(v, float) else v}\n")
        f.write(f"\n  Meilleur MAE CV : {study.best_value:.6f}\n")
        f.write(f"  Essais total    : {len(study.trials)}\n")
    print(f"[SAVE] Métriques → {metrics_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "▓" * 72)
    print("  PIPELINE XGBOOST — Prix Spot Day-Ahead Danemark (DK1/DK2)")
    print("  Optuna (TPE bayésien) + SHAP (TreeExplainer)")
    print("▓" * 72 + "\n")

    # Étape 1 — Chargement
    X_train, y_train, X_test, y_test, df_test, feature_cols = load_data()

    # Étape 2 — Optimisation Optuna
    best_model, study = run_optuna(X_train, y_train)

    # Étape 3 — Évaluation Test 2024
    y_pred, metrics = evaluate(best_model, X_test, y_test)

    # Étape 4 — Explicabilité SHAP
    run_shap_analysis(best_model, X_train, X_test, feature_cols)

    # Étape 5 — Graphiques de performance
    timestamps = pd.to_datetime(df_test["HourUTC"].values[:len(y_test)])
    plot_predictions_vs_actual(df_test, y_test, y_pred, metrics)
    plot_residuals(y_test, y_pred, timestamps)
    plot_optuna_convergence(study)

    # Étape 6 — Sauvegarde
    save_artifacts(best_model, study, metrics)

    print("\n" + "▓" * 72)
    print("  PIPELINE TERMINÉ")
    print(f"  → Résultats dans ./{OUTPUT_DIR}/")
    print("  → Fichiers : xgb_best_model.joblib, optuna_trials.csv, metrics.txt")
    print("  → Graphiques : shap_beeswarm.png, shap_bar.png, shap_dependence_top4.png,")
    print("                 shap_waterfall_sample.png, predictions_vs_actual.png,")
    print("                 residuals.png, optuna_convergence.png")
    print("▓" * 72 + "\n")


if __name__ == "__main__":
    main()