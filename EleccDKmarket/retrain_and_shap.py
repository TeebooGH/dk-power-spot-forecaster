#!/usr/bin/env python3
"""
=============================================================================
 retrain_and_shap.py — Réentraînement XGBoost avec les params Optuna déjà
                        trouvés + SHAP complet
=============================================================================
 Les hyperparamètres viennent du run Optuna (trial 137, MAE CV = 1.8080).
 Pas besoin de relancer l'optimisation.
=============================================================================
 Usage :  python retrain_and_shap.py
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "./final_data"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXCLUDE_COLS = ["HourUTC", "PriceArea"]
TARGET       = "SpotPriceEUR"

# Hyperparamètres optimaux — Optuna trial 137 (MAE CV = 1.8080 €/MWh)
BEST_PARAMS = {
    "learning_rate":    0.050105,
    "n_estimators":     242,
    "max_depth":        12,
    "subsample":        0.907314,
    "colsample_bytree": 0.99968,
    "min_child_weight": 3,
    "gamma":            0.010045,
    "reg_alpha":        0.000267,
    "reg_lambda":       1.406613,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "▓" * 72)
print("  RÉENTRAÎNEMENT XGBOOST + SHAP (params Optuna trial 137)")
print("▓" * 72)

df_train = pd.read_csv(os.path.join(DATA_DIR, "train_raw.csv"), parse_dates=["HourUTC"])
df_test  = pd.read_csv(os.path.join(DATA_DIR, "test_raw.csv"),  parse_dates=["HourUTC"])

feature_cols = [c for c in df_train.columns if c not in EXCLUDE_COLS + [TARGET]]

X_train = df_train[feature_cols].copy()
y_train = df_train[TARGET].copy()
X_test  = df_test[feature_cols].copy()
y_test  = df_test[TARGET].copy()

print(f"[DATA] Train : {X_train.shape[0]:,} × {X_train.shape[1]} features")
print(f"[DATA] Test  : {X_test.shape[0]:,} × {X_test.shape[1]} features")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ENTRAÎNEMENT (~ 30 secondes)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TRAIN] Entraînement XGBoost avec les params Optuna...")
model = XGBRegressor(
    **BEST_PARAMS,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("[TRAIN] Terminé")

# Sauvegarde immédiate du modèle
model_path = os.path.join(OUTPUT_DIR, "xgb_best_model.joblib")
joblib.dump(model, model_path)
print(f"[SAVE] Modèle → {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────
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

# Sauvegarde métriques
metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write("=" * 55 + "\n")
    f.write("  XGBoost Baseline — Métriques Test 2024\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"  MAE    : {mae:.6f}\n")
    f.write(f"  RMSE   : {rmse:.6f}\n")
    f.write(f"  R2     : {r2:.6f}\n")
    f.write("\n" + "-" * 55 + "\n")
    f.write("  Hyperparamètres (Optuna trial 137)\n")
    f.write("-" * 55 + "\n\n")
    for k, v in BEST_PARAMS.items():
        f.write(f"  {k}: {v}\n")
print(f"[SAVE] Métriques → {metrics_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRAPHIQUES DE PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
timestamps = pd.to_datetime(df_test["HourUTC"].values[:len(y_test)])

# Série temporelle + scatter
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

axes[0].plot(timestamps, y_test.values, label="Réel",  alpha=0.7, lw=0.5, color="#1E40AF")
axes[0].plot(timestamps, y_pred,        label="Prédit", alpha=0.7, lw=0.5, color="#DC2626")
axes[0].set_title("Prix Spot DK1/DK2 — Réel vs Prédit (2024)", fontsize=14)
axes[0].set_ylabel("€/MWh"); axes[0].legend(); axes[0].grid(alpha=0.3)

zoom_end = timestamps.min() + pd.Timedelta(days=14)
mask_z   = timestamps <= zoom_end
ax_ins = axes[0].inset_axes([0.55, 0.55, 0.42, 0.40])
ax_ins.plot(timestamps[mask_z], y_test.values[mask_z], lw=1.2, color="#1E40AF")
ax_ins.plot(timestamps[mask_z], y_pred[mask_z],        lw=1.2, color="#DC2626")
ax_ins.set_title("Zoom 2 semaines", fontsize=9); ax_ins.tick_params(labelsize=7); ax_ins.grid(alpha=0.3)

axes[1].scatter(y_test, y_pred, alpha=0.15, s=4, color="#6366F1")
lo = min(y_test.min(), y_pred.min()) - 5
hi = max(y_test.max(), y_pred.max()) + 5
axes[1].plot([lo, hi], [lo, hi], "--", color="#DC2626", lw=1, label="Prédiction parfaite")
axes[1].set_xlim(lo, hi); axes[1].set_ylim(lo, hi)
axes[1].set_xlabel("Prix réel (€/MWh)"); axes[1].set_ylabel("Prix prédit (€/MWh)")
axes[1].set_title(f"Scatter — MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}", fontsize=13)
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "predictions_vs_actual.png"), dpi=150)
plt.close(fig)
print("[PLOT] predictions_vs_actual.png")

# Résidus
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
axes[1].set_title("Résidus temporels (2024)"); axes[1].grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "residuals.png"), dpi=150)
plt.close(fig)
print("[PLOT] residuals.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPLICABILITÉ SHAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  EXPLICABILITÉ SHAP — TreeExplainer (tree_path_dependent)")
print("=" * 72)

explainer   = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(X_test)
print(f"[SHAP] {X_test.shape[0]:,} observations expliquées")

# Beeswarm
fig = plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False, max_display=25)
plt.title("SHAP Beeswarm — Impact directionnel des features", fontsize=13, pad=15)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[SHAP] shap_beeswarm.png")

# Bar
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                  plot_type="bar", show=False, max_display=25)
plt.title("SHAP — Importance moyenne |SHAP value|", fontsize=13, pad=15)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[SHAP] shap_bar.png")

# Dependence — Top 4
mean_abs = np.abs(shap_values).mean(axis=0)
top4_idx = np.argsort(mean_abs)[::-1][:4]
top4     = [feature_cols[i] for i in top4_idx]
print(f"[SHAP] Dependence plots : {top4}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for feat, ax in zip(top4, axes.flatten()):
    fi = feature_cols.index(feat)
    shap.dependence_plot(fi, shap_values, X_test,
                         feature_names=feature_cols, ax=ax, show=False)
    ax.set_title(f"Dependence : {feat}", fontsize=11)
plt.suptitle("SHAP Dependence — Top 4 Features", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "shap_dependence_top4.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[SHAP] shap_dependence_top4.png")

# Waterfall — 1ère observation
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
    fig.savefig(os.path.join(OUTPUT_DIR, "shap_waterfall_sample.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[SHAP] shap_waterfall_sample.png")
except Exception as e:
    print(f"[SHAP] Waterfall ignoré : {e}")


print("\n" + "▓" * 72)
print("  TERMINÉ — Tous les fichiers dans ./results/")
print("  Modèle : xgb_best_model.joblib")
print("  Métriques : metrics.txt")
print("  Graphiques : predictions_vs_actual.png, residuals.png")
print("  SHAP : shap_beeswarm.png, shap_bar.png, shap_dependence_top4.png,")
print("         shap_waterfall_sample.png")
print("▓" * 72 + "\n")