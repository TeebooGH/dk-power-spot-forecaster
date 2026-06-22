#!/usr/bin/env python3
"""
==========================================================================
  train_lstm.py — LSTM pour la prédiction du prix spot Day-Ahead (DK1/DK2)
==========================================================================

Pipeline complet PyTorch :
  1. Chargement des données pré-standardisées (train_scaled / test_scaled)
  2. Création des séquences glissantes (sliding window, lookback = 24 h)
  3. Architecture LSTM bi-couche avec Dropout + Dense
  4. Entraînement avec Early Stopping sur validation temporelle
  5. Inférence sur Test 2024, dé-standardisation, métriques finales

Auteur : Annie / Claude  —  Juin 2026
Inspiré de : Kılıç et al. (2024), Dumas et al. (2021)
"""

import os
import time
import copy
import warnings
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib

matplotlib.use("Agg")  # backend non-interactif pour serveur
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION CENTRALE
# ═══════════════════════════════════════════════════════════════

class Config:
    """Hyperparamètres et chemins — un seul endroit à modifier."""

    # ── Chemins des données ──
    TRAIN_SCALED  = "./final_data/train_scaled.csv"
    TEST_SCALED   = "./final_data/test_scaled.csv"
    TRAIN_RAW     = "./final_data/train_raw.csv"   # pour recalculer μ/σ du descaling

    # ── Colonnes ──
    TARGET_COL    = "SpotPriceEUR"
    EXCLUDE_COLS  = ["HourUTC", "PriceArea"]       # présentes mais exclues de X

    # ── Séquences (Windowing) ──
    LOOKBACK      = 24          # 24 heures d'historique en entrée
    HORIZON       = 1           # prédiction t+1

    # ── Architecture LSTM ──
    HIDDEN_SIZE   = 64          # unités par couche LSTM
    NUM_LAYERS    = 2           # couches LSTM empilées
    DROPOUT       = 0.2         # dropout entre couches LSTM

    # ── Entraînement ──
    EPOCHS        = 150
    BATCH_SIZE    = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-5        # régularisation L2
    VAL_RATIO     = 0.15        # 15 % du train (fin de période) → validation
    PATIENCE      = 15          # early stopping patience (époques)
    MIN_DELTA     = 1e-4        # amélioration minimale pour compter

    # ── Reproductibilité ──
    SEED          = 42

    # ── Sorties ──
    OUTPUT_DIR    = "./lstm_results"
    MODEL_PATH    = "./lstm_results/best_lstm_model.pt"


def set_seed(seed: int) -> None:
    """Fixe les graines aléatoires pour la reproductibilité."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════
#  1. CHARGEMENT & PRÉPARATION DES DONNÉES
# ═══════════════════════════════════════════════════════════════

def load_and_prepare(
    path: str, target: str, exclude: list
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, list]:
    """
    Charge un CSV scalé, sépare X / Y en arrays NumPy.

    Returns
    -------
    df_meta : DataFrame contenant HourUTC + PriceArea (pour le suivi)
    X       : array (n_samples, n_features)
    Y       : array (n_samples,)
    feature_names : liste des noms de colonnes utilisées en features
    """
    df = pd.read_csv(path, parse_dates=["HourUTC"])

    # Colonnes de suivi (non utilisées pour l'apprentissage)
    meta_cols = [c for c in exclude if c in df.columns]
    df_meta = df[meta_cols].copy()

    # Séparation features / cible
    drop_cols = [c for c in exclude + [target] if c in df.columns]
    feature_names = [c for c in df.columns if c not in drop_cols]

    X = df[feature_names].values.astype(np.float32)
    Y = df[target].values.astype(np.float32)

    return df_meta, X, Y, feature_names


def compute_descale_params(
    raw_train_path: str, scaled_train_path: str, target_col: str
) -> Tuple[Optional[float], Optional[float]]:
    """
    Détecte automatiquement si la cible a été standardisée dans le fichier
    scalé, puis retourne (μ, σ) pour le descaling — ou (None, None) si la
    cible est déjà en €/MWh bruts (pas de descaling nécessaire).

    Diagnostic : une cible StandardScaler-transformée a mean ≈ 0 et std ≈ 1.
    Si mean(target_scaled) >> 1, la cible n'a pas été touchée par le scaler.
    """
    df_raw    = pd.read_csv(raw_train_path)
    df_scaled = pd.read_csv(scaled_train_path)

    raw_mu  = df_raw[target_col].mean()
    raw_std = df_raw[target_col].std(ddof=0)

    scaled_mu  = df_scaled[target_col].mean()
    scaled_std = df_scaled[target_col].std(ddof=0)

    print(f"[DESCALE] Train RAW    → μ = {raw_mu:.4f}, σ = {raw_std:.4f}")
    print(f"[DESCALE] Train SCALED → μ = {scaled_mu:.4f}, σ = {scaled_std:.4f}")

    # Si la cible dans le fichier scalé a un mean éloigné de 0,
    # elle n'a PAS été standardisée → pas de descaling
    if abs(scaled_mu) > 5.0 or scaled_std > 5.0:
        print("[DESCALE] ⚠ La cible SpotPriceEUR n'est PAS standardisée "
              "dans train_scaled.csv → descaling désactivé.")
        return None, None

    print("[DESCALE] ✓ Cible standardisée détectée → descaling actif.")
    return raw_mu, raw_std


def inverse_scale(
    arr: np.ndarray, mu: Optional[float], std: Optional[float]
) -> np.ndarray:
    """
    Dé-standardise : x_real = x_scaled × σ + μ.
    Si mu/std sont None, retourne l'array inchangé (cible déjà en €/MWh).
    """
    if mu is None or std is None:
        return arr
    return arr * std + mu


# ═══════════════════════════════════════════════════════════════
#  2. DATASET PYTORCH AVEC SLIDING WINDOW
# ═══════════════════════════════════════════════════════════════

class ElectricityWindowDataset(Dataset):
    """
    Dataset qui fabrique des séquences glissantes (sliding windows).

    Pour chaque index t, retourne :
        X_seq : (lookback, n_features) — les `lookback` heures précédentes
        y     : scalaire — le prix spot à l'heure t (= prédiction horizon +1
                par rapport au début de la fenêtre)
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, lookback: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.Y) - self.lookback

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fenêtre d'entrée : indices [idx, idx+lookback)
        x_seq = self.X[idx : idx + self.lookback]
        # Cible : l'heure immédiatement après la fenêtre
        y = self.Y[idx + self.lookback]
        return x_seq, y


# ═══════════════════════════════════════════════════════════════
#  3. ARCHITECTURE DU RÉSEAU LSTM
# ═══════════════════════════════════════════════════════════════

class SpotPriceLSTM(nn.Module):
    """
    LSTM bi-couche pour la prédiction du prix spot horaire.

    Architecture :
        Input (batch, lookback, n_features)
          → LSTM × num_layers (hidden_size, dropout entre couches)
          → On récupère le dernier hidden state
          → Dense (hidden_size → 1)
          → Prédiction scalaire
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Couche de sortie : projection linéaire → 1 valeur de prix
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape : (batch, lookback, n_features)
        lstm_out, (h_n, _) = self.lstm(x)
        # h_n shape : (num_layers, batch, hidden_size)
        # On prend le hidden state de la dernière couche
        last_hidden = h_n[-1]            # (batch, hidden_size)
        out = self.fc(last_hidden)       # (batch, 1)
        return out.squeeze(-1)           # (batch,)


# ═══════════════════════════════════════════════════════════════
#  4. EARLY STOPPING
# ═══════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Arrête l'entraînement si la loss de validation ne s'améliore plus
    pendant `patience` époques consécutives.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ═══════════════════════════════════════════════════════════════
#  5. BOUCLE D'ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Tuple[nn.Module, list, list]:
    """
    Boucle d'entraînement complète avec Early Stopping.

    Returns
    -------
    model           : modèle avec les meilleurs poids restaurés
    train_losses    : historique loss train par époque
    val_losses      : historique loss validation par époque
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    # Scheduler : réduit le LR si la val_loss stagne
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    early_stop = EarlyStopping(patience=cfg.PATIENCE, min_delta=cfg.MIN_DELTA)

    train_losses, val_losses = [], []

    print("\n" + "=" * 70)
    print("  ENTRAÎNEMENT LSTM — Early Stopping activé")
    print("=" * 70)
    print(f"  Device : {device}")
    print(f"  Époques max : {cfg.EPOCHS} | Patience : {cfg.PATIENCE}")
    print(f"  Batch size : {cfg.BATCH_SIZE} | LR : {cfg.LEARNING_RATE}")
    print(f"  Train batches : {len(train_loader)} | Val batches : {len(val_loader)}")
    print("=" * 70 + "\n")

    t_start = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        # ── Phase Train ──
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        epoch_train_loss /= len(train_loader.dataset)

        # ── Phase Validation ──
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)
        epoch_val_loss /= len(val_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Affichage ──
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(
                f"  Epoch {epoch:>3d}/{cfg.EPOCHS} │ "
                f"Train Loss: {epoch_train_loss:.6f} │ "
                f"Val Loss: {epoch_val_loss:.6f} │ "
                f"LR: {current_lr:.2e} │ "
                f"ES: {early_stop.counter}/{cfg.PATIENCE} │ "
                f"{elapsed:.0f}s"
            )

        # ── Early Stopping check ──
        early_stop(epoch_val_loss, model)
        if early_stop.early_stop:
            print(f"\n  ✓ Early Stopping déclenché à l'époque {epoch}")
            print(f"    Meilleure Val Loss : {early_stop.best_loss:.6f}")
            break

    # Restaurer les meilleurs poids
    if early_stop.best_model_state is not None:
        model.load_state_dict(early_stop.best_model_state)
        print("  ✓ Meilleurs poids restaurés.")

    total_time = time.time() - t_start
    print(f"\n  Temps total d'entraînement : {total_time:.1f}s")

    return model, train_losses, val_losses


# ═══════════════════════════════════════════════════════════════
#  6. INFÉRENCE & ÉVALUATION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    """Génère toutes les prédictions sur un DataLoader."""
    model.eval()
    all_preds = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, label: str = "Test"
) -> dict:
    """Calcule et affiche MAE, RMSE, R² en €/MWh (valeurs dé-standardisées)."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n{'=' * 50}")
    print(f"  MÉTRIQUES FINALES — {label} (€/MWh)")
    print(f"{'=' * 50}")
    print(f"  MAE   : {mae:.2f} €/MWh")
    print(f"  RMSE  : {rmse:.2f} €/MWh")
    print(f"  R²    : {r2:.4f}")
    print(f"{'=' * 50}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ═══════════════════════════════════════════════════════════════
#  7. GRAPHIQUES
# ═══════════════════════════════════════════════════════════════

def plot_training_curves(
    train_losses: list, val_losses: list, output_dir: str
) -> None:
    """Courbes de loss train/val par époque."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", linewidth=1.2)
    ax.plot(val_losses, label="Validation Loss", linewidth=1.2)
    ax.set_xlabel("Époque")
    ax.set_ylabel("MSE Loss (espace standardisé)")
    ax.set_title("Courbes d'apprentissage — LSTM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → Sauvegardé : {path}")


def plot_predictions(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
) -> None:
    """
    Trace les prédictions vs valeurs réelles sur le Test 2024.
    Inclut un zoom sur un mois représentatif (mars).
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # ── Vue complète ──
    ax = axes[0]
    ax.plot(timestamps, y_true, label="Prix réel", alpha=0.7, linewidth=0.5)
    ax.plot(timestamps, y_pred, label="Prédiction LSTM", alpha=0.7, linewidth=0.5)
    ax.set_ylabel("SpotPriceEUR (€/MWh)")
    ax.set_title("Prédictions LSTM vs Prix Réels — Test 2024 (complet)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Zoom mars 2024 ──
    ax2 = axes[1]
    mask = (timestamps.dt.month == 3) & (timestamps.dt.year == 2024)
    if mask.sum() > 0:
        ax2.plot(
            timestamps[mask], y_true[mask],
            label="Prix réel", linewidth=1.0
        )
        ax2.plot(
            timestamps[mask], y_pred[mask],
            label="Prédiction LSTM", linewidth=1.0, linestyle="--"
        )
        ax2.set_title("Zoom — Mars 2024")
    else:
        # Fallback : premier mois disponible
        first_month = timestamps.dt.month.iloc[0]
        mask = timestamps.dt.month == first_month
        ax2.plot(timestamps[mask], y_true[mask], label="Prix réel", linewidth=1.0)
        ax2.plot(
            timestamps[mask], y_pred[mask],
            label="Prédiction LSTM", linewidth=1.0, linestyle="--"
        )
        ax2.set_title(f"Zoom — Mois {first_month}")

    ax2.set_ylabel("SpotPriceEUR (€/MWh)")
    ax2.set_xlabel("Date (UTC)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "predictions_vs_real.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → Sauvegardé : {path}")


def plot_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, output_dir: str
) -> None:
    """Scatter plot prédictions vs réel avec droite de parité."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.15, s=3, color="steelblue")
    lims = [
        min(y_true.min(), y_pred.min()) - 5,
        max(y_true.max(), y_pred.max()) + 5,
    ]
    ax.plot(lims, lims, "r--", linewidth=1.0, label="Parité parfaite")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Prix réel (€/MWh)")
    ax.set_ylabel("Prédiction LSTM (€/MWh)")
    ax.set_title("Scatter — Prédiction vs Réalité (Test 2024)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    path = os.path.join(output_dir, "scatter_pred_vs_real.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → Sauvegardé : {path}")


# ═══════════════════════════════════════════════════════════════
#  8. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    cfg = Config()
    set_seed(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Chargement ──
    print("\n📂 Chargement des données...")
    train_meta, X_train_full, Y_train_full, feature_names = load_and_prepare(
        cfg.TRAIN_SCALED, cfg.TARGET_COL, cfg.EXCLUDE_COLS
    )
    test_meta, X_test, Y_test, _ = load_and_prepare(
        cfg.TEST_SCALED, cfg.TARGET_COL, cfg.EXCLUDE_COLS
    )

    n_features = X_train_full.shape[1]
    print(f"  Features : {n_features} colonnes")
    print(f"  Train    : {X_train_full.shape[0]:,} heures")
    print(f"  Test     : {X_test.shape[0]:,} heures")

    # ── Paramètres de descaling (auto-détection) ──
    print("\n🔄 Détection du scaling de la cible...")
    target_mu, target_std = compute_descale_params(
        cfg.TRAIN_RAW, cfg.TRAIN_SCALED, cfg.TARGET_COL
    )

    # ── Split Train / Validation (Time-Based, pas de shuffle !) ──
    n_total = len(Y_train_full)
    n_val   = int(n_total * cfg.VAL_RATIO)
    n_train = n_total - n_val

    X_train, X_val = X_train_full[:n_train], X_train_full[n_train:]
    Y_train, Y_val = Y_train_full[:n_train], Y_train_full[n_train:]

    print(f"\n  Split temporel → Train : {n_train:,} h | Val : {n_val:,} h")

    # ── Datasets & DataLoaders ──
    print("\n🔧 Création des DataLoaders (lookback = {})...".format(cfg.LOOKBACK))

    train_dataset = ElectricityWindowDataset(X_train, Y_train, cfg.LOOKBACK)
    val_dataset   = ElectricityWindowDataset(X_val, Y_val, cfg.LOOKBACK)
    test_dataset  = ElectricityWindowDataset(X_test, Y_test, cfg.LOOKBACK)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False  # séries temporelles !
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False
    )

    print(f"  Séquences Train : {len(train_dataset):,}")
    print(f"  Séquences Val   : {len(val_dataset):,}")
    print(f"  Séquences Test  : {len(test_dataset):,}")

    # ── Modèle ──
    model = SpotPriceLSTM(
        input_size=n_features,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Modèle initialisé : {total_params:,} paramètres")
    print(model)

    # ── Entraînement ──
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, cfg, device
    )

    # Sauvegarde du modèle
    torch.save(model.state_dict(), cfg.MODEL_PATH)
    print(f"\n💾 Modèle sauvegardé → {cfg.MODEL_PATH}")

    # ── Inférence sur le Test 2024 ──
    print("\n🔮 Inférence sur le Test 2024...")
    preds_scaled = predict(model, test_loader, device)

    # Les vraies valeurs Y du test (alignées avec les prédictions)
    # Le dataset retourne Y[lookback:], donc on aligne
    y_true_scaled = Y_test[cfg.LOOKBACK:]

    # ── Dé-standardisation (retour en €/MWh) ──
    y_true_eur = inverse_scale(y_true_scaled, target_mu, target_std)
    y_pred_eur = inverse_scale(preds_scaled, target_mu, target_std)

    # Timestamps alignés (on perd les `lookback` premières heures)
    test_timestamps = test_meta["HourUTC"].iloc[cfg.LOOKBACK:].reset_index(drop=True)

    # ── Évaluation finale ──
    descale_label = "dé-standardisé" if target_mu is not None else "brut — pas de descaling"
    metrics = evaluate(y_true_eur, y_pred_eur, label=f"Test 2024 ({descale_label})")

    # ── Graphiques ──
    print("\n📊 Génération des graphiques...")
    plot_training_curves(train_losses, val_losses, cfg.OUTPUT_DIR)
    plot_predictions(test_timestamps, y_true_eur, y_pred_eur, cfg.OUTPUT_DIR)
    plot_scatter(y_true_eur, y_pred_eur, cfg.OUTPUT_DIR)

    # ── Export CSV des prédictions ──
    results_df = pd.DataFrame({
        "HourUTC": test_timestamps.values,
        "PriceArea": test_meta["PriceArea"].iloc[cfg.LOOKBACK:].values,
        "SpotPriceEUR_real": y_true_eur,
        "SpotPriceEUR_pred_LSTM": y_pred_eur,
        "Error_EUR": y_pred_eur - y_true_eur,
    })
    csv_path = os.path.join(cfg.OUTPUT_DIR, "lstm_predictions_2024.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  → Prédictions exportées : {csv_path}")

    # ── Résumé par zone ──
    print("\n📋 Métriques par zone de marché :")
    for zone in results_df["PriceArea"].unique():
        mask = results_df["PriceArea"] == zone
        sub = results_df[mask]
        z_mae  = mean_absolute_error(sub["SpotPriceEUR_real"], sub["SpotPriceEUR_pred_LSTM"])
        z_rmse = np.sqrt(mean_squared_error(sub["SpotPriceEUR_real"], sub["SpotPriceEUR_pred_LSTM"]))
        z_r2   = r2_score(sub["SpotPriceEUR_real"], sub["SpotPriceEUR_pred_LSTM"])
        print(f"  {zone} → MAE: {z_mae:.2f} | RMSE: {z_rmse:.2f} | R²: {z_r2:.4f}")

    print("\n✅ Pipeline LSTM terminé avec succès !")
    print(f"   Tous les résultats dans : {cfg.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()