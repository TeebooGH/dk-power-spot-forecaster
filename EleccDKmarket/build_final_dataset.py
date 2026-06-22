import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

print("⏳ ÉTAPE 1 : Chargement et Fusion des datasets propres...")
df_weather = pd.read_csv('clean_data/dmi_weather_clean.csv')
df_energinet = pd.read_csv('clean_data/energinet_clean.csv')

# Sécurité Datetime
df_weather['HourUTC'] = pd.to_datetime(df_weather['HourUTC'])
df_energinet['HourUTC'] = pd.to_datetime(df_energinet['HourUTC'])

# Fusion par intersection stricte (Inner Join)
# On supprime Station_ID de la météo qui est une redondance de PriceArea
df_weather = df_weather.drop(columns=['Station_ID'], errors='ignore')
df_master = pd.merge(df_energinet, df_weather, on=['HourUTC', 'PriceArea'], how='inner')

# Tri chronologique obligatoire pour les étapes de fenêtres glissantes
df_master = df_master.sort_values(['HourUTC', 'PriceArea']).reset_index(drop=True)

# ==========================================================
# ÉTAPE 2 : FEATURE ENGINEERING AVANCÉ (Lags & Rolling)
# ==========================================================
print("🔄 ÉTAPE 2 : Génération des Lags et Rolling Features par zone...")

# Variables clés à temporiser
lag_features = ['SpotPriceEUR', 'Brute_Consumption_MWh', 'wind_speed', 'temp_dry']

for col in lag_features:
    # Lags classiques (1h, 24h pour la veille, 168h pour la semaine dernière)
    df_master[f'{col}_lag_1'] = df_master.groupby('PriceArea')[col].shift(1)
    df_master[f'{col}_lag_24'] = df_master.groupby('PriceArea')[col].shift(24)
    df_master[f'{col}_lag_168'] = df_master.groupby('PriceArea')[col].shift(168)
    
    # Moyenne glissante sur 6h pour capter l'inertie (Ex: tendance météo ou rampe de conso)
    # closed='left' est CRUCIAL pour éviter de voir le futur proche au moment de la prédiction
    df_master[f'{col}_roll_mean_6h'] = df_master.groupby('PriceArea')[col].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=1).mean()
    )

# Nettoyage des lignes instables générées par le lag de 168h (première semaine de 2022)
df_master = df_master.dropna().reset_index(drop=True)

# ==========================================================
# ÉTAPE 3 : PROCESSING & ENCODING DES TEXTES
# ==========================================================
print("🔢 ÉTAPE 3 : Encodage des variables catégorielles...")
# 'DK1' devient 0, 'DK2' devient 1 (Format parfait pour XGBoost et les réseaux de neurones)
df_master['PriceArea_Code'] = df_master['PriceArea'].map({'DK1': 0, 'DK2': 1})

# Sauvegarde des colonnes temporelles et d'index pour le tracking final
timestamps = df_master['HourUTC']
zones = df_master['PriceArea']

# Définition de la cible (Y) et des features (X)
target_col = 'SpotPriceEUR'
# On exclut les colonnes cibles, les textes originaux et les index temporels du X
exclude_cols = ['HourUTC', 'PriceArea', 'SpotPriceEUR', 'SpotPriceDKK']
feature_cols = [col for col in df_master.columns if col not in exclude_cols]

X = df_master[feature_cols].copy()
y = df_master[target_col].copy()

# ==========================================================
# ÉTAPE 4 : TIME-BASED SPLIT ÉTANCHE (Pas de shuffle !)
# ==========================================================
print("📅 ÉTAPE 4 : Séparation temporelle Train (2022-2023) / Test (2024)...")
train_mask = timestamps < pd.to_datetime('2024-01-01 00:00:00')

X_train_raw, X_test_raw = X[train_mask].copy(), X[~train_mask].copy()
y_train, y_test = y[train_mask].copy(), y[~train_mask].copy()

timestamps_train, timestamps_test = timestamps[train_mask], timestamps[~train_mask]
zones_train, zones_test = zones[train_mask], zones[~train_mask]

# ==========================================================
# ÉTAPE 5 : STANDARDISATION SÉCURISÉE (Anti-Leakage)
# ==========================================================
print("⚖️  ÉTAPE 5 : Standardisation des données (Spécial LSTM)...")
scaler = StandardScaler()

# On apprend la moyenne et l'écart-type UNIQUEMENT sur le Train
X_train_scaled_values = scaler.fit_transform(X_train_raw)
# On applique sans réapprendre sur le Test
X_test_scaled_values = scaler.transform(X_test_raw)

# Reconstruction des DataFrames standardisés
X_train_scaled = pd.DataFrame(X_train_scaled_values, columns=feature_cols, index=X_train_raw.index)
X_test_scaled = pd.DataFrame(X_test_scaled_values, columns=feature_cols, index=X_test_raw.index)

# ==========================================================
# ÉTAPE 6 : EXPORTATION VERS ./FINAL_DATA
# ==========================================================
print("💾 ÉTAPE 6 : Exportation des datasets finaux de production...")
output_dir = './final_data'
os.makedirs(output_dir, exist_ok=True)

# Reconstitution des fichiers CSV complets pour l'utilisateur
# Version RAW (Idéale pour XGBoost / LightGBM)
train_raw_final = pd.concat([timestamps_train, zones_train, y_train, X_train_raw], axis=1)
test_raw_final = pd.concat([timestamps_test, zones_test, y_test, X_test_raw], axis=1)

train_raw_final.to_csv(os.path.join(output_dir, 'train_raw.csv'), index=False)
test_raw_final.to_csv(os.path.join(output_dir, 'test_raw.csv'), index=False)

# Version SCALED (Obligatoire pour ton LSTM)
train_scaled_final = pd.concat([timestamps_train, zones_train, y_train, X_train_scaled], axis=1)
test_scaled_final = pd.concat([timestamps_test, zones_test, y_test, X_test_scaled], axis=1)

train_scaled_final.to_csv(os.path.join(output_dir, 'train_scaled.csv'), index=False)
test_scaled_final.to_csv(os.path.join(output_dir, 'test_scaled.csv'), index=False)

print("\n" + "="*50)
print("=== PIPELINE DE TRAITEMENT TERMINÉ AVEC SUCCÈS ===")
print("="*50)
print(f"Dataset de TRAIN (2022-2023) : {train_raw_final.shape[0]} lignes | {train_raw_final.shape[1]} colonnes")
print(f"Dataset de TEST  (2024)      : {test_raw_final.shape[0]} lignes | {test_raw_final.shape[1]} colonnes")
print(f"Features incluses ({len(feature_cols)}) : {feature_cols}")
print(f"💾 Fichiers disponibles dans : {output_dir}/")
print("   -> train_raw.csv & test_raw.csv     (Pour XGBoost)")
print("   -> train_scaled.csv & test_scaled.csv (Pour LSTM)")
print("="*50)
print("🚀 Vos données sont 100% qualifiées pour l'entraînement de vos modèles.")