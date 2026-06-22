import os
import pandas as pd

# Définition stricte du dossier contenant tes fichiers
DATA_DIR = "/home/jean-marie/dk-power-spot-forecaster/EleccDKmarket/raw_data"

print(
    f"⏳ Lecture et assemblage des données brutes depuis le dossier : {DATA_DIR}"
)

# Reconstruction des chemins absolus pour éviter les FileNotFoundError
path_cons = os.path.join(DATA_DIR, "ConsumptionDK3619IndustryHour.csv")
path_spot = os.path.join(DATA_DIR, "Elspotprices.csv")
path_realtime = os.path.join(DATA_DIR, "RealtimeMarket.csv")

# ==========================================
# 1. LECTURE DE LA CONSOMMATION
# ==========================================
print("[1/3] Lecture de la Consommation...")
# On force le séparateur ';' et la décimale ',' pour transformer le texte en nombres (float)
df_cons = pd.read_csv(path_cons, sep=";", decimal=",")

# Agrégation mécanique par heure : on regroupe les miettes sectorielles
df_cons_brute = (
    df_cons.groupby("TimeUTC")["Consumption_MWh"].sum().reset_index()
)
df_cons_brute = df_cons_brute.rename(
    columns={"TimeUTC": "HourUTC", "Consumption_MWh": "Brute_Consumption_MWh"}
)


# ==========================================
# 2. LECTURE DES PRIX SPOT (Filtre zones d'étude)
# ==========================================
print("[2/3] Lecture des Prix Spot...")
df_spot = pd.read_csv(path_spot, sep=";", decimal=",")
df_spot_brute = df_spot[df_spot["PriceArea"].isin(["DK1", "DK2"])].copy()


# ==========================================
# 3. LECTURE DU MARCHÉ TEMPS RÉEL (Filtre zones d'étude)
# ==========================================
print("[3/3] Lecture du Marché Temps Réel...")
df_realtime = pd.read_csv(path_realtime, sep=";", decimal=",")
df_realtime_brute = df_realtime[
    df_realtime["PriceArea"].isin(["DK1", "DK2"])
].copy()


# ==========================================
# FUSION BRUTE (CONSERVATION DES NaNs POUR TON EDA)
# ==========================================
print("🔀 Fusion des fichiers sur HourUTC et PriceArea...")

# Sélection des colonnes cibles demandées pour ton analyse
cols_spot = ["HourUTC", "PriceArea", "SpotPriceEUR", "SpotPriceDKK"]
cols_realtime = [
    "HourUTC",
    "PriceArea",
    "mFRRUpActBal",
    "mFRRDownActBal",
    "mFRRUpActSpec",
    "mFRRDownActSpec",
    "ImbalanceMWh",
    "ImbalancePriceEUR",
    "BalancingPowerPriceUpEUR",
    "BalancingPowerPriceDownEUR",
]

# Fusion avec 'outer' pour conserver l'intégralité des lignes brutes (sans perte s'il y a un décalage)
df_merged = pd.merge(
    df_spot_brute[cols_spot],
    df_realtime_brute[cols_realtime],
    on=["HourUTC", "PriceArea"],
    how="outer",
)

# Assemblage final avec la courbe de charge
df_gros_csv = pd.merge(df_merged, df_cons_brute, on="HourUTC", how="left")

# Tri chronologique propre pour faciliter tes graphiques et observations
df_gros_csv = df_gros_csv.sort_values(by=["HourUTC", "PriceArea"]).reset_index(
    drop=True
)


# ==========================================
# EXPORTATION DANS LE DOSSIER RAW_DATA
# ==========================================
output_path = os.path.join(DATA_DIR, "gros_dataset_brut.csv")
df_gros_csv.to_csv(output_path, index=False)

print(f"\n🎉 Succès ! Le gros fichier brut est disponible : '{output_path}'")
print(
    f"📊 Structure obtenue : {df_gros_csv.shape[0]} lignes et {df_gros_csv.shape[1]} colonnes."
)