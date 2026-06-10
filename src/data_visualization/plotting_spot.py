import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from scipy.stats import kurtosis
from statsmodels.graphics.tsaplots import plot_acf

# ==========================================
# 1. ACQUISITION ET PRÉPARATION DES DONNÉES
# ==========================================
print("Téléchargement des données via l'API d'Energinet...")

url = "https://api.energidataservice.dk/dataset/Elspotprices"
# 15000 heures couvrent environ 300 jours pour 2 zones, ce qui est parfait.
params = {
    "limit": 15000,
    "filter": '{"PriceArea":["DK1", "DK2"]}',
    "sort": "HourUTC DESC",
}

response = requests.get(url, params=params)
data = response.json()["records"]

# Conversion en DataFrame Pandas
df = pd.DataFrame(data)
df["HourUTC"] = pd.to_datetime(df["HourUTC"])
df = df[["HourUTC", "PriceArea", "SpotPriceEUR"]].dropna()

# Pivot pour aligner temporellement DK1 et DK2 sur chaque ligne
df_pivot = df.pivot(
    index="HourUTC", columns="PriceArea", values="SpotPriceEUR"
).sort_index()

# 1. Ensure the index is datetime
df_pivot.index = pd.to_datetime(df_pivot.index)

# 2. Use the 'last' approach via filtering the index directly
# This avoids doing math on the index object itself
lookback_date = df_pivot.index.max() - pd.Timedelta(days=90)
df_3m = df_pivot[df_pivot.index > lookback_date].copy()

# Configuration esthétique pour les slides
sns.set_theme(style="darkgrid", context="talk")

# ==========================================
# 2. GRAPHIQUE 1 : DISTRIBUTION ET QUEUES ÉPAISSES
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(df_3m["DK1"], bins=100, kde=True, color="darkblue", stat="density")

# Calcul du kurtosis pour appuyer ton argumentaire
k_val = kurtosis(df_3m["DK1"].dropna())
mean_val = df_3m["DK1"].mean()

plt.axvline(mean_val, color="red", linestyle="--", label=f"Moyenne ({mean_val:.2f} €)")
plt.title(
    f"Distribution des Prix Spot (DK1) - Preuve de Non-Normalité\nKurtosis: {k_val:.2f} (Leptokurtique)",
    fontweight="bold",
)
plt.xlabel("Prix Spot (EUR/MWh)")
plt.ylabel("Densité")
plt.legend()
plt.tight_layout()
plt.savefig("plot_1_distribution.png", dpi=300)
plt.show()

# ==========================================
# 3. GRAPHIQUE 2 : AUTOCORRÉLATION (LAGS 1, 24, 168)
# ==========================================
plt.figure(figsize=(12, 5))
# On trace l'ACF sur 170 heures pour bien voir le pic de 168h (1 semaine)
plot_acf(df_3m["DK1"].dropna(), lags=170, ax=plt.gca(), color="teal", alpha=0.05)

# Mise en évidence des lags clés
plt.axvline(24, color="orange", linestyle="--", alpha=0.7, label="Lag 24h (Jour)")
plt.axvline(168, color="purple", linestyle="--", alpha=0.7, label="Lag 168h (Semaine)")

plt.title(
    "Fonction d'Autocorrélation (ACF) - Justification des Features AR",
    fontweight="bold",
)
plt.xlabel("Retard (Heures)")
plt.ylabel("Corrélation")
plt.legend()
plt.tight_layout()
plt.savefig("plot_2_autocorrelation.png", dpi=300)
plt.show()

# ==========================================
# 4. GRAPHIQUE 3 : SPREAD DK1-DK2 (RETOUR À LA MOYENNE)
# ==========================================
df_3m["Spread"] = df_3m["DK1"] - df_3m["DK2"]

plt.figure(figsize=(12, 6))
plt.plot(df_3m.index, df_3m["Spread"], color="crimson", linewidth=1)
plt.axhline(0, color="black", linestyle="-", linewidth=1.5)

plt.title(
    "Dynamique du Spread DK1 - DK2 (Processus d'Ornstein-Uhlenbeck)", fontweight="bold"
)
plt.xlabel("Date (Heure UTC)")
plt.ylabel("Différentiel de Prix (EUR/MWh)")
plt.tight_layout()
plt.savefig("plot_3_spread.png", dpi=300)
plt.show()

print(
    "Génération terminée. Les trois graphiques ont été sauvegardés en PNG haute résolution dans le dossier courant."
)
