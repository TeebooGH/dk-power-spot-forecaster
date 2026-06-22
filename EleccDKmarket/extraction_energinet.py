"""
Extraction Energi Data Service — v3
====================================
Datasets par période :
  - Prix spot     : Elspotprices (≤2024) → DayAheadPrices (≥2025)
  - Régulation    : RealtimeMarket (≤2024) → ImbalancePrice + MfrrEnergyActivation (≥2025)
  - Consommation  : ConsumptionDK3619IndustryHour (toutes périodes)

Le script peut fonctionner en 2 modes :
  1. MODE API   : télécharge via l'API (lent à cause du rate limiting ~2min/requête)
  2. MODE CSV   : lit des fichiers CSV déjà téléchargés manuellement depuis le site

Usage :
  python extraction_energinet.py --mode api
  ppython extraction_energinet.py --mode csv --csv-dir ./raw_data
"""

import os
import sys
import time
import json
import re
import argparse
import pandas as pd
import requests
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────
YEARS = [2022, 2023, 2024]
ZONES = ["DK1", "DK2"]
OUTPUT_DIR = "raw_data"
FINAL_OUTPUT = "dataset_final.csv"
DELAY_BETWEEN_REQUESTS = 8   # secondes de base entre requêtes
MAX_RETRIES = 3

# Mapping des datasets selon la période
# Clé = nom logique, valeur = {dataset_name, time_col, has_price_area, columns_map}
DATASET_CONFIG = {
    "spot": {
        # Elspotprices pour 2022-2024 (données historiques)
        "pre2025": {
            "dataset": "Elspotprices",
            "time_col": "HourDK",
            "time_col_utc": "HourUTC",
            "has_price_area": True,
            "price_col": "SpotPriceEUR",
        },
        # DayAheadPrices pour 2025+
        "post2025": {
            "dataset": "DayAheadPrices",
            "time_col": "TimeDK",
            "time_col_utc": "TimeUTC",
            "has_price_area": True,
            "price_col": "DayAheadPriceEUR",
        },
    },
    "regulation": {
        # RealtimeMarket pour 2022-2024
        "pre2025": {
            "dataset": "RealtimeMarket",
            "time_col": "HourDK",
            "time_col_utc": "HourUTC",
            "has_price_area": True,
        },
        # ImbalancePrice pour 2025+ (on pourrait aussi ajouter MfrrEnergyActivationMarket)
        "post2025": {
            "dataset": "ImbalancePrice",
            "time_col": "TimeDK",
            "time_col_utc": "TimeUTC",
            "has_price_area": True,
        },
    },
    "consumption": {
        # Même dataset pour toutes les périodes (pas de PriceArea)
        "pre2025": {
            "dataset": "ConsumptionDK3619IndustryHour",
            "time_col": "TimeDK",
            "time_col_utc": "TimeUTC",
            "has_price_area": False,
        },
        "post2025": {
            "dataset": "ConsumptionDK3619IndustryHour",
            "time_col": "TimeDK",
            "time_col_utc": "TimeUTC",
            "has_price_area": False,
        },
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# MODE API
# ══════════════════════════════════════════════════════════════════════

def fetch_with_retry(url, params, retries=MAX_RETRIES):
    """GET avec retry automatique sur 429."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=90)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = 30 * attempt
                try:
                    msg = resp.text
                    m = re.search(r'(\d+)\s*second', msg.lower())
                    if m:
                        wait = int(m.group(1)) + 5
                except Exception:
                    pass
                print(f"      ⏳ 429 — attente {wait}s (tentative {attempt}/{retries})...")
                time.sleep(wait)
                continue
            print(f"      ❌ HTTP {resp.status_code} — {resp.text[:200]}")
            return None
        except requests.exceptions.Timeout:
            print(f"      ⏳ Timeout (tentative {attempt}/{retries})...")
            time.sleep(15 * attempt)
        except Exception as e:
            print(f"      ❌ Erreur réseau : {e}")
            return None
    print(f"      ❌ Échec après {retries} tentatives.")
    return None


def api_fetch_dataset(dataset_name, years, has_price_area=True):
    """Télécharge un dataset mois par mois via l'API."""
    print(f"\n   📥 API → {dataset_name}")

    # Inspection rapide
    url = f"https://api.energidataservice.dk/dataset/{dataset_name}"
    try:
        r = requests.get(url, params={"limit": 1}, timeout=15)
        if r.status_code == 200:
            recs = r.json().get("records", [])
            if recs:
                print(f"      Colonnes : {list(recs[0].keys())}")
        elif r.status_code == 404:
            print(f"      ❌ Dataset '{dataset_name}' introuvable (404)")
            return None
    except Exception:
        pass
    time.sleep(DELAY_BETWEEN_REQUESTS)

    all_chunks = []
    for year in years:
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            if month == 12:
                end = f"{year + 1}-01-01"
            else:
                end = f"{year}-{month + 1:02d}-01"

            label = f"{year}-{month:02d}"
            print(f"      {label}...", end=" ", flush=True)

            params = {"start": start, "end": end, "limit": 0}
            if has_price_area:
                params["filter"] = json.dumps({"PriceArea": ZONES})

            resp = fetch_with_retry(url, params)
            if resp is None:
                print("SKIP")
                time.sleep(DELAY_BETWEEN_REQUESTS)
                continue

            records = resp.json().get("records", [])
            if not records:
                print("(vide)")
            else:
                all_chunks.append(pd.DataFrame(records))
                print(f"✅ {len(records)}")

            time.sleep(DELAY_BETWEEN_REQUESTS)

    if all_chunks:
        df = pd.concat(all_chunks, ignore_index=True)
        print(f"      TOTAL : {len(df)} lignes")
        return df
    return None


def run_api_mode():
    """Télécharge tous les datasets via l'API."""
    # On sépare les années pré et post 2025
    years_pre = [y for y in YEARS if y < 2025]
    years_post = [y for y in YEARS if y >= 2025]

    results = {}

    for logical_name, configs in DATASET_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"📦 {logical_name.upper()}")
        print(f"{'='*60}")

        dfs = []
        if years_pre:
            cfg = configs["pre2025"]
            df = api_fetch_dataset(cfg["dataset"], years_pre, cfg["has_price_area"])
            if df is not None:
                dfs.append(df)

        if years_post:
            cfg = configs["post2025"]
            df = api_fetch_dataset(cfg["dataset"], years_post, cfg["has_price_area"])
            if df is not None:
                dfs.append(df)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            outfile = f"{OUTPUT_DIR}/raw_{logical_name}.csv"
            combined.to_csv(outfile, index=False)
            print(f"   💾 {outfile} ({len(combined)} lignes)")
            results[logical_name] = combined
        else:
            print(f"   ⚠️  Aucune donnée pour {logical_name}")

    return results


# ══════════════════════════════════════════════════════════════════════
# MODE CSV (fichiers téléchargés manuellement)
# ══════════════════════════════════════════════════════════════════════

def read_european_csv(filepath):
    """
    Lit un CSV au format européen (séparateur ; décimale ,)
    tel que téléchargé depuis energidataservice.dk
    """
    print(f"   📄 Lecture de {filepath}...")
    # Tester d'abord le séparateur
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline()

    if ";" in header:
        df = pd.read_csv(filepath, sep=";", decimal=",", encoding="utf-8")
    else:
        df = pd.read_csv(filepath, encoding="utf-8")

    print(f"      {len(df)} lignes × {len(df.columns)} colonnes")
    print(f"      Colonnes : {list(df.columns)}")
    return df


def run_csv_mode(csv_dir):
    """Lit les CSV déjà téléchargés depuis le répertoire spécifié."""
    results = {}

    # Mapping nom de fichier → nom logique
    # On cherche les fichiers de manière flexible
    file_mapping = {
        "spot": ["Elspotprices", "DayAheadPrices", "spot"],
        "regulation": ["RealtimeMarket", "Imbalance", "realtime", "regulation"],
        "consumption": ["Consumption", "consumption"],
    }

    csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith(".csv")]
    print(f"\n   Fichiers CSV trouvés dans {csv_dir} : {csv_files}")

    for logical_name, patterns in file_mapping.items():
        matched = None
        for f in csv_files:
            for pat in patterns:
                if pat.lower() in f.lower():
                    matched = f
                    break
            if matched:
                break

        if matched:
            filepath = os.path.join(csv_dir, matched)
            df = read_european_csv(filepath)
            results[logical_name] = df
        else:
            print(f"   ⚠️  Pas de fichier trouvé pour '{logical_name}' "
                  f"(cherché : {patterns})")

    return results


# ══════════════════════════════════════════════════════════════════════
# NORMALISATION & MERGE
# ══════════════════════════════════════════════════════════════════════

def normalize_spot(df):
    """Normalise le DataFrame spot (Elspotprices ou DayAheadPrices)."""
    df = df.copy()

    # Unifier le nom de la colonne temps
    if "HourDK" in df.columns:
        df.rename(columns={"HourDK": "TimeDK"}, inplace=True)
    if "HourUTC" in df.columns:
        df.rename(columns={"HourUTC": "TimeUTC"}, inplace=True)

    # Unifier le nom de la colonne prix
    if "SpotPriceEUR" in df.columns:
        df.rename(columns={"SpotPriceEUR": "PriceEUR"}, inplace=True)
    elif "DayAheadPriceEUR" in df.columns:
        df.rename(columns={"DayAheadPriceEUR": "PriceEUR"}, inplace=True)

    # Supprimer les colonnes DKK (on garde EUR)
    dkk_cols = [c for c in df.columns if "DKK" in c]
    df.drop(columns=dkk_cols, errors="ignore", inplace=True)

    df["TimeDK"] = pd.to_datetime(df["TimeDK"])

    # Filtrer sur DK1/DK2
    if "PriceArea" in df.columns:
        df = df[df["PriceArea"].isin(ZONES)]

    # Garder les colonnes utiles
    keep = ["TimeDK", "PriceArea", "PriceEUR"]
    keep = [c for c in keep if c in df.columns]
    if "TimeUTC" in df.columns:
        keep.insert(0, "TimeUTC")

    return df[keep].sort_values(["TimeDK", "PriceArea"]).reset_index(drop=True)


def normalize_regulation(df):
    """Normalise le DataFrame régulation (RealtimeMarket ou ImbalancePrice)."""
    df = df.copy()

    # Unifier les colonnes temps
    if "HourDK" in df.columns:
        df.rename(columns={"HourDK": "TimeDK"}, inplace=True)
    if "HourUTC" in df.columns:
        df.rename(columns={"HourUTC": "TimeUTC"}, inplace=True)

    df["TimeDK"] = pd.to_datetime(df["TimeDK"])

    # Filtrer sur DK1/DK2
    if "PriceArea" in df.columns:
        df = df[df["PriceArea"].isin(ZONES)]

    # Supprimer les colonnes DKK et IGCC (bruit)
    drop_cols = [c for c in df.columns if "DKK" in c or "IGCC" in c]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # Si résolution infra-horaire, agréger à l'heure
    df["HourDK"] = df["TimeDK"].dt.floor("h")
    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Prix → moyenne, Volumes → somme
    price_cols = [c for c in num_cols if "Price" in c or "EUR" in c]
    vol_cols = [c for c in num_cols if c not in price_cols]

    agg_dict = {c: "mean" for c in price_cols}
    agg_dict.update({c: "sum" for c in vol_cols})

    group_cols = ["HourDK", "PriceArea"] if "PriceArea" in df.columns else ["HourDK"]

    if agg_dict:
        df = df.groupby(group_cols).agg(agg_dict).reset_index()
    else:
        df = df.groupby(group_cols).first().reset_index()

    df.rename(columns={"HourDK": "TimeDK"}, inplace=True)
    sort_cols = ["TimeDK"] + (["PriceArea"] if "PriceArea" in df.columns else [])
    return df.sort_values(sort_cols).reset_index(drop=True)


def normalize_consumption(df):
    """
    Normalise le DataFrame consommation.
    ConsumptionDK3619IndustryHour n'a pas de PriceArea →
    on agrège la conso totale par heure (somme de tous les secteurs).
    """
    df = df.copy()

    # Unifier les colonnes temps
    if "HourDK" in df.columns:
        df.rename(columns={"HourDK": "TimeDK"}, inplace=True)
    if "HourUTC" in df.columns:
        df.rename(columns={"HourUTC": "TimeUTC"}, inplace=True)

    df["TimeDK"] = pd.to_datetime(df["TimeDK"])

    # Trouver la colonne de consommation
    cons_col = next((c for c in df.columns if "consumption" in c.lower()), None)
    if cons_col is None:
        print("   ⚠️  Aucune colonne 'Consumption' trouvée !")
        return None

    # S'assurer que la colonne est numérique
    df[cons_col] = pd.to_numeric(df[cons_col], errors="coerce")

    # Agréger : somme de tous les secteurs DK36 par heure
    df_agg = df.groupby("TimeDK")[cons_col].sum().reset_index()
    df_agg.rename(columns={cons_col: "TotalConsumption_MWh"}, inplace=True)

    return df_agg.sort_values("TimeDK").reset_index(drop=True)


def merge_final(results):
    """Merge tous les datasets normalisés en un seul DataFrame."""
    print(f"\n{'='*60}")
    print("🔗 NORMALISATION & MERGE")
    print(f"{'='*60}")

    # 1. Spot prices
    if "spot" not in results or results["spot"] is None:
        print("❌ Pas de données spot → impossible de construire le dataset final.")
        return None
    df_spot = normalize_spot(results["spot"])
    print(f"   ✅ Spot normalisé : {len(df_spot)} lignes")

    # 2. Regulation
    df_reg = None
    if "regulation" in results and results["regulation"] is not None:
        df_reg = normalize_regulation(results["regulation"])
        print(f"   ✅ Régulation normalisée : {len(df_reg)} lignes")

    # 3. Consumption
    df_cons = None
    if "consumption" in results and results["consumption"] is not None:
        df_cons = normalize_consumption(results["consumption"])
        if df_cons is not None:
            print(f"   ✅ Consommation normalisée : {len(df_cons)} lignes")

    # ── Merge ──
    df_final = df_spot.copy()

    if df_reg is not None:
        merge_keys = ["TimeDK", "PriceArea"] if "PriceArea" in df_reg.columns else ["TimeDK"]
        df_final = df_final.merge(df_reg, on=merge_keys, how="left", suffixes=("", "_reg"))
        print(f"   🔗 Après merge régulation : {len(df_final)} lignes × {len(df_final.columns)} col")

    if df_cons is not None:
        # Consommation = pas de PriceArea, merge sur TimeDK uniquement
        # Chaque zone DK1/DK2 aura la même conso nationale
        df_final = df_final.merge(df_cons, on="TimeDK", how="left")
        print(f"   🔗 Après merge consommation : {len(df_final)} lignes × {len(df_final.columns)} col")

    # Supprimer les doublons de colonnes TimeUTC
    utc_dupes = [c for c in df_final.columns if c.startswith("TimeUTC") and c != "TimeUTC"]
    df_final.drop(columns=utc_dupes, errors="ignore", inplace=True)

    # Tri final
    df_final.sort_values(["TimeDK", "PriceArea"], inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    return df_final


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction Energi Data Service")
    parser.add_argument("--mode", choices=["api", "csv"], default="csv",
                        help="'api' = télécharge via l'API, 'csv' = lit des fichiers locaux")
    parser.add_argument("--csv-dir", default=".",
                        help="Répertoire contenant les CSV (mode csv)")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS,
                        help="Années à télécharger (mode api)")
    args = parser.parse_args()

    YEARS = args.years
    print("🚀 EXTRACTION ENERGI DATA SERVICE")
    print(f"   Mode   : {args.mode}")
    print(f"   Années : {YEARS}")
    print(f"   Zones  : {ZONES}")

    # ── Extraction ──
    if args.mode == "api":
        results = run_api_mode()
    else:
        results = run_csv_mode(args.csv_dir)

    if not results:
        print("\n❌ Aucune donnée extraite. Fin.")
        sys.exit(1)

    # ── Merge ──
    df_final = merge_final(results)

    if df_final is None:
        sys.exit(1)

    # ── Sauvegarde ──
    df_final.to_csv(FINAL_OUTPUT, index=False)

    print(f"\n{'='*60}")
    print(f"🏁 TERMINÉ")
    print(f"{'='*60}")
    print(f"   📊 Fichier  : {FINAL_OUTPUT}")
    print(f"   📐 Taille   : {len(df_final)} lignes × {len(df_final.columns)} colonnes")
    print(f"   📅 Période  : {df_final['TimeDK'].min()} → {df_final['TimeDK'].max()}")
    print(f"   📋 Colonnes : {list(df_final.columns)}")

    # Stats rapides
    print(f"\n   Valeurs manquantes :")
    missing = df_final.isnull().sum()
    has_missing = missing[missing > 0]
    if len(has_missing) == 0:
        print(f"      Aucune !")
    else:
        for col in has_missing.index:
            pct = 100 * has_missing[col] / len(df_final)
            print(f"      {col}: {has_missing[col]} ({pct:.1f}%)")

    # Aperçu
    print(f"\n   Aperçu (5 premières lignes) :")
    print(df_final.head().to_string(index=False))