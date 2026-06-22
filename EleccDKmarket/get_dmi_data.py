import requests
import pandas as pd
import time
import os

os.makedirs('raw_data', exist_ok=True)
# CONFIGURATION
ENDPOINT = "https://opendataapi.dmi.dk/v2/metObs/collections/observation/items"

# Stations stratégiques : Esbjerg (DK1 / Ouest) et Copenhague (DK2 / Est)
STATIONS = {
    '06080': 'DK1',
    '06180': 'DK2'
}

params_list = ['wind_speed', 'wind_dir', 'temp_dry']
all_station_data = []

print("🚀 Lancement de la collecte Open Data DMI (Accès Public Global)...")

for station_id, zone in STATIONS.items():
    print(f"\n📡 Extraction des données pour la station {station_id} (Zone {zone})...")
    
    # Liste pour accumuler les données de cette station spécifique
    station_records = []
    
    for param in params_list:
        print(f"   -> Téléchargement du paramètre : {param}...")
        
        query_params = {
            'stationId': station_id,
            'parameterId': param,  # Un seul paramètre à la fois pour éviter la 400
            'datetime': '2022-01-01T00:00:00Z/2024-12-31T23:00:00Z',
            'limit': 300000  # ~26 300 heures sur 3 ans, 40k suffit pour 1 paramètre
        }
        
        response = requests.get(ENDPOINT, params=query_params)
        
        if response.status_code == 200:
            json_data = response.json()
            features = json_data.get('features', [])
            
            for feat in features:
                prop = feat['properties']
                station_records.append({
                    'HourUTC': prop['observed'],
                    'parameter': prop['parameterId'],
                    'value': prop['value']
                })
            
            # Petite pause de courtoisie pour l'API
            time.sleep(0.1)
        else:
            print(f"❌ Erreur {response.status_code} sur {param} (Station {station_id})")

    # Une fois les 3 paramètres récupérés pour la station, on pivote
    if station_records:
        df_station = pd.DataFrame(station_records)
        
        df_pivot = df_station.pivot_table(
            index='HourUTC', 
            columns='parameter', 
            values='value', 
            aggfunc='mean'
        ).reset_index()
        
        df_pivot['PriceArea'] = zone
        df_pivot['Station_ID'] = station_id
        all_station_data.append(df_pivot)
        print(f"✅ Station {station_id} traitée : {df_pivot.shape[0]} heures structurées.")

# Assemblage et sauvegarde finale
if all_station_data:
    df_dmi_final = pd.concat(all_station_data, ignore_index=True)
    
    # Harmonisation du format de date
    df_dmi_final['HourUTC'] = pd.to_datetime(df_dmi_final['HourUTC']).dt.tz_localize(None)
    df_dmi_final = df_dmi_final.sort_values(['HourUTC', 'PriceArea']).reset_index(drop=True)
    
    df_dmi_final.to_csv('raw_data/dmi_weather_raw.csv', index=False)
    print(f"\n📊 Fichier enregistré avec succès : 'raw_data/dmi_weather_raw.csv' ({df_dmi_final.shape[0]} lignes).")
else:
    print("❌ Échec global de la collecte.")