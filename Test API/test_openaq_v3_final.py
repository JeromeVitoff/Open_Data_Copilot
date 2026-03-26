"""
Test OpenAQ v3 - Endpoints corrects d'après doc
"""
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv('OPENAQ_API_KEY')
base_url = "https://api.openaq.org/v3"

headers = {'X-API-Key': api_key}

print("="*70)
print("OPENAQ V3 - ENDPOINTS CORRECTS")
print("="*70)

# Test 1 : Latest PM2.5 (parameter ID 2)
print("\n1️⃣ Latest PM2.5 values (worldwide)")
print("-" * 70)

try:
    response = requests.get(
        f"{base_url}/parameters/2/latest",
        headers=headers,
        params={'limit': 5},
        timeout=10
    )
    
    print(f"Status : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Latest PM2.5 récupérées !")
        
        if 'results' in data and len(data['results']) > 0:
            print(f"\nNombre : {len(data['results'])}")
            
            for measure in data['results'][:3]:
                location = measure.get('location', {})
                sensor = measure.get('sensor', {})
                value = measure.get('value')
                unit = measure.get('unit')
                datetime_utc = measure.get('datetime', {}).get('utc', 'N/A')
                
                print(f"\n   📍 {location.get('name')}")
                print(f"      Country : {location.get('country', {}).get('name')}")
                print(f"      Value : {value} {unit}")
                print(f"      Date : {datetime_utc}")
        
    else:
        print(f"❌ Erreur : {response.status_code}")
        print(f"Response : {response.text}")
        
except Exception as e:
    print(f"❌ Erreur : {e}")

# Test 2 : Latest NO2 (parameter ID 5 ou 7 ?)
print("\n\n2️⃣ Latest NO2 values")
print("-" * 70)

# Essayer les deux IDs NO2
for no2_id in [5, 7]:
    try:
        response = requests.get(
            f"{base_url}/parameters/{no2_id}/latest",
            headers=headers,
            params={'limit': 3},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                print(f"\n   ✅ Parameter ID {no2_id} : {len(data['results'])} mesures")
                
                # Afficher première
                measure = data['results'][0]
                location = measure.get('location', {})
                print(f"      Exemple : {location.get('name')} = {measure.get('value')} {measure.get('unit')}")
                
    except Exception as e:
        print(f"   ❌ ID {no2_id} : {e}")

# Test 3 : Locations avec coordonnées Paris
print("\n\n3️⃣ Locations proche Paris (géoloc)")
print("-" * 70)

try:
    # Coordonnées Paris : 48.8566, 2.3522
    response = requests.get(
        f"{base_url}/locations",
        headers=headers,
        params={
            'coordinates': '48.8566,2.3522',
            'radius': 50000,  # 50 km autour de Paris
            'limit': 10
        },
        timeout=10
    )
    
    print(f"Status : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            print(f"✅ {len(data['results'])} stations proches Paris !")
            
            for station in data['results'][:5]:
                station_id = station.get('id')
                station_name = station.get('name')
                locality = station.get('locality')
                country = station.get('country', {}).get('name')
                
                print(f"\n   📍 {station_name}")
                print(f"      ID : {station_id}")
                print(f"      Locality : {locality}")
                print(f"      Country : {country}")
                
        else:
            print(f"⚠️ Aucune station proche Paris")
            
    else:
        print(f"❌ Erreur : {response.status_code}")
        
except Exception as e:
    print(f"❌ Erreur : {e}")

# Test 4 : Measurements pour un sensor spécifique
print("\n\n4️⃣ Measurements pour sensor 3917 (exemple doc)")
print("-" * 70)

try:
    response = requests.get(
        f"{base_url}/sensors/3917/measurements",
        headers=headers,
        params={'limit': 3},
        timeout=10
    )
    
    print(f"Status : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Measurements récupérées !")
        
        if 'results' in data:
            print(f"Nombre : {len(data['results'])}")
            print(f"Exemple : {json.dumps(data['results'][0], indent=2)[:400]}...")
        
    else:
        print(f"❌ Erreur : {response.status_code}")
        print(f"Response : {response.text}")
        
except Exception as e:
    print(f"❌ Erreur : {e}")

# Test 5 : Locations avec bbox Île-de-France
print("\n\n5️⃣ Locations bbox Île-de-France")
print("-" * 70)

try:
    # Bbox approximatif Île-de-France
    # lon_min, lat_min, lon_max, lat_max
    bbox = "1.4,48.1,3.6,49.2"
    
    response = requests.get(
        f"{base_url}/locations",
        headers=headers,
        params={
            'bbox': bbox,
            'limit': 10
        },
        timeout=10
    )
    
    print(f"Status : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            print(f"✅ {len(data['results'])} stations en Île-de-France !")
            
            for station in data['results']:
                print(f"   • {station.get('name')} ({station.get('country', {}).get('name')})")
                
        else:
            print(f"⚠️ Aucune station en Île-de-France")
            
except Exception as e:
    print(f"❌ Erreur : {e}")

print("\n" + "="*70)
print("FIN TEST")
print("="*70)