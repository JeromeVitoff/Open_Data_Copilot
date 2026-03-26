"""
Test OpenAQ v3 - Trouver bons endpoints
"""
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv('OPENAQ_API_KEY')
base_url = "https://api.openaq.org/v3"

headers = {
    'X-API-Key': api_key
}

print("="*70)
print("OPENAQ V3 - RECHERCHE BONS ENDPOINTS")
print("="*70)

# Test différents endpoints possibles
endpoints_to_test = [
    "/latest",
    "/measurements/latest",
    "/measurements",
    "/latest/measurements",
    "/sensors",
]

print("\n1️⃣ Test endpoints measurements")
print("-" * 70)

for endpoint in endpoints_to_test:
    try:
        response = requests.get(
            f"{base_url}{endpoint}",
            headers=headers,
            params={'limit': 1, 'country': 'FR'},
            timeout=10
        )
        
        print(f"\n{endpoint}")
        print(f"   Status : {response.status_code}")
        
        if response.status_code == 200:
            print(f"   ✅ FONCTIONNE !")
            data = response.json()
            print(f"   Keys : {list(data.keys())}")
            if 'results' in data and len(data['results']) > 0:
                print(f"   Exemple : {str(data['results'][0])[:200]}...")
        elif response.status_code == 404:
            print(f"   ❌ Not Found")
        else:
            print(f"   Response : {response.text[:100]}")
            
    except Exception as e:
        print(f"   ❌ Erreur : {e}")

# Test 2 : Recherche France spécifiquement
print("\n\n2️⃣ Locations France (code FR)")
print("-" * 70)

try:
    response = requests.get(
        f"{base_url}/locations",
        headers=headers,
        params={'country': 'FR', 'limit': 5},
        timeout=10
    )
    
    print(f"Status : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            print(f"✅ {len(data['results'])} stations France trouvées !")
            
            for station in data['results']:
                station_id = station.get('id')
                station_name = station.get('name')
                locality = station.get('locality')
                country = station.get('country', {}).get('name')
                
                print(f"\n   📍 {station_name}")
                print(f"      ID : {station_id}")
                print(f"      Locality : {locality}")
                print(f"      Country : {country}")
                
        else:
            print(f"⚠️ Aucune station France")
            
    else:
        print(f"❌ Erreur : {response.status_code}")
        
except Exception as e:
    print(f"❌ Erreur : {e}")

# Test 3 : Essayer avec location_id spécifique
print("\n\n3️⃣ Mesures pour location spécifique")
print("-" * 70)

# Si on a trouvé des locations France ci-dessus, utiliser leur ID
# Sinon, tester avec ID aléatoire

# Essayer endpoint measurements avec location_id
try:
    # Essayer plusieurs variations
    test_urls = [
        f"{base_url}/measurements?location_id=1&limit=5",
        f"{base_url}/locations/1/measurements?limit=5",
        f"{base_url}/sensors?location_id=1&limit=5",
    ]
    
    for test_url in test_urls:
        try:
            response = requests.get(
                test_url,
                headers=headers,
                timeout=10
            )
            
            print(f"\n{test_url.split(base_url)[1]}")
            print(f"   Status : {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ FONCTIONNE !")
                data = response.json()
                if 'results' in data:
                    print(f"   Results : {len(data['results'])}")
                    
        except:
            pass
            
except Exception as e:
    print(f"❌ Erreur : {e}")

# Test 4 : Documentation OpenAQ v3
print("\n\n4️⃣ Info API Documentation")
print("-" * 70)

try:
    # Essayer endpoint root ou docs
    response = requests.get(
        base_url,
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Root endpoint accessible")
        print(f"Data : {data}")
        
except Exception as e:
    print(f"Info : L'endpoint root ne donne pas de doc")

print("\n" + "="*70)
print("DOCUMENTATION OFFICIELLE :")
print("https://docs.openaq.org/docs")
print("https://api.openaq.org/v3/docs (Swagger possible)")
print("="*70)