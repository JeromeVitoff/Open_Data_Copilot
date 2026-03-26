"""
Test OpenAQ v3 - Explorer structure données
"""
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv('OPENAQ_API_KEY')
base_url = "https://api.openaq.org/v3"

headers = {
    'X-API-Key': api_key,
    'Content-Type': 'application/json'
}

print("="*70)
print("OPENAQ V3 - EXPLORATION STRUCTURE")
print("="*70)

# Test 1 : Locations Paris
print("\n1️⃣ Locations Paris")
print("-" * 70)

try:
    response = requests.get(
        f"{base_url}/locations",
        headers=headers,
        params={
            'city': 'Paris',
            'limit': 3
        },
        timeout=10
    )
    
    print(f"Status : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Locations trouvées !")
        print(f"\nMeta : {data.get('meta')}")
        
        if 'results' in data and len(data['results']) > 0:
            print(f"\nNombre de stations : {len(data['results'])}")
            
            # Afficher première station
            station = data['results'][0]
            print(f"\n📍 Première station :")
            print(json.dumps(station, indent=2)[:500])
            print("...")
            
            # Extraire ID pour mesures
            location_id = station.get('id')
            print(f"\n   ID : {location_id}")
            print(f"   Nom : {station.get('name')}")
            print(f"   Pays : {station.get('country', {}).get('name')}")
            
    else:
        print(f"❌ Erreur : {response.status_code}")
        print(f"Response : {response.text}")
        
except Exception as e:
    print(f"❌ Erreur : {e}")

# Test 2 : Latest measurements
print("\n\n2️⃣ Latest Measurements (mesures récentes)")
print("-" * 70)

try:
    response = requests.get(
        f"{base_url}/latest",
        headers=headers,
        params={
            'country': 'FR',  # France
            'limit': 5
        },
        timeout=10
    )
    
    print(f"Status : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Mesures récupérées !")
        
        if 'results' in data and len(data['results']) > 0:
            print(f"\nNombre de mesures : {len(data['results'])}")
            
            # Afficher première mesure
            measure = data['results'][0]
            print(f"\n📊 Première mesure :")
            print(json.dumps(measure, indent=2)[:700])
            print("...")
            
    else:
        print(f"❌ Erreur : {response.status_code}")
        print(f"Response : {response.text}")
        
except Exception as e:
    print(f"❌ Erreur : {e}")

# Test 3 : Parameters disponibles
print("\n\n3️⃣ Parameters (polluants disponibles)")
print("-" * 70)

try:
    response = requests.get(
        f"{base_url}/parameters",
        headers=headers,
        params={'limit': 20},
        timeout=10
    )
    
    print(f"Status : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Parameters récupérés !")
        
        if 'results' in data:
            print(f"\nPolluants disponibles :")
            for param in data['results'][:10]:
                param_id = param.get('id')
                param_name = param.get('name')
                print(f"   • {param_name} (ID: {param_id})")
                
    else:
        print(f"❌ Erreur : {response.status_code}")
        
except Exception as e:
    print(f"❌ Erreur : {e}")

# Test 4 : Countries
print("\n\n4️⃣ Countries")
print("-" * 70)

try:
    response = requests.get(
        f"{base_url}/countries",
        headers=headers,
        params={'limit': 5},
        timeout=10
    )
    
    print(f"Status : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'results' in data:
            print(f"✅ {len(data['results'])} pays")
            for country in data['results']:
                print(f"   • {country.get('name')} ({country.get('code')})")
                
except Exception as e:
    print(f"❌ Erreur : {e}")

print("\n" + "="*70)
print("FIN EXPLORATION")
print("="*70)