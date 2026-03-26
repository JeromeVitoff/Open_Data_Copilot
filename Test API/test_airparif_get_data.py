"""
Test récupération données Airparif avec bon format auth
"""
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv('AIRPARIF_API_KEY')
base_url = "https://api.airparif.asso.fr"

print("="*70)
print("RÉCUPÉRATION DONNÉES AIRPARIF")
print("="*70)

# Headers corrects
headers = {
    'X-API-Key': api_key,
    'Content-Type': 'application/json'
}

# Test 1 : Indices prévision Paris
print("\n1️⃣ Indices prévision commune Paris (75056)")
print("-" * 70)

try:
    response = requests.get(
        f"{base_url}/indices/prevision/commune",
        headers=headers,
        params={'insee': '75056'},
        timeout=10
    )
    
    print(f"Status : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Données récupérées !")
        print(f"\nType : {type(data)}")
        print(f"Longueur : {len(data) if isinstance(data, list) else 'N/A'}")
        
        # Afficher joliment
        print(f"\nContenu (formaté) :")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:1000])
        print("...")
        
        # Extraire infos clés si structure connue
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            print(f"\n📊 Premier enregistrement :")
            print(f"   Date : {first.get('date', 'N/A')}")
            print(f"   Indice : {first.get('indice', 'N/A')}")
            print(f"   Qualificatif : {first.get('qualificatif', 'N/A')}")
            
            if 'polluants' in first:
                print(f"\n   Polluants :")
                for polluant, info in first['polluants'].items():
                    print(f"      {polluant} : {info}")
    else:
        print(f"❌ Erreur : {response.status_code}")
        print(f"Response : {response.text}")
        
except Exception as e:
    print(f"❌ Erreur : {e}")

# Test 2 : Autres départements Île-de-France
print("\n\n2️⃣ Test autres communes")
print("-" * 70)

communes = [
    ('75056', 'Paris'),
    ('92050', 'Nanterre'),
    ('93008', 'Aubervilliers'),
]

for insee, nom in communes:
    try:
        response = requests.get(
            f"{base_url}/indices/prevision/commune",
            headers=headers,
            params={'insee': insee},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                indice = data[0].get('indice', 'N/A')
                qual = data[0].get('qualificatif', 'N/A')
                print(f"   ✅ {nom} ({insee}) : Indice {indice} - {qual}")
        else:
            print(f"   ⚠️ {nom} ({insee}) : Status {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ {nom} : {e}")

# Test 3 : Explorer endpoints disponibles
print("\n\n3️⃣ Explorer autres endpoints")
print("-" * 70)

endpoints = [
    "/indices/prevision/commune",
    "/mesures",
    "/episodes",
    "/stations",
]

for endpoint in endpoints:
    try:
        response = requests.get(
            f"{base_url}{endpoint}",
            headers=headers,
            timeout=10
        )
        
        print(f"\n{endpoint}")
        print(f"   Status : {response.status_code}")
        
        if response.status_code == 200:
            print(f"   ✅ Disponible")
        elif response.status_code == 404:
            print(f"   ⚠️ Non trouvé")
        else:
            print(f"   Response : {response.text[:100]}")
            
    except Exception as e:
        print(f"   ❌ Erreur : {e}")

print("\n" + "="*70)
print("FIN TEST")
print("="*70)