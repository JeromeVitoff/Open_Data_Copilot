"""
Test OpenAQ depuis machine locale
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAQ_API_KEY')
print(f"🔑 Clé OpenAQ : {'✅' if api_key else '❌'}")

if not api_key:
    print("❌ OPENAQ_API_KEY non trouvée")
    exit(1)

print(f"   Longueur : {len(api_key)}")
print(f"   Préfixe : {api_key[:15]}...")

print("\n" + "="*70)
print("TEST OPENAQ API")
print("="*70)

# Test v2
print("\n🌐 OpenAQ v2 (legacy) :")
try:
    response = requests.get(
        "https://api.openaq.org/v2/latest",
        headers={'X-API-Key': api_key},
        params={'city': 'Paris', 'parameter': 'no2', 'limit': 5},
        timeout=10
    )
    
    print(f"   Status : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ V2 fonctionne !")
        print(f"   Résultats : {len(data.get('results', []))}")
        if data.get('results'):
            print(f"   Exemple : {data['results'][0]}")
    else:
        print(f"   ❌ Échec v2")
        print(f"   Message : {response.text[:300]}")
        
except Exception as e:
    print(f"   ❌ Erreur v2 : {e}")

# Test v3
print("\n🌐 OpenAQ v3 (nouveau) :")
try:
    response = requests.get(
        "https://api.openaq.org/v3/locations",
        headers={'X-API-Key': api_key},
        params={'city': 'Paris', 'limit': 5},
        timeout=10
    )
    
    print(f"   Status : {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ V3 fonctionne !")
        print(f"   Données : {str(data)[:300]}...")
    else:
        print(f"   ❌ Échec v3")
        print(f"   Message : {response.text[:300]}")
        
except Exception as e:
    print(f"   ❌ Erreur v3 : {e}")

# Test authentification simple
print("\n🔐 Test compte OpenAQ :")
try:
    # Endpoint pour vérifier clé
    response = requests.get(
        "https://api.openaq.org/v2/measurements",
        headers={'X-API-Key': api_key},
        params={'limit': 1},
        timeout=10
    )
    
    print(f"   Status : {response.status_code}")
    
    if response.status_code == 200:
        print(f"   ✅ Clé API VALIDE")
    elif response.status_code == 401:
        print(f"   ❌ Clé API INVALIDE ou EXPIRÉE")
        print(f"   → Créer nouveau compte sur https://explore.openaq.org/register")
    else:
        print(f"   ⚠️ Status inattendu : {response.status_code}")
        
except Exception as e:
    print(f"   ❌ Erreur : {e}")

print("\n" + "="*70)
print("FIN TEST")
print("="*70)