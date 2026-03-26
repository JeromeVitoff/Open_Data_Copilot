"""
Test différents formats d'authentification Airparif
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('AIRPARIF_API_KEY')
print(f"🔑 Clé Airparif : {api_key[:20]}...{api_key[-4:]}")

# URL qui répond (403 = URL valide, juste auth incorrecte)
base_url = "https://api.airparif.asso.fr"
endpoint = "/indices/prevision/commune"
full_url = f"{base_url}{endpoint}"
params = {'insee': '75056'}  # Paris

print(f"\n🌐 URL : {full_url}")
print(f"📍 Params : {params}")

print("\n" + "="*70)
print("TEST FORMATS D'AUTHENTIFICATION")
print("="*70)

# Format 1 : Bearer token (déjà testé, échoue)
print("\n1️⃣ Authorization: Bearer {key}")
try:
    response = requests.get(
        full_url,
        headers={'Authorization': f'Bearer {api_key}'},
        params=params,
        timeout=10
    )
    print(f"   Status : {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ SUCCÈS !")
    else:
        print(f"   Response : {response.text[:150]}")
except Exception as e:
    print(f"   ❌ Erreur : {e}")

# Format 2 : X-API-Key
print("\n2️⃣ X-API-Key: {key}")
try:
    response = requests.get(
        full_url,
        headers={'X-API-Key': api_key},
        params=params,
        timeout=10
    )
    print(f"   Status : {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ SUCCÈS !")
    else:
        print(f"   Response : {response.text[:150]}")
except Exception as e:
    print(f"   ❌ Erreur : {e}")

# Format 3 : apikey (lowercase)
print("\n3️⃣ apikey: {key}")
try:
    response = requests.get(
        full_url,
        headers={'apikey': api_key},
        params=params,
        timeout=10
    )
    print(f"   Status : {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ SUCCÈS !")
        data = response.json()
        print(f"   Données : {str(data)[:200]}...")
    else:
        print(f"   Response : {response.text[:150]}")
except Exception as e:
    print(f"   ❌ Erreur : {e}")

# Format 4 : Api-Key
print("\n4️⃣ Api-Key: {key}")
try:
    response = requests.get(
        full_url,
        headers={'Api-Key': api_key},
        params=params,
        timeout=10
    )
    print(f"   Status : {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ SUCCÈS !")
        data = response.json()
        print(f"   Données : {str(data)[:200]}...")
    else:
        print(f"   Response : {response.text[:150]}")
except Exception as e:
    print(f"   ❌ Erreur : {e}")

# Format 5 : X-Airparif-Key (custom)
print("\n5️⃣ X-Airparif-Key: {key}")
try:
    response = requests.get(
        full_url,
        headers={'X-Airparif-Key': api_key},
        params=params,
        timeout=10
    )
    print(f"   Status : {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ SUCCÈS !")
        data = response.json()
        print(f"   Données : {str(data)[:200]}...")
    else:
        print(f"   Response : {response.text[:150]}")
except Exception as e:
    print(f"   ❌ Erreur : {e}")

# Format 6 : Clé dans l'URL (query param)
print("\n6️⃣ URL query param : ?apikey={key}")
try:
    response = requests.get(
        full_url,
        params={**params, 'apikey': api_key},
        timeout=10
    )
    print(f"   Status : {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ SUCCÈS !")
        data = response.json()
        print(f"   Données : {str(data)[:200]}...")
    else:
        print(f"   Response : {response.text[:150]}")
except Exception as e:
    print(f"   ❌ Erreur : {e}")

# Format 7 : ?api_key={key}
print("\n7️⃣ URL query param : ?api_key={key}")
try:
    response = requests.get(
        full_url,
        params={**params, 'api_key': api_key},
        timeout=10
    )
    print(f"   Status : {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ SUCCÈS !")
        data = response.json()
        print(f"   Données : {str(data)[:200]}...")
    else:
        print(f"   Response : {response.text[:150]}")
except Exception as e:
    print(f"   ❌ Erreur : {e}")

# Format 8 : Authorization sans Bearer
print("\n8️⃣ Authorization: {key} (sans Bearer)")
try:
    response = requests.get(
        full_url,
        headers={'Authorization': api_key},
        params=params,
        timeout=10
    )
    print(f"   Status : {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ SUCCÈS !")
        data = response.json()
        print(f"   Données : {str(data)[:200]}...")
    else:
        print(f"   Response : {response.text[:150]}")
except Exception as e:
    print(f"   ❌ Erreur : {e}")

# Format 9 : Geodair (vous avez aussi GEODAIR_API_KEY)
geodair_key = os.getenv('GEODAIR_API_KEY')
if geodair_key:
    print("\n9️⃣ Test avec GEODAIR_API_KEY")
    print(f"   Clé Geodair : {geodair_key[:20]}...{geodair_key[-4:]}")
    
    for header_name in ['apikey', 'Api-Key', 'X-API-Key', 'Authorization']:
        try:
            if header_name == 'Authorization':
                header_value = f'Bearer {geodair_key}'
            else:
                header_value = geodair_key
                
            response = requests.get(
                full_url,
                headers={header_name: header_value},
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"   ✅ SUCCÈS avec {header_name} + GEODAIR_KEY !")
                data = response.json()
                print(f"   Données : {str(data)[:200]}...")
                break
            
        except:
            pass

print("\n" + "="*70)
print("FIN TEST")
print("="*70)