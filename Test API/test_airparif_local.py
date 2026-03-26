"""
Test Airparif depuis machine locale (pas VM)
"""
import os
import requests
from dotenv import load_dotenv

# Charger .env
load_dotenv()

api_key = os.getenv('AIRPARIF_API_KEY')
print(f"🔑 Clé Airparif : {'✅' if api_key else '❌'}")

if not api_key:
    print("❌ AIRPARIF_API_KEY non trouvée dans .env")
    exit(1)

print(f"   Longueur : {len(api_key)}")
print(f"   Préfixe : {api_key[:15]}...")

# Test différents endpoints Airparif
base_urls = [
    "https://api.airparif.asso.fr",
    "https://api.airparif.fr",
    "https://www.airparif.asso.fr/services/api",
]

endpoints = [
    "/indices/prevision/commune",
    "/indices/commune",
    "/mesures/temps-reel",
    "/v1/indices/commune",
]

print("\n" + "="*70)
print("TEST AIRPARIF API")
print("="*70)

for base_url in base_urls:
    print(f"\n🌐 Base URL : {base_url}")
    
    for endpoint in endpoints:
        full_url = f"{base_url}{endpoint}"
        print(f"\n   Testing : {endpoint}")
        
        try:
            # Essayer avec Bearer token
            response = requests.get(
                full_url,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                params={'insee': '75056'},  # Paris
                timeout=10
            )
            
            print(f"   Status : {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ SUCCÈS !")
                data = response.json()
                print(f"   Type : {type(data)}")
                print(f"   Contenu : {str(data)[:300]}...")
                print(f"\n   🎉 URL VALIDE : {full_url}")
                break
                
            elif response.status_code == 401:
                print(f"   ❌ 401 Unauthorized")
                print(f"   Message : {response.text[:200]}")
                
                # Essayer avec X-API-Key
                response2 = requests.get(
                    full_url,
                    headers={'X-API-Key': api_key},
                    params={'insee': '75056'},
                    timeout=10
                )
                
                if response2.status_code == 200:
                    print(f"   ✅ SUCCÈS avec X-API-Key !")
                    print(f"   🎉 URL VALIDE : {full_url}")
                    break
                    
            elif response.status_code == 404:
                print(f"   ⚠️ 404 Not Found")
                
            else:
                print(f"   ⚠️ Status {response.status_code}")
                print(f"   Response : {response.text[:200]}")
                
        except requests.exceptions.ConnectionError as e:
            print(f"   ❌ Connexion impossible : {str(e)[:100]}")
        except requests.exceptions.Timeout:
            print(f"   ⚠️ Timeout")
        except Exception as e:
            print(f"   ❌ Erreur : {e}")

print("\n" + "="*70)
print("FIN TEST")
print("="*70)