"""
Wrapper API Airparif temps réel
Documentation : https://api.airparif.fr/docs
"""
import requests
from datetime import datetime
import os
from typing import Dict, List, Optional

class AirparifRealtimeAPI:
    """
    API Airparif pour pollution Île-de-France temps réel
    
    Authentification : Header X-API-Key
    Licence : ODbL
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('AIRPARIF_API_KEY')
        if not self.api_key:
            raise ValueError("AIRPARIF_API_KEY non trouvée dans .env")
        
        # Base URL Airparif (corrigée)
        self.base_url = "https://api.airparif.fr"
        
        self.session = requests.Session()
        # Format authentification correct : X-API-Key
        self.session.headers.update({
            'X-API-Key': self.api_key,
            'Content-Type': 'application/json'
        })
    
    def get_current_pollution(
        self, 
        city: str = "Paris",
        insee_code: str = "75056"
    ) -> Dict:
        """
        Récupère prévisions pollution pour une commune
        
        Args:
            city: Nom de la ville (pour affichage)
            insee_code: Code INSEE de la commune
                       Paris: 75056
                       Nanterre: 92050
                       Aubervilliers: 93008
            
        Returns:
            Données pollution actuelles et prévisions
        """
        try:
            # Endpoint indices prevision
            url = f"{self.base_url}/indices/prevision/commune"
            params = {'insee': insee_code}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parser réponse Airparif
            # Format : {"75056": [{"date": "2026-03-26", ...}, ...]}
            if insee_code in data and len(data[insee_code]) > 0:
                previsions = data[insee_code]
                today = previsions[0]  # Prévision aujourd'hui
                
                result = {
                    'city': city,
                    'insee_code': insee_code,
                    'date': today.get('date'),
                    'indice': today.get('indice'),  # "Bon", "Moyen", etc.
                    'qualificatifs': {
                        'NO2': today.get('no2'),
                        'O3': today.get('o3'),
                        'PM10': today.get('pm10'),
                        'PM2.5': today.get('pm25'),
                        'SO2': today.get('so2')
                    },
                    'source': 'Airparif',
                    'previsions': previsions  # Toutes les prévisions J, J+1...
                }
                
                print(f"✅ Airparif: Données récupérées pour {city}")
                return result
            
            print(f"⚠️ Airparif: Aucune donnée pour {city} ({insee_code})")
            return {}
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur Airparif API: {e}")
            return {}
        except Exception as e:
            print(f"❌ Erreur parsing Airparif: {e}")
            return {}
    
    def get_episode_pollution(self) -> Dict:
        """
        Récupère alertes épisodes de pollution en cours et prévus
        
        Returns:
            Info sur épisodes pollution (aujourd'hui + demain)
        """
        try:
            url = f"{self.base_url}/episodes/en-cours-et-prevus"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Format : 
            # {
            #   "actif": true,
            #   "jour": {"actif": false, "polluants": []},
            #   "demain": {"actif": true, "polluants": ["NO2", "information"]},
            #   "message": {"fr": "...", "en": "..."}
            # }
            
            print(f"✅ Airparif: Épisodes pollution récupérés")
            return data
            
        except Exception as e:
            print(f"❌ Erreur épisodes pollution: {e}")
            return {}
    
    def get_bulletin_prevision(self) -> str:
        """
        Récupère bulletin de prévision des prévisionnistes
        
        Returns:
            Texte du bulletin
        """
        try:
            url = f"{self.base_url}/indices/prevision/bulletin"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Le bulletin est probablement du texte ou JSON
            data = response.text if response.headers.get('content-type') == 'text/plain' else response.json()
            
            print(f"✅ Airparif: Bulletin récupéré")
            return data
            
        except Exception as e:
            print(f"❌ Erreur bulletin: {e}")
            return ""
    
    def get_multiple_cities(
        self,
        cities: List[tuple]  # [(nom, code_insee), ...]
    ) -> List[Dict]:
        """
        Récupère pollution pour plusieurs villes
        
        Args:
            cities: Liste de tuples (nom_ville, code_insee)
            
        Returns:
            Liste de résultats par ville
        """
        results = []
        for city_name, insee_code in cities:
            data = self.get_current_pollution(city_name, insee_code)
            if data:
                results.append(data)
        return results
    
    def format_for_rag(self, data: Dict) -> Dict:
        """
        Formate données Airparif pour RAG
        """
        if not data:
            return {}
        
        # Construire texte descriptif
        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        city = data.get('city', 'Ville')
        indice = data.get('indice', 'N/A')
        qualifs = data.get('qualificatifs', {})
        
        # Texte pour LLM
        text_parts = [
            f"Le {date}, à {city}",
            f"indice de qualité de l'air : {indice}"
        ]
        
        # Ajouter qualificatifs par polluant
        polluants_desc = []
        for polluant, qual in qualifs.items():
            if qual:
                polluants_desc.append(f"{polluant} {qual}")
        
        if polluants_desc:
            text_parts.append(", ".join(polluants_desc))
        
        return {
            'text': ": ".join(text_parts) + ".",
            'metadata': {
                'source': 'Airparif',
                'date': date,
                'city': city,
                'insee_code': data.get('insee_code'),
                'type': 'pollution_prevision',
                'is_realtime': True,
                'indice': indice
            }
        }


# Codes INSEE communes Île-de-France (pour référence)
CODES_INSEE_IDF = {
    'Paris': '75056',
    'Nanterre': '92050',
    'Aubervilliers': '93008',
    'Créteil': '94028',
    'Argenteuil': '95018',
    'Versailles': '78646',
    'Meaux': '77284',
    'Melun': '77288',
}