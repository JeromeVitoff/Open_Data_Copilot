"""
api_wrappers — clients pour APIs temps réel (SPF, Airparif, OpenAQ)
"""
from .spf_api import SPFRealtimeAPI
from .airparif_api import AirparifRealtimeAPI
from .openaq_api import OpenAQRealtimeAPI

__all__ = ["SPFRealtimeAPI", "AirparifRealtimeAPI", "OpenAQRealtimeAPI"]
