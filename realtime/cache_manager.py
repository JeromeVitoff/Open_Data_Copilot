"""
Cache fichier pour les données temps réel — évite les appels API répétés.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


class CacheManager:
    """Cache JSON sur disque avec TTL par source API."""

    # Durée de validité par API (minutes)
    VALIDITY: Dict[str, int] = {
        "spf":      60,   # 1 h
        "airparif": 30,   # 30 min
        "openaq":   30,   # 30 min
    }

    def __init__(self, cache_dir: str = "data/cache_realtime") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _key(self, api_name: str, params: Dict) -> str:
        digest = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()
        return f"{api_name}_{digest}"

    def _path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    # ── Interface publique ────────────────────────────────────────────────────

    def get(self, api_name: str, params: Dict) -> Optional[Any]:
        """
        Retourne les données du cache si elles sont encore valides.

        Returns:
            None si cache absent ou expiré, sinon les données.
        """
        path = self._path(self._key(api_name, params))
        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)

            ts = datetime.fromisoformat(cached["timestamp"])
            ttl = timedelta(minutes=self.VALIDITY.get(api_name, 30))
            if datetime.now() - ts > ttl:
                return None  # expiré

            print(f"✅ Cache hit: {api_name}")
            return cached["data"]

        except Exception as exc:
            print(f"⚠️  Cache read error ({api_name}): {exc}")
            return None

    def set(self, api_name: str, params: Dict, data: Any) -> None:
        """Sauvegarde les données dans le cache."""
        path = self._path(self._key(api_name, params))
        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "api": api_name,
                "params": params,
                "data": data,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            print(f"✅ Cache saved: {api_name}")
        except Exception as exc:
            print(f"⚠️  Cache write error ({api_name}): {exc}")

    def clear(self, api_name: Optional[str] = None) -> None:
        """Efface le cache — pour une API spécifique ou en totalité."""
        pattern = f"{api_name}_*.json" if api_name else "*.json"
        for f in self.cache_dir.glob(pattern):
            f.unlink()
        label = api_name or "all"
        print(f"✅ Cache cleared: {label}")
