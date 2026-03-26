"""
Tests unitaires — TemporalDetector.
Lance avec : pytest realtime/tests/test_temporal_detector.py -v
"""
import pytest
from datetime import date, timedelta

from realtime.temporal_detector import TemporalDetector


@pytest.fixture
def detector():
    return TemporalDetector()


# ── Classification ────────────────────────────────────────────────────────────

class TestDetectType:

    def test_realtime_aujourdhui(self, detector):
        r = detector.detect("Pollution à Paris aujourd'hui ?")
        assert r["type"] == "realtime"
        assert r["needs_realtime_data"] is True

    def test_realtime_actuellement(self, detector):
        r = detector.detect("Qualité air Paris actuellement")
        assert r["type"] == "realtime"

    def test_realtime_cette_semaine(self, detector):
        r = detector.detect("Évolution grippe cette semaine")
        assert r["type"] == "realtime"

    def test_realtime_recent(self, detector):
        r = detector.detect("Données COVID récentes")
        assert r["type"] == "realtime"

    def test_historical_year(self, detector):
        r = detector.detect("Hospitalisations COVID en mars 2021")
        assert r["type"] == "historical"
        assert r["needs_realtime_data"] is False

    def test_historical_2020(self, detector):
        r = detector.detect("Lien pollution santé 2020")
        assert r["type"] == "historical"

    def test_unspecified(self, detector):
        r = detector.detect("Taux vaccination en France")
        assert r["type"] == "unspecified"
        assert r["needs_realtime_data"] is False

    def test_mixed(self, detector):
        r = detector.detect("Évolution COVID 2021 et actuellement")
        assert r["type"] == "mixed"


# ── Scores ────────────────────────────────────────────────────────────────────

class TestScores:

    def test_realtime_score_range(self, detector):
        r = detector.detect("aujourd'hui actuellement")
        assert 0.0 <= r["realtime_score"] <= 1.0

    def test_historical_score_range(self, detector):
        r = detector.detect("en 2020 avant pendant")
        assert 0.0 <= r["historical_score"] <= 1.0

    def test_no_match_zero_scores(self, detector):
        r = detector.detect("grippe")
        assert r["realtime_score"] == 0.0
        assert r["historical_score"] == 0.0


# ── Plages temporelles ────────────────────────────────────────────────────────

class TestSuggestedTimerange:

    def test_aujourdhui_range(self, detector):
        r = detector.detect("Pollution aujourd'hui")
        start, end = r["suggested_timerange"]
        assert start == end == date.today()

    def test_cette_semaine_range(self, detector):
        r = detector.detect("Données cette semaine")
        start, end = r["suggested_timerange"]
        assert end == date.today()
        assert (date.today() - start).days <= 6

    def test_ce_mois_range(self, detector):
        r = detector.detect("Stats ce mois")
        start, end = r["suggested_timerange"]
        assert start.day == 1
        assert start.month == date.today().month

    def test_default_realtime_7days(self, detector):
        r = detector.detect("données récentes")
        start, end = r["suggested_timerange"]
        assert (end - start).days == 7

    def test_historical_year_range(self, detector):
        r = detector.detect("données 2021")
        start, end = r["suggested_timerange"]
        assert start.year == 2021
        assert end.year == 2021

    def test_unspecified_no_range(self, detector):
        r = detector.detect("grippe")
        assert r["suggested_timerange"] is None


# ── Références temporelles ────────────────────────────────────────────────────

class TestTimeReferences:

    def test_refs_not_empty_for_realtime(self, detector):
        r = detector.detect("aujourd'hui")
        assert len(r["time_references"]) >= 1

    def test_refs_empty_for_unspecified(self, detector):
        r = detector.detect("grippe")
        assert r["time_references"] == []
