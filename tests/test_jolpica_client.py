"""
Tests for the Jolpica client. These use MOCKED HTTP responses only - see
the honesty note in src/data/jolpica_client.py: this sandbox cannot reach
the real Jolpica API, so live-integration testing was not possible while
building this project. These tests verify the parsing/caching/rate-limit
logic against a realistic, schema-accurate fake response instead.
"""
import json

from unittest.mock import patch, Mock
import pytest

from src.data.jolpica_client import (
    JolpicaClient, parse_laps_response, parse_pitstops_response, _parse_mmssms,
)

FAKE_LAPS_RESPONSE = {
    "MRData": {
        "RaceTable": {
            "Races": [
                {
                    "season": "2023",
                    "round": "1",
                    "Laps": [
                        {
                            "number": "1",
                            "Timings": [
                                {"driverId": "verstappen", "position": "1", "time": "1:35.123"},
                                {"driverId": "hamilton", "position": "2", "time": "1:36.456"},
                            ],
                        },
                        {
                            "number": "2",
                            "Timings": [
                                {"driverId": "verstappen", "position": "1", "time": "1:34.789"},
                            ],
                        },
                    ],
                }
            ]
        }
    }
}

FAKE_PITSTOPS_RESPONSE = {
    "MRData": {
        "RaceTable": {
            "Races": [
                {
                    "season": "2023",
                    "round": "1",
                    "PitStops": [
                        {"driverId": "verstappen", "stop": "1", "lap": "15", "time": "14:23:11", "duration": "22.456"},
                    ],
                }
            ]
        }
    }
}


def test_parse_mmssms():
    assert _parse_mmssms("1:35.123") == pytest.approx(95.123)
    assert _parse_mmssms(None) is None
    assert _parse_mmssms("garbage") is None


def test_parse_laps_response():
    df = parse_laps_response(FAKE_LAPS_RESPONSE)
    assert len(df) == 3
    assert set(df["driver_id"]) == {"verstappen", "hamilton"}
    assert abs(df.loc[df["driver_id"] == "verstappen", "lap_time_s"].iloc[0] - 95.123) < 1e-6


def test_parse_pitstops_response():
    df = parse_pitstops_response(FAKE_PITSTOPS_RESPONSE)
    assert len(df) == 1
    assert df.iloc[0]["pit_lap"] == 15
    assert df.iloc[0]["pit_duration_s"] == 22.456


def test_client_caches_to_disk(tmp_path):
    client = JolpicaClient(cache_dir=str(tmp_path), requests_per_window=100)
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = FAKE_LAPS_RESPONSE
    mock_resp.raise_for_status = Mock()

    with patch("src.data.jolpica_client.requests.get", return_value=mock_resp) as mock_get:
        data1 = client.get_laps(2023, 1)
        data2 = client.get_laps(2023, 1)  # should hit cache, not network again

    assert mock_get.call_count == 1  # second call served from cache
    assert data1 == FAKE_LAPS_RESPONSE
    assert data2 == FAKE_LAPS_RESPONSE
    cached_file = tmp_path / "laps_2023_1.json"
    assert cached_file.exists()


def test_client_retries_on_429(tmp_path):
    client = JolpicaClient(cache_dir=str(tmp_path), requests_per_window=100, max_retries=2)

    resp_429 = Mock(status_code=429, headers={"Retry-After": "0"})
    resp_ok = Mock(status_code=200)
    resp_ok.json.return_value = FAKE_PITSTOPS_RESPONSE
    resp_ok.raise_for_status = Mock()

    with patch("src.data.jolpica_client.requests.get", side_effect=[resp_429, resp_ok]):
        data = client.get_pitstops(2023, 1)
    assert data == FAKE_PITSTOPS_RESPONSE
