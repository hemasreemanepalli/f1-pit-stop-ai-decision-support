"""
Client for the Jolpica-F1 API (https://github.com/jolpica/jolpica-f1),
the maintained successor to the deprecated Ergast API.

IMPORTANT / HONESTY NOTE
-------------------------
This module was developed and unit-tested against mocked HTTP responses
only (see tests/test_jolpica_client.py). The sandboxed environment this
project was built in has an outbound network allow-list that does NOT
include api.jolpi.ca (or any other F1-data host), so live calls to the
real Jolpica API could not be executed while building this project. The
request/parsing/caching logic below follows Jolpica's documented,
Ergast-compatible response schema exactly, but it has not been validated
against a live response. When you run this on a machine with normal
internet access:

    python run_pipeline.py --download

will hit the real API and you should treat the FIRST run as a
verification step - inspect a couple of the cached JSON files in
data/raw/jolpica_cache/ and re-check the parsing in
`parse_laps_response` / `parse_pitstops_response` against what actually
comes back before trusting downstream numbers.

Endpoints used (per jolpica-f1/docs/README.md):
    /ergast/f1/{season}/{round}/laps/       -> per-lap times, all drivers
    /ergast/f1/{season}/{round}/pitstops/   -> pit stops, all drivers
    /ergast/f1/{season}/{round}/results/    -> race results (final position etc.)
    /ergast/f1/{season}.json                -> season race list (round -> race info)

Rate limiting: Jolpica publishes a public rate limit (a handful of
requests per second, burstier limits per hour). This client is
deliberately conservative (default: 3 req/sec) and retries with backoff
on 429/5xx.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests


class JolpicaClient:
    def __init__(
        self,
        base_url: str = "https://api.jolpi.ca/ergast/f1",
        cache_dir: str = "data/raw/jolpica_cache",
        requests_per_window: int = 2,
        window_seconds: float = 1.0,
        timeout_seconds: int = 30,
        max_retries: int = 8,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._request_times: list[float] = []
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "F1-Pit-Stop-AI/1.0 (educational project)",
            "Accept": "application/json",
        })

    # ---------------------------------------------------------- rate limit
    def _throttle(self):
        now = time.monotonic()
        self._request_times = [t for t in self._request_times if now - t < self.window_seconds]
        if len(self._request_times) >= self.requests_per_window:
            sleep_for = self.window_seconds - (now - self._request_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._request_times.append(time.monotonic())

    # ---------------------------------------------------------------- http
    def _get(self, path: str, cache_key: str) -> dict:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        url = f"{self.base_url}/{path}"
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=(10, self.timeout_seconds))
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        wait = float(retry_after) if retry_after else min(90.0, 5.0 * attempt)
                    except ValueError:
                        wait = min(90.0, 5.0 * attempt)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict) or "MRData" not in data:
                    raise ValueError("Jolpica returned an unexpected JSON structure")
                cache_file.write_text(json.dumps(data))
                return data
            except (requests.RequestException, ValueError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(min(45.0, 2 ** (attempt - 1)))
        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts: {last_err}")

    # ------------------------------------------------------------ fetchers
    def get_season_races(self, season: int) -> dict:
        return self._get(f"{season}.json", f"season_{season}_races")

    def get_laps(self, season: int, round_: int) -> dict:
        """Fetch the complete lap-timing response, following Jolpica pagination.

        Jolpica currently caps a page of lap timings at 100 records even when a
        larger ``limit`` is requested.  A single race therefore commonly has
        many pages.  The first page is kept in the normal ``laps_SEASON_ROUND``
        cache file for backward compatibility; additional pages use offset
        specific cache files.  The returned payload is a single combined
        Ergast-compatible response so downstream parsing does not need to know
        about pagination.
        """
        import copy

        # Load/cache the first page using the historical cache key.  If an older
        # run already cached an incomplete first page, it is still valid as page 0.
        first = self._get(
            f"{season}/{round_}/laps.json?limit=100&offset=0",
            f"laps_{season}_{round_}",
        )

        mr = first.get("MRData", {})
        total = int(mr.get("total", 0) or 0)
        races = mr.get("RaceTable", {}).get("Races", [])

        # Count timing records actually present in the cached first page.
        def timing_count(payload: dict) -> int:
            count = 0
            for race in payload.get("MRData", {}).get("RaceTable", {}).get("Races", []):
                for lap in race.get("Laps", []):
                    count += len(lap.get("Timings", []))
            return count

        collected = timing_count(first)
        page_size = int(mr.get("limit", 100) or 100)

        # Jolpica's total is the number of timing records, not the number of
        # distinct lap numbers.  Keep requesting pages until all timings exist.
        offset = page_size
        while offset < total:
            page = self._get(
                f"{season}/{round_}/laps.json?limit={page_size}&offset={offset}",
                f"laps_{season}_{round_}_offset_{offset}",
            )
            page_races = page.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not page_races:
                break

            # Normally there is exactly one race. Merge its Laps into the
            # corresponding race object from page 0.
            for page_race in page_races:
                target = next(
                    (r for r in races
                     if r.get("season") == page_race.get("season")
                     and r.get("round") == page_race.get("round")),
                    None,
                )
                if target is None:
                    races.append(copy.deepcopy(page_race))
                else:
                    target.setdefault("Laps", []).extend(page_race.get("Laps", []))

            n = timing_count(page)
            if n == 0:
                break
            collected += n
            offset += page_size

        # Write the fully combined response back to the normal cache file.
        # This makes subsequent pipeline runs fast and removes dependence on
        # the individual page files.
        combined = copy.deepcopy(first)
        combined.setdefault("MRData", {}).setdefault("RaceTable", {})["Races"] = races
        combined["MRData"]["limit"] = str(collected if total and collected >= total else page_size)
        combined["MRData"]["offset"] = "0"
        combined["MRData"]["total"] = str(total or collected)
        cache_file = self.cache_dir / f"laps_{season}_{round_}.json"
        cache_file.write_text(json.dumps(combined))
        return combined

    def get_pitstops(self, season: int, round_: int) -> dict:
        return self._get(f"{season}/{round_}/pitstops.json?limit=200", f"pitstops_{season}_{round_}")

    def get_results(self, season: int, round_: int) -> dict:
        return self._get(f"{season}/{round_}/results.json", f"results_{season}_{round_}")


# ----------------------------------------------------------------- parsing
def parse_laps_response(payload: dict) -> "pd.DataFrame":
    """MRData.RaceTable.Races[0].Laps[i].Timings[j] -> long dataframe
    with one row per (driver, lap)."""
    import pandas as pd

    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows = []
    for race in races:
        season = race.get("season")
        round_ = race.get("round")
        for lap in race.get("Laps", []):
            lap_number = lap.get("number")
            for timing in lap.get("Timings", []):
                rows.append(
                    {
                        "season": season,
                        "round": round_,
                        "lap": int(lap_number),
                        "driver_id": timing.get("driverId"),
                        "position": timing.get("position"),
                        "lap_time_str": timing.get("time"),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["lap_time_s"] = df["lap_time_str"].apply(_parse_mmssms)
    return df


def parse_pitstops_response(payload: dict) -> "pd.DataFrame":
    import pandas as pd

    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows = []
    for race in races:
        season = race.get("season")
        round_ = race.get("round")
        for ps in race.get("PitStops", []):
            rows.append(
                {
                    "season": season,
                    "round": round_,
                    "driver_id": ps.get("driverId"),
                    "pit_stop_number": ps.get("stop"),
                    "pit_lap": int(ps.get("lap")) if ps.get("lap") else None,
                    "pit_time_of_day": ps.get("time"),
                    "pit_duration_s": float(ps.get("duration")) if ps.get("duration") else None,
                }
            )
    return pd.DataFrame(rows)



def parse_results_driver_mapping(payload: dict) -> "pd.DataFrame":
    """Parse Jolpica race results into a CSV-driver-name bridge.

    Jolpica uses short driver IDs (for example ``hamilton``), while the
    cleaned CSV uses a slug made from the driver's full name (for example
    ``lewis_hamilton``). The results endpoint contains the driver's given
    and family names, so we can construct the same slug without hard-coding
    individual drivers.
    """
    import re
    import unicodedata
    import pandas as pd

    def slugify(text: str) -> str:
        text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
        return text

    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows = []
    for race in races:
        season = race.get("season")
        round_ = race.get("round")
        for result in race.get("Results", []):
            driver = result.get("Driver", {})
            given = driver.get("givenName", "")
            family = driver.get("familyName", "")
            full_name = f"{given} {family}".strip()
            rows.append({
                "season": int(season) if season is not None else None,
                "round": int(round_) if round_ is not None else None,
                "driver_id": driver.get("driverId"),
                "given_name": given,
                "family_name": family,
                "full_name": full_name,
                "driver_name_slug": slugify(full_name),
            })
    return pd.DataFrame(rows)

def _parse_mmssms(t: Optional[str]) -> Optional[float]:
    """Jolpica lap times look like 'M:SS.mmm'."""
    if not t:
        return None
    try:
        m, rest = t.split(":")
        return int(m) * 60 + float(rest)
    except ValueError:
        return None
