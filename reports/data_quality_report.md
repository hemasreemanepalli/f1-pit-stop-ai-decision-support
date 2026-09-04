# Data Quality Report - raw CSV cleaning

- Driver-race groups (stint sets): 2821
- Driver-races where sum(stint_length) != total race laps: 645 (22.9%)
- Driver-races with NO usable stint length data at all (sum==0): 109
- Duplicate stint rows removed: 0
- Races missing weather/metadata entirely: 9
    - {'season': 2018, 'round': 19, 'circuit': 'Autódromo Hermanos Rodríguez'}
    - {'season': 2019, 'round': 18, 'circuit': 'Autódromo Hermanos Rodríguez'}
    - {'season': 2020, 'round': 11, 'circuit': 'Nürburgring'}
    - {'season': 2020, 'round': 12, 'circuit': 'Autódromo Internacional do Algarve'}
    - {'season': 2021, 'round': 3, 'circuit': 'Autódromo Internacional do Algarve'}
    - {'season': 2022, 'round': 20, 'circuit': 'Autódromo Hermanos Rodríguez'}
    - {'season': 2023, 'round': 19, 'circuit': 'Autódromo Hermanos Rodríguez'}
    - {'season': 2024, 'round': 20, 'circuit': 'Autódromo Hermanos Rodríguez'}
    - {'season': 2024, 'round': 21, 'circuit': 'Autódromo José Carlos Pace'}

- Rows after cleaning: 7374
- Seasons covered: [2018, 2019, 2020, 2021, 2022, 2023, 2024]

- Columns excluded from ML features because they are computed from the FULL race result (post-race aggregates -> leakage if used to predict an in-race decision):
    - Position
    - TotalPitStops
    - AvgPitStopTime
    - race_lap_time_variation
    - race_total_pit_stops_dup
    - race_tire_usage_aggression
    - race_fast_lap_attempts
    - race_position_changes
    - race_driver_aggression_score