#!/usr/bin/env python3
"""
Precipitation forecasting pipeline for UnB/DF with dynamic upwind features.

This script implements:
1. Daily panel building from INMET cleaned CSV files (2019-2024).
2. Dynamic upwind station selection based on target wind direction/speed.
3. Physical lag features from station distance and wind speed.
4. Walk-forward ARIMAX backtests for D+1, D+2, D+3.
5. Comparative evaluation across baseline, fixed-radius, and dynamic-upwind methods.

Expected inputs:
- data/all_stations.csv
- data/cleaned_data/<YEAR>_cleaned/*.CSV
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COORD_UNB_DF = (-15.760063, -47.871154)
DEFAULT_YEARS = tuple(range(2019, 2025))

COL_DATE = "date"
COL_STATION = "station_code"
COL_LAT = "latitude"
COL_LON = "longitude"
COL_PRECIP = "precip_mm"
COL_HUMIDITY = "humidity_pct"
COL_DEWPOINT = "dewpoint_c"
COL_PRESSURE = "pressure_mb"
COL_WIND_DIR = "wind_dir_deg"
COL_WIND_SPEED = "wind_speed_ms"

LOCAL_COLS = [
    f"local_{COL_WIND_DIR}",
    f"local_{COL_WIND_SPEED}",
    f"local_{COL_DEWPOINT}",
    f"local_{COL_HUMIDITY}",
    f"local_{COL_PRESSURE}",
]

RADIUS_COLS = [
    f"radius_{COL_HUMIDITY}",
    f"radius_{COL_DEWPOINT}",
    f"radius_{COL_PRESSURE}",
    f"radius_{COL_PRECIP}",
]

UPWIND_COLS = [
    "upwind_humidity_lag",
    "upwind_dewpoint_lag",
    "upwind_pressure_lag",
    "upwind_precip_lag",
]

TARGET_COL = f"local_{COL_PRECIP}"


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class UpwindFeatureConfig:
    angle_tol_deg: float = 35.0
    max_dist_km: float = 400.0
    min_wind_ms: float = 1.5
    min_lag_days: int = 0
    max_lag_days: int = 5
    wind_dir_is_from: bool = True


@dataclass
class BacktestConfig:
    order: Tuple[int, int, int] = (3, 0, 1)
    trend: str = "ct"
    seasonality_days: float = 365.25
    fourier_terms: int = 4
    test_start: str = "2023-01-01"
    retrain_every_days: int = 7
    min_train_days: int = 730
    min_train_rows: int = 365


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def normalize_text(value: str) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def parse_float(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def circular_mean_deg(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")

    rad = np.deg2rad(np.mod(arr, 360.0))
    s = np.sin(rad).mean()
    c = np.cos(rad).mean()
    if np.isclose(s, 0.0) and np.isclose(c, 0.0):
        return float("nan")
    return float(np.rad2deg(np.arctan2(s, c)) % 360.0)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angular_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def create_fourier_terms(time_idx: np.ndarray, period: float, num_terms: int) -> np.ndarray:
    cols: List[np.ndarray] = []
    for k in range(1, num_terms + 1):
        cols.append(np.sin(2 * np.pi * k * time_idx / period))
        cols.append(np.cos(2 * np.pi * k * time_idx / period))
    return np.column_stack(cols)


def mean_absolute_error_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.abs(y_true) + np.abs(y_pred)
    valid = den > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(200.0 * np.abs(y_pred[valid] - y_true[valid]) / den[valid]))


def f1_binary_np(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    y_true_bin = y_true_bin.astype(int)
    y_pred_bin = y_pred_bin.astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def choose_first_column(
    columns_norm: Dict[str, str],
    includes: Sequence[str],
    excludes: Sequence[str] = (),
    required: bool = True,
) -> Optional[str]:
    include_norm = [normalize_text(x) for x in includes]
    exclude_norm = [normalize_text(x) for x in excludes]

    for col, norm_col in columns_norm.items():
        if all(token in norm_col for token in include_norm) and all(
            token not in norm_col for token in exclude_norm
        ):
            return col
    if required:
        raise KeyError(f"Column not found for includes={includes}, excludes={excludes}")
    return None


def extract_station_code(filename: str) -> Optional[str]:
    match = re.search(r"_([A-Z]\d{3})_", filename)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------


def load_station_metadata(stations_csv: Path) -> pd.DataFrame:
    encodings = ["latin1", "utf-8"]
    last_error: Optional[Exception] = None
    df_raw: Optional[pd.DataFrame] = None
    for enc in encodings:
        try:
            df_raw = pd.read_csv(stations_csv, sep=";", dtype=str, encoding=enc)
            break
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc

    if df_raw is None:
        raise RuntimeError(
            f"Unable to read station metadata: {stations_csv}. Last error: {last_error}"
        )

    columns_norm = {col: normalize_text(col) for col in df_raw.columns}

    col_file = choose_first_column(columns_norm, ["arquivo"])
    col_lat = choose_first_column(columns_norm, ["latitude"])
    col_lon = choose_first_column(columns_norm, ["longitude"])
    col_code = choose_first_column(columns_norm, ["codigo"], required=False)
    col_name = choose_first_column(columns_norm, ["estac"], required=False)
    col_uf = choose_first_column(columns_norm, ["uf"], required=False)

    station_code = (
        df_raw[col_code].astype(str).str.strip() if col_code else df_raw[col_file].map(extract_station_code)
    )
    station_code = station_code.where(station_code.str.fullmatch(r"[A-Z]\d{3}"), df_raw[col_file].map(extract_station_code))

    df_meta = pd.DataFrame(
        {
            COL_STATION: station_code,
            "station_name": df_raw[col_name].astype(str).str.strip() if col_name else "",
            "uf": df_raw[col_uf].astype(str).str.strip() if col_uf else "",
            COL_LAT: df_raw[col_lat].map(parse_float),
            COL_LON: df_raw[col_lon].map(parse_float),
            "source_file_2019": df_raw[col_file].astype(str).str.strip(),
        }
    )

    df_meta = df_meta.dropna(subset=[COL_STATION, COL_LAT, COL_LON])
    df_meta = df_meta[df_meta[COL_STATION].str.fullmatch(r"[A-Z]\d{3}")]
    df_meta = df_meta.drop_duplicates(subset=[COL_STATION]).reset_index(drop=True)
    return df_meta


def resolve_hourly_columns(df_hourly: pd.DataFrame) -> Dict[str, str]:
    columns_norm = {col: normalize_text(col) for col in df_hourly.columns}
    precip_col = (
        choose_first_column(columns_norm, ["precipitacao", "total"], required=False)
        or choose_first_column(columns_norm, ["precipitacao", "horaria"], required=False)
        or choose_first_column(columns_norm, ["chuva"], required=False)
    )
    humidity_col = (
        choose_first_column(columns_norm, ["umidade", "relativa", "ar"], required=False)
        or choose_first_column(columns_norm, ["umidade", "horaria"], required=False)
    )
    dewpoint_col = (
        choose_first_column(columns_norm, ["ponto", "orvalho"], required=False)
        or choose_first_column(columns_norm, ["orvalho"], required=False)
    )
    wind_dir_col = (
        choose_first_column(columns_norm, ["vento", "direcao", "horaria"], required=False)
        or choose_first_column(columns_norm, ["vento", "direcao"], required=False)
    )
    wind_speed_col = (
        choose_first_column(columns_norm, ["vento", "velocidade", "horaria"], required=False)
        or choose_first_column(columns_norm, ["vento", "velocidade"], required=False)
    )

    return {
        "date": choose_first_column(columns_norm, ["data"]),
        "precip": precip_col,
        "humidity": humidity_col,
        "dewpoint": dewpoint_col,
        "pressure": choose_first_column(columns_norm, ["pressao", "nivel", "estacao"], required=False)
        or choose_first_column(columns_norm, ["pressao", "min", "hora", "aut"], required=False),
        "wind_dir": wind_dir_col,
        "wind_speed": wind_speed_col,
    }


def aggregate_station_daily(
    file_path: Path,
    station_code: str,
    latitude: float,
    longitude: float,
) -> Optional[pd.DataFrame]:
    df_hourly = None
    cols = None
    for enc in ("utf-8", "latin1"):
        try:
            candidate = pd.read_csv(file_path, sep=";", dtype=str, encoding=enc)
        except Exception:
            continue
        try:
            candidate_cols = resolve_hourly_columns(candidate)
        except KeyError:
            continue
        required = [
            candidate_cols.get("date"),
            candidate_cols.get("precip"),
            candidate_cols.get("humidity"),
            candidate_cols.get("dewpoint"),
            candidate_cols.get("wind_dir"),
            candidate_cols.get("wind_speed"),
        ]
        if any(c is None for c in required):
            continue
        df_hourly = candidate
        cols = candidate_cols
        break

    if df_hourly is None or cols is None:
        return None

    df = pd.DataFrame(
        {
            COL_DATE: pd.to_datetime(df_hourly[cols["date"]], errors="coerce"),
            COL_PRECIP: pd.to_numeric(df_hourly[cols["precip"]], errors="coerce"),
            COL_HUMIDITY: pd.to_numeric(df_hourly[cols["humidity"]], errors="coerce"),
            COL_DEWPOINT: pd.to_numeric(df_hourly[cols["dewpoint"]], errors="coerce"),
            COL_PRESSURE: pd.to_numeric(df_hourly[cols["pressure"]], errors="coerce")
            if cols["pressure"] is not None
            else np.nan,
            COL_WIND_DIR: pd.to_numeric(df_hourly[cols["wind_dir"]], errors="coerce"),
            COL_WIND_SPEED: pd.to_numeric(df_hourly[cols["wind_speed"]], errors="coerce"),
        }
    )

    # Basic physical ranges.
    df.loc[df[COL_PRECIP] < 0, COL_PRECIP] = np.nan
    df.loc[(df[COL_HUMIDITY] < 0) | (df[COL_HUMIDITY] > 100), COL_HUMIDITY] = np.nan
    df.loc[(df[COL_WIND_DIR] < 0) | (df[COL_WIND_DIR] > 360), COL_WIND_DIR] = np.nan
    df.loc[df[COL_WIND_SPEED] < 0, COL_WIND_SPEED] = np.nan
    df.loc[(df[COL_PRESSURE] < 700) | (df[COL_PRESSURE] > 1100), COL_PRESSURE] = np.nan

    df = df.dropna(subset=[COL_DATE])
    if df.empty:
        return None

    df[COL_DATE] = df[COL_DATE].dt.normalize()
    df = df.sort_values(COL_DATE)

    grouped = df.groupby(COL_DATE)
    df_daily = pd.DataFrame(
        {
            COL_DATE: grouped.size().index,
            COL_PRECIP: grouped[COL_PRECIP].sum(min_count=1),
            COL_HUMIDITY: grouped[COL_HUMIDITY].mean(),
            COL_DEWPOINT: grouped[COL_DEWPOINT].mean(),
            COL_PRESSURE: grouped[COL_PRESSURE].mean(),
            COL_WIND_SPEED: grouped[COL_WIND_SPEED].mean(),
            COL_WIND_DIR: grouped[COL_WIND_DIR].apply(circular_mean_deg),
        }
    ).reset_index(drop=True)

    df_daily[COL_STATION] = station_code
    df_daily[COL_LAT] = latitude
    df_daily[COL_LON] = longitude
    return df_daily


def build_daily_panel(
    stations_csv: Path,
    cleaned_data_root: Path,
    years: Sequence[int] = DEFAULT_YEARS,
    station_codes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    meta = load_station_metadata(stations_csv)
    meta_idx = meta.set_index(COL_STATION)
    station_code_filter = set(station_codes) if station_codes is not None else None
    all_daily: List[pd.DataFrame] = []

    for year in years:
        year_dir = cleaned_data_root / f"{year}_cleaned"
        if not year_dir.exists():
            continue

        for file_path in sorted(year_dir.glob("*.CSV")):
            code = extract_station_code(file_path.name)
            if not code or code not in meta_idx.index:
                continue
            if station_code_filter is not None and code not in station_code_filter:
                continue

            row = meta_idx.loc[code]
            df_daily = aggregate_station_daily(
                file_path=file_path,
                station_code=code,
                latitude=float(row[COL_LAT]),
                longitude=float(row[COL_LON]),
            )
            if df_daily is not None and not df_daily.empty:
                all_daily.append(df_daily)

    if not all_daily:
        raise RuntimeError("No daily records built from cleaned data.")

    panel = pd.concat(all_daily, ignore_index=True)
    panel = panel.sort_values([COL_STATION, COL_DATE])

    # In case duplicate daily rows exist for a station/date, keep aggregated mean/sum.
    grouped = panel.groupby([COL_STATION, COL_DATE], as_index=False)
    panel = grouped.agg(
        {
            COL_LAT: "first",
            COL_LON: "first",
            COL_PRECIP: "sum",
            COL_HUMIDITY: "mean",
            COL_DEWPOINT: "mean",
            COL_PRESSURE: "mean",
            COL_WIND_DIR: circular_mean_deg,
            COL_WIND_SPEED: "mean",
        }
    )
    return panel


def validate_panel_ranges(panel: pd.DataFrame) -> pd.DataFrame:
    checks = []

    checks.append(
        {
            "check": "precip_non_negative",
            "violations": int((panel[COL_PRECIP] < 0).sum()),
        }
    )
    checks.append(
        {
            "check": "humidity_0_100",
            "violations": int(((panel[COL_HUMIDITY] < 0) | (panel[COL_HUMIDITY] > 100)).sum()),
        }
    )
    checks.append(
        {
            "check": "wind_dir_0_360",
            "violations": int(((panel[COL_WIND_DIR] < 0) | (panel[COL_WIND_DIR] > 360)).sum()),
        }
    )
    checks.append(
        {
            "check": "wind_speed_non_negative",
            "violations": int((panel[COL_WIND_SPEED] < 0).sum()),
        }
    )
    return pd.DataFrame(checks)


def station_coverage_report(panel: pd.DataFrame, station_code: str) -> pd.DataFrame:
    station = panel[panel[COL_STATION] == station_code].copy()
    if station.empty:
        raise ValueError(f"Station {station_code} not found in panel.")

    station = station.sort_values(COL_DATE)
    full_idx = pd.date_range(station[COL_DATE].min(), station[COL_DATE].max(), freq="D")
    missing_days = int(len(full_idx) - station[COL_DATE].nunique())

    report = pd.DataFrame(
        [
            {"metric": "start_date", "value": str(station[COL_DATE].min().date())},
            {"metric": "end_date", "value": str(station[COL_DATE].max().date())},
            {"metric": "observed_days", "value": int(station[COL_DATE].nunique())},
            {"metric": "expected_days", "value": int(len(full_idx))},
            {"metric": "missing_days", "value": missing_days},
        ]
    )
    return report


def station_coverage_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for code, g in panel.groupby(COL_STATION):
        g = g.sort_values(COL_DATE)
        start = g[COL_DATE].min()
        end = g[COL_DATE].max()
        full_idx = pd.date_range(start, end, freq="D")
        observed_days = int(g[COL_DATE].nunique())
        expected_days = int(len(full_idx))
        missing_days = int(expected_days - observed_days)
        rows.append(
            {
                COL_STATION: code,
                "start_date": str(start.date()),
                "end_date": str(end.date()),
                "observed_days": observed_days,
                "expected_days": expected_days,
                "missing_days": missing_days,
                "missing_pct": (missing_days / expected_days * 100.0) if expected_days > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(COL_STATION).reset_index(drop=True)


def resolve_target_station(
    metadata: pd.DataFrame,
    target_coord: Tuple[float, float],
    target_station_code: Optional[str] = None,
) -> str:
    if target_station_code:
        if target_station_code not in set(metadata[COL_STATION]):
            raise ValueError(f"Target station {target_station_code} not found in metadata.")
        return target_station_code

    lat_t, lon_t = target_coord
    distances = metadata.apply(
        lambda row: haversine_km(lat_t, lon_t, float(row[COL_LAT]), float(row[COL_LON])),
        axis=1,
    )
    idx = int(distances.idxmin())
    return str(metadata.loc[idx, COL_STATION])


def select_station_codes_by_distance(
    metadata: pd.DataFrame,
    target_coord: Tuple[float, float],
    max_dist_km: float,
    target_station_code: Optional[str] = None,
) -> List[str]:
    lat_t, lon_t = target_coord
    codes: List[str] = []
    for _, row in metadata.iterrows():
        dist = haversine_km(lat_t, lon_t, float(row[COL_LAT]), float(row[COL_LON]))
        if dist <= max_dist_km:
            codes.append(str(row[COL_STATION]))
    if target_station_code and target_station_code not in codes:
        codes.append(target_station_code)
    return sorted(set(codes))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_fixed_radius_features(
    target_coord: Tuple[float, float],
    daily_panel: pd.DataFrame,
    radius_km: float = 35.0,
    target_station_code: Optional[str] = None,
) -> pd.DataFrame:
    metadata = (
        daily_panel[[COL_STATION, COL_LAT, COL_LON]]
        .drop_duplicates(subset=[COL_STATION])
        .reset_index(drop=True)
    )
    target_station = resolve_target_station(metadata, target_coord, target_station_code)

    lat_t, lon_t = target_coord
    metadata["distance_km"] = metadata.apply(
        lambda row: haversine_km(lat_t, lon_t, float(row[COL_LAT]), float(row[COL_LON])),
        axis=1,
    )

    radius_stations = set(
        metadata[(metadata["distance_km"] <= radius_km) & (metadata[COL_STATION] != target_station)][COL_STATION]
    )

    subset = daily_panel[daily_panel[COL_STATION].isin(radius_stations)].copy()
    if subset.empty:
        empty = (
            daily_panel[daily_panel[COL_STATION] == target_station][[COL_DATE]]
            .drop_duplicates()
            .sort_values(COL_DATE)
            .copy()
        )
        for out_col in RADIUS_COLS:
            empty[out_col] = np.nan
        empty["radius_station_count"] = 0
        empty["radius_station_codes"] = ""
        return empty.set_index(COL_DATE)

    grouped = subset.groupby(COL_DATE, as_index=False).agg(
        {
            COL_HUMIDITY: "mean",
            COL_DEWPOINT: "mean",
            COL_PRESSURE: "mean",
            COL_PRECIP: "mean",
            COL_STATION: lambda s: "|".join(sorted(set(map(str, s)))),
        }
    )

    grouped = grouped.rename(
        columns={
            COL_HUMIDITY: f"radius_{COL_HUMIDITY}",
            COL_DEWPOINT: f"radius_{COL_DEWPOINT}",
            COL_PRESSURE: f"radius_{COL_PRESSURE}",
            COL_PRECIP: f"radius_{COL_PRECIP}",
            COL_STATION: "radius_station_codes",
        }
    )
    grouped["radius_station_count"] = grouped["radius_station_codes"].map(
        lambda text: 0 if text == "" else len(text.split("|"))
    )
    return grouped.set_index(COL_DATE).sort_index()


def build_upwind_features(
    target_coord: Tuple[float, float],
    daily_panel: pd.DataFrame,
    angle_tol_deg: float = 35.0,
    max_dist_km: float = 400.0,
    min_wind_ms: float = 1.5,
    target_station_code: Optional[str] = None,
    min_lag_days: int = 0,
    max_lag_days: int = 5,
    wind_dir_is_from: bool = True,
) -> pd.DataFrame:
    """
    Build dynamic upwind features for a target location and station panel.

    Returns a DataFrame indexed by date with:
    - upwind_humidity_lag
    - upwind_dewpoint_lag
    - upwind_pressure_lag
    - upwind_precip_lag
    - upwind_station_count
    - upwind_selected_station_codes
    - upwind_mean_lag_days
    """

    panel = daily_panel.copy()
    panel[COL_DATE] = pd.to_datetime(panel[COL_DATE]).dt.normalize()

    metadata = (
        panel[[COL_STATION, COL_LAT, COL_LON]]
        .drop_duplicates(subset=[COL_STATION])
        .reset_index(drop=True)
    )
    target_station = resolve_target_station(metadata, target_coord, target_station_code)

    target_df = panel[panel[COL_STATION] == target_station].copy().sort_values(COL_DATE)
    if target_df.empty:
        raise ValueError(f"No daily data found for target station {target_station}.")
    target_df = target_df.set_index(COL_DATE)

    candidates = metadata[metadata[COL_STATION] != target_station].copy()
    lat_t, lon_t = target_coord
    candidates["distance_km"] = candidates.apply(
        lambda row: haversine_km(float(row[COL_LAT]), float(row[COL_LON]), lat_t, lon_t),
        axis=1,
    )
    candidates["bearing_to_target_deg"] = candidates.apply(
        lambda row: bearing_deg(float(row[COL_LAT]), float(row[COL_LON]), lat_t, lon_t),
        axis=1,
    )
    candidates = candidates[candidates["distance_km"] <= max_dist_km].copy()

    candidate_codes = set(candidates[COL_STATION])
    panel_candidates = panel[panel[COL_STATION].isin(candidate_codes)].copy()
    panel_candidates = panel_candidates.set_index([COL_STATION, COL_DATE]).sort_index()

    geom_idx = candidates.set_index(COL_STATION)
    out_rows: List[Dict[str, object]] = []

    for day, row_target in target_df.iterrows():
        wind_dir_target = float(row_target.get(COL_WIND_DIR, np.nan))
        wind_speed_target = float(row_target.get(COL_WIND_SPEED, np.nan))

        day_result: Dict[str, object] = {
            COL_DATE: day,
            "upwind_humidity_lag": np.nan,
            "upwind_dewpoint_lag": np.nan,
            "upwind_pressure_lag": np.nan,
            "upwind_precip_lag": np.nan,
            "upwind_station_count": 0,
            "upwind_selected_station_codes": "",
            "upwind_mean_lag_days": np.nan,
            "upwind_total_weight": 0.0,
        }

        if not np.isfinite(wind_dir_target):
            out_rows.append(day_result)
            continue

        flow_to_target_deg = (
            (wind_dir_target + 180.0) % 360.0 if wind_dir_is_from else wind_dir_target
        )
        wind_speed_for_lag_ms = wind_speed_target if np.isfinite(wind_speed_target) else min_wind_ms
        wind_speed_for_lag_ms = max(wind_speed_for_lag_ms, min_wind_ms)
        wind_km_day = wind_speed_for_lag_ms * 86.4

        selected_codes: List[str] = []
        selected_lags: List[int] = []
        accum = {
            "humidity": 0.0,
            "dewpoint": 0.0,
            "pressure": 0.0,
            "precip": 0.0,
            "weight_humidity": 0.0,
            "weight_dewpoint": 0.0,
            "weight_pressure": 0.0,
            "weight_precip": 0.0,
            "weight_total": 0.0,
        }

        for code in candidate_codes:
            geom = geom_idx.loc[code]
            dist = float(geom["distance_km"])
            bearing = float(geom["bearing_to_target_deg"])

            diff = angular_diff_deg(flow_to_target_deg, bearing)
            if diff > angle_tol_deg:
                continue

            lag_days = int(round(dist / max(wind_km_day, 1e-6)))
            lag_days = int(np.clip(lag_days, min_lag_days, max_lag_days))
            source_day = day - pd.Timedelta(days=lag_days)

            try:
                source_row = panel_candidates.loc[(code, source_day)]
            except KeyError:
                continue

            align_weight = max(0.0, math.cos(math.radians(diff)))
            distance_weight = math.exp(-dist / max_dist_km)
            speed_weight = float(np.clip(wind_speed_target / (min_wind_ms * 2.0), 0.15, 1.5)) if np.isfinite(wind_speed_target) else 0.15
            weight = align_weight * distance_weight * speed_weight

            if weight <= 0:
                continue

            selected_codes.append(code)
            selected_lags.append(lag_days)
            accum["weight_total"] += weight

            val_h = float(source_row.get(COL_HUMIDITY, np.nan))
            if np.isfinite(val_h):
                accum["humidity"] += weight * val_h
                accum["weight_humidity"] += weight

            val_d = float(source_row.get(COL_DEWPOINT, np.nan))
            if np.isfinite(val_d):
                accum["dewpoint"] += weight * val_d
                accum["weight_dewpoint"] += weight

            val_p = float(source_row.get(COL_PRESSURE, np.nan))
            if np.isfinite(val_p):
                accum["pressure"] += weight * val_p
                accum["weight_pressure"] += weight

            val_r = float(source_row.get(COL_PRECIP, np.nan))
            if np.isfinite(val_r):
                accum["precip"] += weight * val_r
                accum["weight_precip"] += weight

        if selected_codes:
            day_result["upwind_station_count"] = len(selected_codes)
            day_result["upwind_selected_station_codes"] = "|".join(sorted(selected_codes))
            day_result["upwind_mean_lag_days"] = float(np.mean(selected_lags))
            day_result["upwind_total_weight"] = float(accum["weight_total"])

            if accum["weight_humidity"] > 0:
                day_result["upwind_humidity_lag"] = accum["humidity"] / accum["weight_humidity"]
            if accum["weight_dewpoint"] > 0:
                day_result["upwind_dewpoint_lag"] = accum["dewpoint"] / accum["weight_dewpoint"]
            if accum["weight_pressure"] > 0:
                day_result["upwind_pressure_lag"] = accum["pressure"] / accum["weight_pressure"]
            if accum["weight_precip"] > 0:
                day_result["upwind_precip_lag"] = accum["precip"] / accum["weight_precip"]

        out_rows.append(day_result)

    out = pd.DataFrame(out_rows)
    out = out.sort_values(COL_DATE).set_index(COL_DATE)
    return out


def build_feature_table(
    daily_panel: pd.DataFrame,
    target_coord: Tuple[float, float],
    target_station_code: Optional[str] = None,
    radius_km: float = 35.0,
    upwind_cfg: Optional[UpwindFeatureConfig] = None,
) -> Tuple[pd.DataFrame, str]:
    upwind_cfg = upwind_cfg or UpwindFeatureConfig()

    metadata = (
        daily_panel[[COL_STATION, COL_LAT, COL_LON]]
        .drop_duplicates(subset=[COL_STATION])
        .reset_index(drop=True)
    )
    target_station = resolve_target_station(metadata, target_coord, target_station_code)

    local_df = (
        daily_panel[daily_panel[COL_STATION] == target_station]
        .copy()
        .sort_values(COL_DATE)
        .set_index(COL_DATE)
    )
    local_df = local_df.rename(
        columns={
            COL_PRECIP: f"local_{COL_PRECIP}",
            COL_HUMIDITY: f"local_{COL_HUMIDITY}",
            COL_DEWPOINT: f"local_{COL_DEWPOINT}",
            COL_PRESSURE: f"local_{COL_PRESSURE}",
            COL_WIND_DIR: f"local_{COL_WIND_DIR}",
            COL_WIND_SPEED: f"local_{COL_WIND_SPEED}",
        }
    )

    radius_features = build_fixed_radius_features(
        target_coord=target_coord,
        daily_panel=daily_panel,
        radius_km=radius_km,
        target_station_code=target_station,
    )

    upwind_features = build_upwind_features(
        target_coord=target_coord,
        daily_panel=daily_panel,
        angle_tol_deg=upwind_cfg.angle_tol_deg,
        max_dist_km=upwind_cfg.max_dist_km,
        min_wind_ms=upwind_cfg.min_wind_ms,
        target_station_code=target_station,
        min_lag_days=upwind_cfg.min_lag_days,
        max_lag_days=upwind_cfg.max_lag_days,
        wind_dir_is_from=upwind_cfg.wind_dir_is_from,
    )

    out = local_df.join(radius_features, how="left").join(upwind_features, how="left")
    out.index = pd.to_datetime(out.index).normalize()
    out = out.sort_index()

    # Fallbacks for days without upwind stations.
    if f"local_{COL_HUMIDITY}" in out.columns:
        out["upwind_humidity_lag"] = out["upwind_humidity_lag"].fillna(out[f"local_{COL_HUMIDITY}"])
    if f"local_{COL_DEWPOINT}" in out.columns:
        out["upwind_dewpoint_lag"] = out["upwind_dewpoint_lag"].fillna(out[f"local_{COL_DEWPOINT}"])
    if f"local_{COL_PRESSURE}" in out.columns:
        out["upwind_pressure_lag"] = out["upwind_pressure_lag"].fillna(out[f"local_{COL_PRESSURE}"])
    if f"local_{COL_PRECIP}" in out.columns:
        out["upwind_precip_lag"] = out["upwind_precip_lag"].fillna(out[f"local_{COL_PRECIP}"])

    return out, target_station


# ---------------------------------------------------------------------------
# Modeling and backtest
# ---------------------------------------------------------------------------


def fill_exog_from_train_climatology(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    # Copy to avoid inplace mutations.
    train_part = train_df.copy()
    test_part = test_df.copy()

    profile = train_part.copy()
    profile["__mm"] = profile.index.month
    profile["__dd"] = profile.index.day
    profile_mmdd = profile.groupby(["__mm", "__dd"])[list(feature_cols)].mean()

    test_idx = pd.MultiIndex.from_arrays(
        [test_part.index.month, test_part.index.day], names=["__mm", "__dd"]
    )
    fallback = profile_mmdd.reindex(test_idx)
    fallback.index = test_part.index

    for col in feature_cols:
        if col not in test_part.columns:
            test_part[col] = np.nan
        test_part[col] = test_part[col].fillna(fallback[col])
        fill_value = train_part[col].mean()
        if not np.isfinite(fill_value):
            fill_value = 0.0
        test_part[col] = test_part[col].fillna(fill_value)

    return test_part


def walk_forward_direct_arimax(
    df_features: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    horizon_days: int,
    model_cfg: BacktestConfig,
    method_name: str,
) -> pd.DataFrame:
    df = df_features.copy().sort_index()
    df["target_future"] = df[target_col].shift(-horizon_days)
    df = df.dropna(subset=["target_future"])

    # Basic exogenous completeness in training side is handled by forward-fill + mean.
    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"Feature column '{col}' missing for method={method_name}.")
        # Forward-only fill to avoid future leakage.
        df[col] = df[col].ffill()

    test_start = pd.Timestamp(model_cfg.test_start).normalize()
    eval_dates = df.index[df.index >= test_start]
    if len(eval_dates) == 0:
        raise ValueError(
            f"No test rows for method={method_name}, horizon={horizon_days}, test_start={model_cfg.test_start}."
        )

    predictions: List[Dict[str, object]] = []
    last_fit_date: Optional[pd.Timestamp] = None
    model_fit = None
    t_global = np.arange(len(df))
    fourier_all = create_fourier_terms(
        t_global, model_cfg.seasonality_days, model_cfg.fourier_terms
    )

    for current_date in eval_dates:
        if last_fit_date is None:
            should_refit = True
        else:
            should_refit = (current_date - last_fit_date).days >= model_cfg.retrain_every_days

        # Anti-leakage: train rows must satisfy t + horizon < current_date.
        cutoff_date = current_date - pd.Timedelta(days=horizon_days)
        train_mask = df.index < cutoff_date
        train_df = df.loc[train_mask].copy()

        if len(train_df) < max(model_cfg.min_train_rows, 2 * model_cfg.fourier_terms + 5):
            continue
        if (current_date - train_df.index.min()).days < model_cfg.min_train_days:
            continue

        if should_refit or model_fit is None:
            train_idx_positions = np.where(df.index.isin(train_df.index))[0]
            train_exog = train_df[list(feature_cols)].copy()
            for col in feature_cols:
                train_exog[col] = train_exog[col].ffill()
                fill_value = train_exog[col].mean()
                if not np.isfinite(fill_value):
                    fill_value = 0.0
                train_exog[col] = train_exog[col].fillna(fill_value)

            X_train = np.hstack([train_exog.to_numpy(), fourier_all[train_idx_positions]])
            y_train = train_df["target_future"].to_numpy()

            try:
                model = ARIMA(
                    y_train,
                    exog=X_train,
                    order=model_cfg.order,
                    trend=model_cfg.trend,
                )
                model_fit = model.fit()
                last_fit_date = current_date
            except Exception:
                model_fit = None

        if model_fit is None:
            continue

        test_row = df.loc[[current_date]].copy()
        test_row = fill_exog_from_train_climatology(train_df, test_row, feature_cols)
        test_pos = int(np.where(df.index == current_date)[0][0])
        X_test = np.hstack([test_row[list(feature_cols)].to_numpy(), fourier_all[[test_pos]]])

        try:
            pred = float(model_fit.forecast(steps=1, exog=X_test)[0])
        except Exception:
            continue

        y_true = float(test_row["target_future"].iloc[0])
        pred = max(0.0, pred)
        predictions.append(
            {
                "date": current_date,
                "horizon": horizon_days,
                "method": method_name,
                "y_true": y_true,
                "y_pred": pred,
                "rain_true": int(y_true > 0),
                "rain_pred": int(pred > 0),
            }
        )

    return pd.DataFrame(predictions)


def compute_method_metrics(pred_df: pd.DataFrame, train_reference: pd.Series) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "horizon",
                "n_obs",
                "rmse",
                "mae",
                "smape",
                "f1_rain",
                "rmse_p95",
                "mae_p95",
            ]
        )

    p95 = float(np.nanpercentile(train_reference.to_numpy(), 95))
    rows: List[Dict[str, object]] = []
    grouped = pred_df.groupby(["method", "horizon"])
    for (method, horizon), g in grouped:
        y_true = g["y_true"].to_numpy(dtype=float)
        y_pred = g["y_pred"].to_numpy(dtype=float)

        peak_mask = y_true >= p95
        if np.any(peak_mask):
            rmse_peak = rmse_np(y_true[peak_mask], y_pred[peak_mask])
            mae_peak = mean_absolute_error_np(y_true[peak_mask], y_pred[peak_mask])
        else:
            rmse_peak = np.nan
            mae_peak = np.nan

        rows.append(
            {
                "method": method,
                "horizon": int(horizon),
                "n_obs": int(len(g)),
                "rmse": rmse_np(y_true, y_pred),
                "mae": mean_absolute_error_np(y_true, y_pred),
                "smape": smape_np(y_true, y_pred),
                "f1_rain": f1_binary_np(
                    g["rain_true"].to_numpy(dtype=int),
                    g["rain_pred"].to_numpy(dtype=int),
                ),
                "rmse_p95": rmse_peak,
                "mae_p95": mae_peak,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "method"]).reset_index(drop=True)


def run_backtest_precip(
    df_features: pd.DataFrame,
    horizon_days: Sequence[int] = (1, 2, 3),
    model_cfg: Optional[BacktestConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run walk-forward backtest comparing:
    - baseline: local exogenous + Fourier
    - fixed_radius: local + radius exogenous + Fourier
    - upwind_dynamic: local + upwind lagged exogenous + Fourier

    Returns:
    {
      "predictions": DataFrame(date, horizon, method, y_true, y_pred, ...),
      "metrics": DataFrame(comparative metrics by method+horizon)
    }
    """

    model_cfg = model_cfg or BacktestConfig()
    df = df_features.copy().sort_index()

    required_cols = [TARGET_COL] + LOCAL_COLS + RADIUS_COLS + UPWIND_COLS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in feature table: {missing}")

    method_features = {
        "baseline": LOCAL_COLS,
        "fixed_radius": LOCAL_COLS + RADIUS_COLS,
        "upwind_dynamic": LOCAL_COLS + UPWIND_COLS,
    }

    all_predictions: List[pd.DataFrame] = []
    for h in horizon_days:
        for method_name, feat_cols in method_features.items():
            pred = walk_forward_direct_arimax(
                df_features=df,
                target_col=TARGET_COL,
                feature_cols=feat_cols,
                horizon_days=int(h),
                model_cfg=model_cfg,
                method_name=method_name,
            )
            all_predictions.append(pred)

    pred_df = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame(columns=["date", "horizon", "method", "y_true", "y_pred", "rain_true", "rain_pred"])
    )
    pred_df = pred_df.sort_values(["date", "horizon", "method"]).reset_index(drop=True)

    train_ref = df[df.index < pd.Timestamp(model_cfg.test_start)][TARGET_COL]
    if train_ref.empty:
        train_ref = df[TARGET_COL]

    metrics_df = compute_method_metrics(pred_df, train_ref)
    return {"predictions": pred_df, "metrics": metrics_df}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def build_upwind_audit_table(df_features: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "upwind_station_count",
        "upwind_selected_station_codes",
        "upwind_mean_lag_days",
        "upwind_total_weight",
    ]
    missing = [c for c in cols if c not in df_features.columns]
    if missing:
        raise KeyError(f"Missing audit columns in df_features: {missing}")

    audit = df_features[cols].copy()
    audit = audit.reset_index().rename(columns={"index": "date"})
    audit["date"] = pd.to_datetime(audit["date"]).dt.normalize()
    return audit


def save_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    upwind_audit: pd.DataFrame,
    panel_range_report: pd.DataFrame,
    target_coverage_report: pd.DataFrame,
    station_coverage_summary_report: pd.DataFrame,
    range_validation_report: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(output_dir / "predictions_by_horizon.csv", index=False)
    metrics.to_csv(output_dir / "metrics_comparison.csv", index=False)
    upwind_audit.to_csv(output_dir / "upwind_selected_stations_daily.csv", index=False)
    panel_range_report.to_csv(output_dir / "panel_period_report.csv", index=False)
    target_coverage_report.to_csv(output_dir / "target_coverage_report.csv", index=False)
    station_coverage_summary_report.to_csv(
        output_dir / "station_coverage_summary.csv", index=False
    )
    range_validation_report.to_csv(output_dir / "range_validation_report.csv", index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="INMET-only precipitation forecast backtest with dynamic upwind features (UnB/DF)."
    )
    parser.add_argument(
        "--stations-csv",
        type=Path,
        default=Path("data/all_stations.csv"),
        help="Path to station metadata CSV.",
    )
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path("data/cleaned_data"),
        help="Path to cleaned data root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/Results/upwind_precip"),
        help="Directory where output CSV files are written.",
    )
    parser.add_argument(
        "--target-station",
        type=str,
        default=None,
        help="Optional target station code (e.g., A001). If omitted, nearest station to --target-lat/--target-lon is used.",
    )
    parser.add_argument("--target-lat", type=float, default=TARGET_COORD_UNB_DF[0], help="Target latitude.")
    parser.add_argument("--target-lon", type=float, default=TARGET_COORD_UNB_DF[1], help="Target longitude.")
    parser.add_argument(
        "--panel-load-radius-km",
        type=float,
        default=500.0,
        help="Pre-filter stations loaded from disk by distance to target (km).",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
        help="Years to load from cleaned_data/<year>_cleaned.",
    )
    parser.add_argument("--radius-km", type=float, default=35.0, help="Fixed-radius baseline in km.")
    parser.add_argument("--angle-tol-deg", type=float, default=35.0, help="Upwind angular tolerance in degrees.")
    parser.add_argument("--max-dist-km", type=float, default=400.0, help="Max upwind candidate distance (km).")
    parser.add_argument("--min-wind-ms", type=float, default=1.5, help="Min wind speed for lag estimation (m/s).")
    parser.add_argument("--min-lag-days", type=int, default=0, help="Min lag in days for transit.")
    parser.add_argument("--max-lag-days", type=int, default=5, help="Max lag in days for transit.")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Forecast horizons in days.",
    )
    parser.add_argument("--test-start", type=str, default="2023-01-01", help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--retrain-every", type=int, default=7, help="Refit ARIMAX every N evaluation days.")
    parser.add_argument("--min-train-days", type=int, default=730, help="Minimum train span in days.")
    parser.add_argument("--min-train-rows", type=int, default=365, help="Minimum train rows for ARIMAX fit.")
    parser.add_argument(
        "--order",
        type=int,
        nargs=3,
        default=[3, 0, 1],
        metavar=("P", "D", "Q"),
        help="ARIMA order (p d q).",
    )
    parser.add_argument("--trend", type=str, default="ct", help="ARIMA trend argument.")
    parser.add_argument("--seasonality-days", type=float, default=365.25, help="Fourier seasonal period.")
    parser.add_argument("--fourier-terms", type=int, default=4, help="Number of Fourier harmonics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    upwind_cfg = UpwindFeatureConfig(
        angle_tol_deg=args.angle_tol_deg,
        max_dist_km=args.max_dist_km,
        min_wind_ms=args.min_wind_ms,
        min_lag_days=args.min_lag_days,
        max_lag_days=args.max_lag_days,
        wind_dir_is_from=True,
    )
    model_cfg = BacktestConfig(
        order=tuple(args.order),
        trend=args.trend,
        seasonality_days=args.seasonality_days,
        fourier_terms=args.fourier_terms,
        test_start=args.test_start,
        retrain_every_days=args.retrain_every,
        min_train_days=args.min_train_days,
        min_train_rows=args.min_train_rows,
    )

    metadata = load_station_metadata(args.stations_csv)
    target_coord = (args.target_lat, args.target_lon)
    target_station = resolve_target_station(metadata, target_coord, args.target_station)
    station_codes = select_station_codes_by_distance(
        metadata=metadata,
        target_coord=target_coord,
        max_dist_km=args.panel_load_radius_km,
        target_station_code=target_station,
    )

    panel = build_daily_panel(
        stations_csv=args.stations_csv,
        cleaned_data_root=args.cleaned_root,
        years=args.years,
        station_codes=station_codes,
    )

    df_features, target_station_resolved = build_feature_table(
        daily_panel=panel,
        target_coord=target_coord,
        target_station_code=target_station,
        radius_km=args.radius_km,
        upwind_cfg=upwind_cfg,
    )

    result = run_backtest_precip(
        df_features=df_features,
        horizon_days=args.horizons,
        model_cfg=model_cfg,
    )

    upwind_audit = build_upwind_audit_table(df_features)
    panel_period_report = pd.DataFrame(
        [
            {"metric": "panel_start", "value": str(panel[COL_DATE].min().date())},
            {"metric": "panel_end", "value": str(panel[COL_DATE].max().date())},
            {"metric": "panel_station_count", "value": int(panel[COL_STATION].nunique())},
            {"metric": "target_station", "value": target_station_resolved},
        ]
    )
    coverage_report = station_coverage_report(panel, target_station_resolved)
    coverage_summary = station_coverage_summary(panel)
    range_validation = validate_panel_ranges(panel)

    save_outputs(
        output_dir=args.output_dir,
        predictions=result["predictions"],
        metrics=result["metrics"],
        upwind_audit=upwind_audit,
        panel_range_report=panel_period_report,
        target_coverage_report=coverage_report,
        station_coverage_summary_report=coverage_summary,
        range_validation_report=range_validation,
    )

    print(f"Target station: {target_station_resolved}")
    print(f"Feature rows: {len(df_features)}")
    print(f"Predictions: {len(result['predictions'])}")
    print(f"Metrics rows: {len(result['metrics'])}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
