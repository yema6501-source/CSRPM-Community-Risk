from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from .utils import ensure_dir


CHICAGO_DOMAIN = "https://data.cityofchicago.org"
CRIME_DATASET = "ijzp-q8t2"
SANITATION_DATASET = "me59-5fac"
ABANDONED_VEHICLES_DATASET = "3c9v-pnva"
SOCIOECONOMIC_DATASET = "kn9c-c2s2"
COMMUNITY_AREAS_GEOJSON = (
    "https://data.cityofchicago.org/resource/igwz-8jzy.geojson"
)


@dataclass
class ChicagoDataConfig:
    start_date: str = "2022-01-01"
    end_date: str = "2025-12-31"
    frequency: str = "W-MON"
    page_size: int = 50000
    timeout: int = 60


class SocrataClient:
    def __init__(self, domain: str = CHICAGO_DOMAIN, app_token: Optional[str] = None, timeout: int = 60):
        self.domain = domain.rstrip("/")
        self.app_token = app_token
        self.timeout = timeout

    def get(
        self,
        dataset_id: str,
        select: str,
        where: Optional[str] = None,
        order: Optional[str] = None,
        page_size: int = 50000,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Paginate a Socrata dataset through the public SODA endpoint."""
        url = f"{self.domain}/resource/{dataset_id}.json"
        headers = {"X-App-Token": self.app_token} if self.app_token else {}
        frames: List[pd.DataFrame] = []
        offset = 0

        while True:
            limit = page_size if max_rows is None else min(page_size, max_rows - offset)
            if limit <= 0:
                break
            params = {"$select": select, "$limit": limit, "$offset": offset}
            if where:
                params["$where"] = where
            if order:
                params["$order"] = order
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            rows = response.json()
            if not rows:
                break
            frames.append(pd.DataFrame(rows))
            offset += len(rows)
            if len(rows) < limit:
                break

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def _date_where(field: str, start_date: str, end_date: str) -> str:
    return (
        f"{field} >= '{start_date}T00:00:00.000' AND "
        f"{field} <= '{end_date}T23:59:59.999'"
    )


def download_chicago_crimes(client: SocrataClient, cfg: ChicagoDataConfig) -> pd.DataFrame:
    select = "date,community_area,primary_type,arrest,domestic"
    df = client.get(
        CRIME_DATASET,
        select=select,
        where=_date_where("date", cfg.start_date, cfg.end_date) + " AND community_area IS NOT NULL",
        order="date ASC",
        page_size=cfg.page_size,
    )
    if df.empty:
        raise RuntimeError("Chicago crime query returned no rows.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["community_area"] = pd.to_numeric(df["community_area"], errors="coerce").astype("Int64")
    df["arrest"] = df["arrest"].astype(str).str.lower().eq("true").astype(int)
    df["domestic"] = df["domestic"].astype(str).str.lower().eq("true").astype(int)
    return df.dropna(subset=["date", "community_area"])


def download_311_sanitation(client: SocrataClient, cfg: ChicagoDataConfig) -> pd.DataFrame:
    select = "creation_date,community_area,status"
    df = client.get(
        SANITATION_DATASET,
        select=select,
        where=_date_where("creation_date", cfg.start_date, cfg.end_date) + " AND community_area IS NOT NULL",
        order="creation_date ASC",
        page_size=cfg.page_size,
    )
    if df.empty:
        return pd.DataFrame(columns=["creation_date", "community_area", "status"])
    df["creation_date"] = pd.to_datetime(df["creation_date"], errors="coerce")
    df["community_area"] = pd.to_numeric(df["community_area"], errors="coerce").astype("Int64")
    return df.dropna(subset=["creation_date", "community_area"])


def download_abandoned_vehicles(client: SocrataClient, cfg: ChicagoDataConfig) -> pd.DataFrame:
    # Historical dataset field names vary in capitalization; Socrata API names are lower-case.
    select = "creation_date,community_area,status"
    try:
        df = client.get(
            ABANDONED_VEHICLES_DATASET,
            select=select,
            where=_date_where("creation_date", cfg.start_date, cfg.end_date) + " AND community_area IS NOT NULL",
            order="creation_date ASC",
            page_size=cfg.page_size,
        )
    except requests.HTTPError:
        return pd.DataFrame(columns=["creation_date", "community_area", "status"])
    if df.empty:
        return pd.DataFrame(columns=["creation_date", "community_area", "status"])
    df["creation_date"] = pd.to_datetime(df["creation_date"], errors="coerce")
    df["community_area"] = pd.to_numeric(df["community_area"], errors="coerce").astype("Int64")
    return df.dropna(subset=["creation_date", "community_area"])


def download_socioeconomic(client: SocrataClient) -> pd.DataFrame:
    select = (
        "ca,community_area_name,percent_of_housing_crowded,"
        "percent_households_below_poverty,percent_aged_16_unemployed,"
        "percent_aged_25_without_high_school_diploma,"
        "percent_aged_under_18_or_over_64,per_capita_income_,hardship_index"
    )
    df = client.get(SOCIOECONOMIC_DATASET, select=select, page_size=500)
    if df.empty:
        raise RuntimeError("Chicago socioeconomic query returned no rows.")
    df["community_id"] = pd.to_numeric(df["ca"], errors="coerce")
    df = df[df["community_id"].between(1, 77)].copy()
    numeric = [
        "percent_of_housing_crowded",
        "percent_households_below_poverty",
        "percent_aged_16_unemployed",
        "percent_aged_25_without_high_school_diploma",
        "percent_aged_under_18_or_over_64",
        "per_capita_income_",
        "hardship_index",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop(columns=["ca"]).reset_index(drop=True)


def download_community_boundaries(timeout: int = 60) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(COMMUNITY_AREAS_GEOJSON)
    candidates = ["area_num_1", "area_numbe", "community"]
    id_col = next((c for c in candidates if c in gdf.columns), None)
    if id_col is None:
        raise RuntimeError(f"Unable to identify community-area ID in boundary columns: {list(gdf.columns)}")
    gdf["community_id"] = pd.to_numeric(gdf[id_col], errors="coerce").astype("Int64")
    return gdf.dropna(subset=["community_id"]).copy()


def build_spatial_adjacency(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Construct queen-contiguity adjacency with self-connections removed."""
    gdf = gdf.sort_values("community_id").reset_index(drop=True)
    ids = gdf["community_id"].astype(int).to_numpy()
    n = len(gdf)
    A = np.zeros((n, n), dtype=np.float32)
    sindex = gdf.sindex
    for i, geom in enumerate(gdf.geometry):
        for j in sindex.query(geom, predicate="intersects"):
            if i == j:
                continue
            if geom.touches(gdf.geometry.iloc[j]) or geom.intersects(gdf.geometry.iloc[j]):
                A[i, j] = 1.0
    return pd.DataFrame(A, index=ids, columns=ids)


def _aggregate_counts(df: pd.DataFrame, date_col: str, prefix: str, frequency: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "community_id", f"{prefix}_count"])
    x = df.copy()
    x["community_id"] = pd.to_numeric(x["community_area"], errors="coerce").astype("Int64")
    x["date"] = pd.to_datetime(x[date_col]).dt.to_period("D").dt.to_timestamp()
    out = (
        x.set_index("date")
        .groupby("community_id")
        .resample(frequency)
        .size()
        .rename(f"{prefix}_count")
        .reset_index()
    )
    return out


def build_chicago_panel(
    crimes: pd.DataFrame,
    sanitation: pd.DataFrame,
    abandoned: pd.DataFrame,
    socioeconomic: pd.DataFrame,
    frequency: str = "W-MON",
) -> pd.DataFrame:
    """Build a community-time panel without using future target values as features."""
    c = crimes.copy()
    c["community_id"] = pd.to_numeric(c["community_area"], errors="coerce").astype("Int64")
    c["date"] = pd.to_datetime(c["date"]).dt.to_period("D").dt.to_timestamp()

    crime = (
        c.set_index("date")
        .groupby("community_id")
        .resample(frequency)
        .agg(
            observed_event_count=("primary_type", "size"),
            arrest_rate=("arrest", "mean"),
            domestic_fraction=("domestic", "mean"),
            crime_type_diversity=("primary_type", "nunique"),
        )
        .reset_index()
    )

    san = _aggregate_counts(sanitation, "creation_date", "sanitation", frequency)
    abn = _aggregate_counts(abandoned, "creation_date", "abandoned_vehicle", frequency)

    panel = crime.merge(san, on=["date", "community_id"], how="left")
    panel = panel.merge(abn, on=["date", "community_id"], how="left")
    panel["sanitation_count"] = panel["sanitation_count"].fillna(0)
    panel["abandoned_vehicle_count"] = panel["abandoned_vehicle_count"].fillna(0)

    socio_cols = [c for c in socioeconomic.columns if c != "community_area_name"]
    panel = panel.merge(socioeconomic[socio_cols], on="community_id", how="left")
    return panel.sort_values(["date", "community_id"]).reset_index(drop=True)


def acquire_chicago_dataset(
    output_dir: str | Path,
    cfg: ChicagoDataConfig,
    app_token: Optional[str] = None,
) -> dict:
    out = ensure_dir(output_dir)
    client = SocrataClient(app_token=app_token, timeout=cfg.timeout)

    crimes = download_chicago_crimes(client, cfg)
    sanitation = download_311_sanitation(client, cfg)
    abandoned = download_abandoned_vehicles(client, cfg)
    socioeconomic = download_socioeconomic(client)
    boundaries = download_community_boundaries(cfg.timeout)
    adjacency = build_spatial_adjacency(boundaries)
    panel = build_chicago_panel(crimes, sanitation, abandoned, socioeconomic, cfg.frequency)

    panel.to_csv(out / "chicago_panel.csv", index=False)
    adjacency.to_csv(out / "chicago_adjacency.csv")
    socioeconomic.to_csv(out / "chicago_socioeconomic.csv", index=False)
    boundaries[["community_id", "geometry"]].to_file(out / "chicago_community_areas.geojson", driver="GeoJSON")

    return {
        "panel": panel,
        "adjacency": adjacency,
        "socioeconomic": socioeconomic,
        "boundaries": boundaries,
    }
