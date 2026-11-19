"""Cleaning and feature preparation for Messi goal records."""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXPECTED_COLUMNS = ["Season", "Competition", "Matchday", "Date", "Venue", "Club", "Opponent", "Result", "Playing_Position", "Minute", "At_score", "Type", "Goal_assist"]


def load_goals(path: str | Path) -> pd.DataFrame:
    """Read source as strings, clean known spreadsheet corruption, and assign stable source-row IDs."""
    df = pd.read_csv(path, dtype="string", encoding_errors="replace")
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="string"):
        df[col] = df[col].str.strip().replace("", pd.NA)
    # Excel converted season labels 11/12 and 12/13 to dates.
    df["Season"] = df["Season"].replace({"11-Dec": "11/12", "Dec-13": "12/13"})
    df["Competition"] = df["Competition"].replace({"Champions League": "UEFA Champions League"})
    df["Competition"] = df["Competition"].str.replace(r"Troph.{1,3}e des Champions", "Trophée des Champions", regex=True)
    df["Playing_Position"] = df["Playing_Position"].str.strip()
    # Source order is immutable and independent of filters/projections.
    df.insert(0, "goal_id", [f"goal-{i:04d}" for i in range(1, len(df) + 1)])
    return df


def parse_minute(value) -> float:
    """Parse standard and stoppage-time minutes; return NaN for malformed values."""
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:\+\s*(\d+(?:\.\d+)?))?\s*", str(value))
    if not match:
        return np.nan
    return float(match.group(1)) + float(match.group(2) or 0)


def parse_season(value) -> float:
    """Return starting calendar year (2000s for 00-99), tolerating common variants."""
    if pd.isna(value):
        return np.nan
    m = re.search(r"(\d{2,4})\s*[/\-]\s*(\d{2,4})", str(value))
    if not m:
        return np.nan
    year = int(m.group(1))
    return float(2000 + year if year < 100 else year)


def parse_score(value):
    """Parse score-like strings such as 2:1 or 5:4 AET; malformed input returns NaNs."""
    if pd.isna(value):
        return (np.nan, np.nan)
    m = re.fullmatch(r"\s*(\d+)\s*[:\-]\s*(\d+)(?:\s+[A-Za-z.]+)?\s*", str(value))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def parse_date(value):
    """Parse mixed source date formats without raising; malformed dates become NaT.

    Slash dates use the source's US month/day/year convention. Hyphen-slash
    strings such as 05-01/05 are day-month/year spreadsheet artifacts.
    """
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if re.fullmatch(r"\d{1,2}-\d{1,2}/\d{2,4}", text):
        return pd.to_datetime(text, format="%d-%m/%y" if len(text.rsplit("/", 1)[1]) == 2 else "%d-%m/%Y", errors="coerce")
    return pd.to_datetime(text, format="mixed", errors="coerce")


def prepare_features(df: pd.DataFrame):
    """Build one scaled mixed-data matrix. Rare opponent categories are pooled by OHE.

    Assists are omitted because 30% are missing and there are 69 values, making
    identity-heavy sparse geometry misleading. Matchday/result/score are also
    omitted as identifiers or post/outcome-like context. Categorical dimensions
    are one-hot encoded; Euclidean distance then treats each field as a bounded
    mismatch (not an ordinal code). Numeric dimensions are standardized.
    """
    work = df.copy()
    work["minute_num"] = work["Minute"].map(parse_minute)
    work["season_start"] = work["Season"].map(parse_season)
    work["date_parsed"] = work["Date"].map(parse_date)
    for source_col, prefix in [("Result", "result_score"), ("At_score", "score_at_goal")]:
        parsed_scores = work[source_col].map(parse_score)
        work[f"{prefix}_1"] = parsed_scores.map(lambda pair: pair[0])
        work[f"{prefix}_2"] = parsed_scores.map(lambda pair: pair[1])
    categorical = ["Competition", "Venue", "Playing_Position", "Type", "Opponent"]
    numerical = ["minute_num", "season_start"]
    for col in categorical:
        work[col] = work[col].fillna("Unknown").astype(str).str.strip()
    work[numerical] = work[numerical].apply(pd.to_numeric, errors="coerce")
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), numerical),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True), categorical),
    ], sparse_threshold=0.0)
    matrix = prep.fit_transform(work[numerical + categorical])
    return work, matrix, prep
