from pathlib import Path
import numpy as np
import pandas as pd
from src.data_processing import load_goals, parse_date, parse_minute, parse_score, parse_season, prepare_features
from src.dr_analysis import project

DATA = Path(__file__).resolve().parents[1] / "data" / "messi_goals.csv"

def test_difficult_parsing():
    assert parse_minute("90+3") == 93
    assert parse_minute("45 + 2") == 47
    assert np.isnan(parse_minute("late"))
    assert parse_season("11/12") == 2011
    assert parse_season("bad") != parse_season("bad")
    assert parse_score("5:4 AET") == (5.0, 4.0)
    assert all(np.isnan(x) for x in parse_score("unknown"))
    assert str(parse_date("05-01/05").date()) == "2005-01-05"
    assert pd.isna(parse_date("not a date"))

def test_feature_projection_alignment():
    raw = load_goals(DATA)
    clean, matrix, _ = prepare_features(raw)
    assert len(clean) == 704
    assert clean.goal_id.is_unique
    assert matrix.shape[0] == len(clean)
    coords, metrics = project(matrix, clean)
    assert coords.goal_id.tolist() == clean.goal_id.tolist()
    assert coords[["PC1", "PC2", "TSNE1", "TSNE2"]].shape == (len(clean), 4)
    assert metrics["tsne_perplexity"] < len(clean)
    assert 0 <= metrics["tsne_trustworthiness"] <= 1
