import pandas as pd
from .config import DATA_PATH


def load_data(path: str | None = None) -> pd.DataFrame:
    csv_path = path or DATA_PATH
    df = pd.read_csv(csv_path)
    return df


def parse_time_control(code: str) -> tuple[float, float]:
    try:
        parts = str(code).split("+")
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return 0.0, 0.0


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().copy()
    df["rating_diff"] = df["white_rating"] - df["black_rating"]
    df["white_win"] = (df["winner"] == "white").astype(int)

    time_parsed = df["increment_code"].apply(parse_time_control)
    df["base_time"] = time_parsed.apply(lambda x: x[0])
    df["increment"] = time_parsed.apply(lambda x: x[1])
    df["estimated_minutes"] = df["base_time"] + (df["turns"] * df["increment"] / 2)

    df["eco_family"] = df["opening_eco"].str[0].fillna("?")
    return df
