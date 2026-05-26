import numpy as np
import pandas as pd
from scipy import stats
from .config import RATING_BINS, RATING_LABELS, MIN_GAMES_THRESHOLD


def _proportion_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=1 - alpha)
    return (ci.low, ci.high)


def rq1_rating_diff_win_prob(df: pd.DataFrame) -> pd.DataFrame:
    df["rating_bin"] = pd.cut(df["rating_diff"], bins=RATING_BINS, labels=RATING_LABELS)
    rows = []
    for _, grp in df.groupby("rating_bin", observed=False):
        n = len(grp)
        k = grp["white_win"].sum()
        rate = grp["white_win"].mean() if n > 0 else 0.0
        lo, hi = _proportion_ci(int(k), n)
        rows.append({
            "Rating Bin": _.cat if hasattr(_, "cat") else _,
            "White Win Probability": round(rate, 4),
            "CI Lower": round(lo, 4),
            "CI Upper": round(hi, 4),
            "Game Count": n,
        })
    return pd.DataFrame(rows)


def rq2_top_openings(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    overall_rate = df["white_win"].mean()
    opening_stats = df.groupby("opening_name").agg(
        white_win_rate=("white_win", "mean"),
        game_count=("white_win", "count"),
    )
    opening_stats = opening_stats[
        opening_stats["game_count"] >= MIN_GAMES_THRESHOLD
    ].copy()

    pvals = []
    for _, row in opening_stats.iterrows():
        k = int(row["white_win_rate"] * row["game_count"])
        n = int(row["game_count"])
        stat, pval = stats.binomtest(k, n, overall_rate, alternative="greater").statistic, None
        _, pval = stats.binomtest(k, n, overall_rate, alternative="greater"), None
        result = stats.binomtest(k, n, overall_rate, alternative="greater")
        pvals.append(result.pvalue)
    opening_stats["p_value"] = pvals
    opening_stats["significant"] = opening_stats["p_value"] < 0.05

    top = opening_stats.sort_values("white_win_rate", ascending=False).head(top_n)
    top = top.reset_index()
    top.columns = ["Opening", "White Win Rate", "Game Count", "p-value", "Significant"]
    return top


def rq3_duration_by_victory(df: pd.DataFrame) -> pd.DataFrame:
    result = df.groupby("victory_status")["turns"].agg(["mean", "std", "count"]).reset_index()
    result.columns = ["Victory Type", "Average Turns", "Std Turns", "Game Count"]
    result["SE"] = result["Std Turns"] / np.sqrt(result["Game Count"])
    return result.sort_values("Average Turns", ascending=False)


def rq4_rated_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rated = df[df["rated"] == True]["turns"]
    unrated = df["rated"] == False
    unrated = df[unrated]["turns"]

    t_stat, t_pval = stats.ttest_ind(rated, unrated, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(rated, unrated, alternative="two-sided")

    turns = df.groupby("rated")["turns"].agg(["mean", "std", "count"]).reset_index()
    turns.columns = ["Rated", "Average Turns", "Std Turns", "Game Count"]
    turns["Test Used"] = "Welch's t-test + Mann-Whitney U"
    turns["t-test p-value"] = round(t_pval, 6)
    turns["U-test p-value"] = round(u_pval, 6)

    win_dist = df.groupby("rated")["winner"].value_counts(normalize=True).unstack()
    win_dist.index.name = "Rated"

    chi2, chi_p, dof, expected = stats.chi2_contingency(
        df.groupby("rated")["winner"].value_counts().unstack(fill_value=0)
    )
    chi_result = pd.DataFrame(
        {"Test": ["Chi-square"], "Statistic": [round(chi2, 4)], "p-value": [round(chi_p, 6)], "DoF": [dof]}
    )

    return turns, win_dist, chi_result


def rq5_time_control(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    stats_df = df.groupby("increment_code").agg(
        avg_turns=("turns", "mean"),
        white_win_rate=("white_win", "mean"),
        game_count=("white_win", "count"),
    )
    stats_df = stats_df[stats_df["game_count"] >= MIN_GAMES_THRESHOLD]
    top = stats_df.sort_values("avg_turns", ascending=False).head(top_n)
    top = top.reset_index()
    top.columns = ["Time Control", "Average Turns", "White Win Rate", "Game Count"]
    return top


def rq6_opening_ply(df: pd.DataFrame) -> pd.DataFrame:
    ply_stats = (
        df.groupby("opening_ply")
        .agg(
            white_win_rate=("white_win", "mean"),
            game_count=("white_win", "count"),
        )
        .reset_index()
    )
    ply_stats = ply_stats[ply_stats["game_count"] >= MIN_GAMES_THRESHOLD].copy()

    rows = []
    for _, row in ply_stats.iterrows():
        lo, hi = _proportion_ci(int(row["white_win_rate"] * row["game_count"]), int(row["game_count"]))
        rows.append({
            "Opening Ply": int(row["opening_ply"]),
            "White Win Rate": round(row["white_win_rate"], 4),
            "CI Lower": round(lo, 4),
            "CI Upper": round(hi, 4),
            "Game Count": int(row["game_count"]),
        })
    return pd.DataFrame(rows)


def rq2_eco_analysis(df: pd.DataFrame) -> pd.DataFrame:
    eco_names = {
        "A": "Flank Openings",
        "B": "Semi-Open Games",
        "C": "Open Games",
        "D": "Closed Games",
        "E": "Indian Defenses",
    }
    eco_stats = df.groupby("eco_family").agg(
        white_win_rate=("white_win", "mean"),
        game_count=("white_win", "count"),
    )
    eco_stats = eco_stats[eco_stats["game_count"] >= MIN_GAMES_THRESHOLD]
    eco_stats["eco_name"] = eco_stats.index.map(lambda x: eco_names.get(x, "Unknown"))
    eco_stats = eco_stats.sort_values("white_win_rate", ascending=False).reset_index()
    eco_stats.columns = ["ECO Code", "White Win Rate", "Game Count", "ECO Family"]
    return eco_stats
