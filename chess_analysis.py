import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC)

import pandas as pd
from chess_ml.config import OUTPUT_FIGURES, OUTPUT_TABLES, MIN_GAMES_THRESHOLD
from chess_ml.data import load_data, preprocess
from chess_ml.analysis import (
    rq1_rating_diff_win_prob,
    rq2_top_openings,
    rq2_eco_analysis,
    rq3_duration_by_victory,
    rq4_rated_comparison,
    rq5_time_control,
    rq6_opening_ply,
)
from chess_ml.modeling import prepare_features, train_and_evaluate, hyperparameter_tuning
from chess_ml.plotting import (
    plot_rating_diff,
    plot_openings,
    plot_duration,
    plot_rated_comparison,
    plot_win_distribution,
    plot_time_control,
    plot_opening_ply,
    plot_model_comparison,
    plot_confusion_matrices,
    plot_feature_importances,
    plot_eco_analysis,
)


def main():
    os.makedirs(OUTPUT_FIGURES, exist_ok=True)
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    df_raw = load_data()
    print(f"Loaded {len(df_raw):,} rows")

    df = preprocess(df_raw)
    print(f"After cleaning: {len(df):,} rows")

    # RQ1 — Rating Difference vs Win Probability
    rq1 = rq1_rating_diff_win_prob(df)
    rq1.to_csv(os.path.join(OUTPUT_TABLES, "RQ1_table.csv"), index=False)
    print("\nRQ1 — Rating Difference vs Win Rate:")
    print(rq1.to_string(index=False))
    plot_rating_diff(rq1)

    # RQ2 — Top Openings
    rq2 = rq2_top_openings(df)
    rq2.to_csv(os.path.join(OUTPUT_TABLES, "RQ2_openings.csv"), index=False)
    print("\nRQ2 — Top Openings (min {} games):".format(MIN_GAMES_THRESHOLD))
    print(rq2.to_string(index=False))
    plot_openings(rq2)

    # RQ2b — ECO Family Analysis
    rq2b = rq2_eco_analysis(df)
    rq2b.to_csv(os.path.join(OUTPUT_TABLES, "RQ2_eco.csv"), index=False)
    print("\nR2b — ECO Family Win Rates:")
    print(rq2b.to_string(index=False))
    plot_eco_analysis(rq2b)

    # RQ3 — Game Duration by Victory Type
    rq3 = rq3_duration_by_victory(df)
    rq3.to_csv(os.path.join(OUTPUT_TABLES, "RQ3_duration.csv"), index=False)
    print("\nRQ3 — Game Duration by Victory Type:")
    print(rq3.to_string(index=False))
    plot_duration(rq3)

    # RQ4 — Rated vs Non-Rated
    rq4_turns, rq4_dist, rq4_chi = rq4_rated_comparison(df)
    rq4_turns.to_csv(os.path.join(OUTPUT_TABLES, "RQ4_rated.csv"), index=False)
    rq4_dist.to_csv(os.path.join(OUTPUT_TABLES, "RQ4_win_distribution.csv"))
    rq4_chi.to_csv(os.path.join(OUTPUT_TABLES, "RQ4_chi_square.csv"), index=False)
    print("\nRQ4 — Rated vs Non-Rated Game Length:")
    print(rq4_turns.to_string(index=False))
    print("\nWin Distribution:")
    print(rq4_dist)
    print("\nChi-square test:")
    print(rq4_chi.to_string(index=False))
    plot_rated_comparison(rq4_turns)
    plot_win_distribution(rq4_dist)

    # RQ5 — Time Control vs Game Duration
    rq5 = rq5_time_control(df)
    rq5.to_csv(os.path.join(OUTPUT_TABLES, "RQ5_time_control.csv"), index=False)
    print("\nRQ5 — Time Control vs Game Duration (min {} games):".format(MIN_GAMES_THRESHOLD))
    print(rq5.to_string(index=False))
    plot_time_control(rq5)

    # RQ6 — Opening Depth vs Win Rate
    rq6 = rq6_opening_ply(df)
    rq6.to_csv(os.path.join(OUTPUT_TABLES, "RQ6_opening_ply.csv"), index=False)
    print("\nRQ6 — Opening Depth vs Win Rate (min {} games):".format(MIN_GAMES_THRESHOLD))
    print(rq6.to_string(index=False))
    plot_opening_ply(rq6)

    # RQ7 — Model Comparison
    X, y, le = prepare_features(df)
    rq7, detailed = train_and_evaluate(X, y, le)
    rq7.to_csv(os.path.join(OUTPUT_TABLES, "RQ7_model.csv"), index=False)
    print("\nRQ7 — Model Comparison:")
    print(rq7.to_string(index=False))

    for name, info in detailed.items():
        info["classification_report"].to_csv(
            os.path.join(OUTPUT_TABLES, f"RQ7_report_{name.replace(' ', '_')}.csv")
        )
        print(f"\n{name} Classification Report:")
        print(info["classification_report"])

    plot_model_comparison(rq7)
    plot_confusion_matrices(detailed, list(le.classes_))
    plot_feature_importances(detailed)

    print("\nDone. Outputs saved to:", OUTPUT_TABLES, "and", OUTPUT_FIGURES)


if __name__ == "__main__":
    main()
