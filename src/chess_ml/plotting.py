import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .config import OUTPUT_FIGURES


def _save_and_show(name: str) -> None:
    os.makedirs(OUTPUT_FIGURES, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES, f"{name}.pdf"))
    plt.show()


def plot_rating_diff(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    x = range(len(df))
    plt.errorbar(
        x, df["White Win Probability"],
        yerr=[
            df["White Win Probability"] - df["CI Lower"],
            df["CI Upper"] - df["White Win Probability"],
        ],
        marker="o", capsize=5, capthick=2,
    )
    plt.xticks(x, df["Rating Bin"])
    plt.xlabel("Rating Difference")
    plt.ylabel("Win Probability")
    plt.title("Rating Difference vs White Win Probability (95% CI)")
    _save_and_show("RQ1_figure")


def plot_openings(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    colors = ["#2ecc71" if s else "#e74c3c" for s in df["Significant"]]
    plt.bar(df["Opening"], df["White Win Rate"], color=colors)
    plt.xticks(rotation=60, ha="right")
    plt.xlabel("Opening Strategy")
    plt.ylabel("White Win Rate")
    plt.title("Top Openings by White Win Rate (green = significant)")
    _save_and_show("RQ2_figure")


def plot_duration(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(df["Victory Type"], df["Average Turns"], yerr=df["SE"], capsize=5)
    plt.xlabel("Victory Type")
    plt.ylabel("Average Turns")
    plt.title("Game Duration by Victory Type (±1 SE)")
    _save_and_show("RQ3_figure")


def plot_rated_comparison(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(df["Rated"].astype(str), df["Average Turns"])
    plt.xlabel("Game Type (Rated)")
    plt.ylabel("Average Turns")
    plt.title("Rated vs Non-Rated Game Length")
    _save_and_show("RQ4_figure")


def plot_win_distribution(df: pd.DataFrame) -> None:
    df.plot(kind="bar", figsize=(8, 5))
    plt.xlabel("Rated")
    plt.ylabel("Proportion")
    plt.title("Win Distribution in Rated vs Non-Rated Games")
    plt.legend(title="Winner", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_and_show("RQ4_win_dist")


def plot_time_control(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(df["Time Control"], df["Average Turns"])
    plt.xticks(rotation=45)
    plt.xlabel("Time Control")
    plt.ylabel("Average Turns")
    plt.title("Time Control vs Game Duration")
    _save_and_show("RQ5_duration")


def plot_opening_ply(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    x = df["Opening Ply"]
    plt.plot(x, df["White Win Rate"], marker="o", label="Win Rate")
    plt.fill_between(
        x, df["CI Lower"], df["CI Upper"], alpha=0.2, label="95% CI",
    )
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    plt.xlabel("Opening Ply (Depth)")
    plt.ylabel("White Win Rate")
    plt.title("Opening Depth vs White Win Rate")
    plt.legend()
    _save_and_show("RQ6_figure")


def plot_model_comparison(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    x = range(len(df))
    plt.bar(x, df["Test Accuracy"])
    plt.errorbar(
        x, df["CV Mean Accuracy"],
        yerr=df["CV Std"], fmt="none", color="black", capsize=5, label="CV ±1 std",
    )
    plt.xticks(x, df["Model"], rotation=15)
    plt.ylabel("Accuracy")
    plt.title("Model Comparison")
    plt.ylim(0, 1)
    plt.legend()
    for i, row in df.iterrows():
        plt.text(
            i, row["Test Accuracy"] + 0.01,
            f"{row['Test Accuracy']:.3f}", ha="center", va="bottom",
        )
    _save_and_show("RQ7_figure")


def plot_confusion_matrices(detailed: dict, class_names: list[str]) -> None:
    models = {k: v for k, v in detailed.items() if v["confusion_matrix"] is not None}
    n = len(models)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, info) in zip(axes, models.items()):
        cm = info["confusion_matrix"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.suptitle("Confusion Matrices", fontsize=14)
    _save_and_show("RQ7_confusion_matrices")


def plot_feature_importances(detailed: dict) -> None:
    models = {k: v for k, v in detailed.items() if v["feature_importances"] is not None}
    n = len(models)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, info) in zip(axes, models.items()):
        fi = info["feature_importances"]
        features = list(fi.keys())
        values = list(fi.values())
        ax.barh(features, values)
        ax.set_title(f"{name}\nFeature Importance")
        ax.set_xlabel("Importance")
    _save_and_show("RQ7_feature_importance")


def plot_eco_analysis(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    labels = df["ECO Code"] + " — " + df["ECO Family"]
    plt.barh(labels, df["White Win Rate"])
    plt.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("White Win Rate")
    plt.title("White Win Rate by ECO Opening Family")
    _save_and_show("RQ2_eco")
