# Changelog — Chess Dataset ML Project

All notable changes made to this project, organized by contribution area.

---

## Table of Contents

1. [Project Structure Refactoring](#1-project-structure-refactoring)
2. [Data & Configuration](#2-data--configuration)
3. [Statistical Significance Testing](#3-statistical-significance-testing)
4. [Sample Thresholds & Edge Case Fixes](#4-sample-thresholds--edge-case-fixes)
5. [Model Interpretability](#5-model-interpretability)
6. [Hyperparameter Tuning](#6-hyperparameter-tuning)
7. [ECO Opening Family Analysis](#7-eco-opening-family-analysis)
8. [Feature Engineering](#8-feature-engineering)
9. [Visualization Improvements](#9-visualization-improvements)
10. [Runner Script](#10-runner-script)
11. [Jupyter Notebook](#11-jupyter-notebook)
12. [Repository Hygiene](#12-repository-hygiene)
13. [Dependencies](#13-dependencies)

---

## 1. Project Structure Refactoring

**Before:** A single ~80-cell monolithic Jupyter notebook (`Jordan Chess data sets.ipynb`) with all code, analysis, and visualization in one file. No reusable functions, no modular structure, repeated import blocks.

**After:** A modular Python package under `src/chess_ml/` with clean separation of concerns:

| Module | File | Responsibility |
|---|---|---|
| Config | `src/chess_ml/config.py` | Centralized paths, constants, thresholds, ML hyperparameters |
| Data | `src/chess_ml/data.py` | Data loading and preprocessing |
| Analysis | `src/chess_ml/analysis.py` | All analytical functions (RQ1–RQ6 + ECO) |
| Modeling | `src/chess_ml/modeling.py` | ML pipeline (feature prep, training, tuning, evaluation) |
| Plotting | `src/chess_ml/plotting.py` | All visualization functions |
| Init | `src/chess_ml/__init__.py` | Clean public API with re-exports |

Each module exports pure functions that accept and return DataFrames/arrays — no hidden state, no side effects beyond optional plot saving. The original notebook is preserved untouched.

---

## 2. Data & Configuration

### `src/chess_ml/config.py` — New file

All magic numbers and paths are now centralized:

```python
# Auto-detects Kaggle vs local environment
DEFAULT_CSV_PATH = "/kaggle/input/datasets/jordanjesudas/chess-game-dataset/games.csv"
LOCAL_CSV_PATH   = os.path.join(BASE_DIR, "Dataset", "games.csv")
DATA_PATH        = DEFAULT_CSV_PATH if os.path.exists(DEFAULT_CSV_PATH) else LOCAL_CSV_PATH

# Rating bins now cover ALL games (no data loss at edges)
RATING_BINS   = [-float("inf"), -200, 0, 200, float("inf")]
RATING_LABELS = ["<-200", "-200 to 0", "0 to 200", ">200"]

# Minimum sample size to avoid small-sample bias
MIN_GAMES_THRESHOLD = 30

# ML hyperparameters
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
```

### `src/chess_ml/data.py` — New file

**`load_data(path=None)`** — Loads CSV from the default Kaggle path, a local fallback, or a user-provided path override. No hardcoded assumptions.

**`preprocess(df)`** — Centralized cleaning pipeline:
- Drops rows with any `NaN` values
- Creates `rating_diff = white_rating - black_rating`
- Creates `white_win = 1 if winner == "white" else 0`
- Parses `increment_code` (e.g., `"15+2"`) into `base_time` and `increment` numeric columns (see [Feature Engineering](#8-feature-engineering))
- Extracts `eco_family` from `opening_eco` first letter (see [ECO Opening Family Analysis](#7-eco-opening-family-analysis))

---

## 3. Statistical Significance Testing

Every research question now includes formal statistical tests. Previously, point estimates were reported with no measure of uncertainty.

### RQ1 — Rating Difference → Win Probability
- **Added:** 95% Clopper-Pearson exact confidence intervals per rating bin
- **Added:** `Game Count` column showing sample size per bin
- **Method:** `scipy.stats.binomtest(k, n).proportion_ci()` — the exact method, not normal approximation
- **Output columns:** `Rating Bin`, `White Win Probability`, `CI Lower`, `CI Upper`, `Game Count`
- **Plot:** Error bars showing 95% CI on the line chart

### RQ2 — Opening Strategy → White Win Rate
- **Added:** Binomial test per opening against the overall mean White win rate
- **Added:** `p-value` column (null hypothesis: opening win rate = overall rate, one-sided "greater" alternative)
- **Added:** `Significant` boolean column (True if p < 0.05)
- **Method:** `scipy.stats.binomtest(k, n, p_overall, alternative="greater")`
- **Plot:** Color-coded bars — green for significant, red for not significant

### RQ3 — Game Duration by Victory Type
- **Added:** Standard deviation (`Std Turns`) and standard error (`SE`) per victory type
- **Method:** `SE = std / sqrt(n)`
- **Output columns:** `Victory Type`, `Average Turns`, `Std Turns`, `Game Count`, `SE`
- **Plot:** Error bars showing ±1 SE on the bar chart

### RQ4 — Rated vs Non-Rated Games
- **Added:** Welch's t-test for turn-count comparison (`equal_var=False` — does not assume equal variances)
- **Added:** Mann-Whitney U test (non-parametric alternative, no normality assumption)
- **Added:** Chi-square test of independence for win distribution (rated × winner contingency table)
- **Output:** Three DataFrames — turns comparison (with p-values), win distribution, chi-square result
- **Method:** `scipy.stats.ttest_ind`, `scipy.stats.mannwhitneyu`, `scipy.stats.chi2_contingency`

### RQ6 — Opening Depth → White Win Rate
- **Added:** 95% Clopper-Pearson confidence intervals per ply depth
- **Added:** `Game Count` column
- **Plot:** Shaded 95% CI band around the win rate line, plus a 50% baseline reference line

---

## 4. Sample Thresholds & Edge Case Fixes

### Minimum Sample Threshold (`MIN_GAMES_THRESHOLD = 30`)

**Problem:** The original notebook's top-10 openings (RQ2) and top-10 time controls (RQ5) were dominated by groups with as few as 1 game, producing misleading 100% or 0% win rates.

**Fix:** All group-based analyses (RQ2, RQ5, RQ6, RQ2b) now filter out groups with fewer than 30 games before ranking. This is configurable via `config.MIN_GAMES_THRESHOLD`.

### Rating Bin Edge Coverage

**Problem:** Original bins `[-500, -200, 0, 200, 500]` silently excluded games where `|rating_diff| > 500`.

**Fix:** Bins changed to `[-inf, -200, 0, 200, inf]` — every game is now included.

---

## 5. Model Interpretability

**Before (RQ7):** Only a single accuracy number per model. No insight into what the models learned or where they failed.

**After:** Three layers of interpretability, all saved as outputs:

### Classification Reports
- Per-model precision, recall, F1-score, and support for each class (`white`, `black`, `draw`)
- Saved as CSV: `outputs/tables/RQ7_report_Logistic_Regression.csv`, `RQ7_report_Random_Forest.csv`, `RQ7_report_Gradient_Boosting.csv`
- **Method:** `sklearn.metrics.classification_report(output_dict=True)`

### Confusion Matrices
- Side-by-side heatmaps for all three models
- Annotated with raw counts; color intensity shows magnitude
- Saved as PDF: `outputs/figures/RQ7_confusion_matrices.pdf`
- **Method:** `sklearn.metrics.confusion_matrix`

### Feature Importances
- Extracted from tree-based models (Random Forest, Gradient Boosting) via `feature_importances_`
- Horizontal bar charts showing relative importance of each feature
- Saved as PDF: `outputs/figures/RQ7_feature_importance.pdf`
- Logistic Regression excluded (coefficients are not directly comparable to tree importance)

---

## 6. Hyperparameter Tuning

**New function:** `hyperparameter_tuning(X, y)` in `src/chess_ml/modeling.py`

Uses `GridSearchCV` with `StratifiedKFold(n_splits=5)` to search over:

**Random Forest:**
| Parameter | Values |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 5, 10, 20, None |
| `min_samples_split` | 2, 5, 10 |

**Gradient Boosting:**
| Parameter | Values |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 3, 5, 7 |
| `learning_rate` | 0.01, 0.05, 0.1 |

Returns a DataFrame with `Best CV Accuracy`, `Test Accuracy`, and `Best Params` per model. All pipelines include `StandardScaler` and use `n_jobs=-1` for parallel search.

---

## 7. ECO Opening Family Analysis

**New function:** `rq2_eco_analysis(df)` in `src/chess_ml/analysis.py`

Groups openings by ECO code first letter into 5 families:

| Code | Family |
|---|---|
| A | Flank Openings |
| B | Semi-Open Games |
| C | Open Games |
| D | Closed Games |
| E | Indian Defenses |

Filters families with >=30 games, ranks by White win rate. Output saved as `outputs/tables/RQ2_eco.csv`.

**New plot:** `plot_eco_analysis()` — horizontal bar chart with 50% baseline reference line. Saved as `outputs/figures/RQ2_eco.pdf`.

---

## 8. Feature Engineering

### Time Control Parsing

**New function:** `parse_time_control(code)` in `src/chess_ml/data.py`

Parses `increment_code` strings (e.g., `"15+2"`, `"90+30"`) into numeric columns:
- `base_time` — the initial clock time in minutes (e.g., `15.0`)
- `increment` — the per-move increment in seconds (e.g., `2.0`)
- `estimated_minutes` — estimated total game time: `base_time + (turns * increment / 2)`

### ECO Family Extraction

Extracts the first character of `opening_eco` into a new `eco_family` column during preprocessing.

### ML Feature Set Expansion

**Before:** 3 features — `white_rating`, `black_rating`, `opening_ply`

**After:** 4 features — adds `rating_diff` (white_rating - black_rating), which is the single most predictive feature for game outcome.

---

## 9. Visualization Improvements

All plots now use `plt.tight_layout()` before saving to prevent label clipping. Specific improvements per RQ:

| RQ | Before | After |
|---|---|---|
| RQ1 | Plain line plot | Line plot with 95% CI error bars |
| RQ2 | Uniform bar colors | Green/red color coding by significance |
| RQ3 | Plain bar chart | Bar chart with ±1 SE error bars |
| RQ6 | Plain line plot | Line plot with shaded 95% CI band + 50% baseline |
| RQ7 | Simple accuracy bar chart | Bar chart with CV std error bars + value labels |
| RQ7 | No confusion matrices | Side-by-side annotated heatmaps |
| RQ7 | No feature importance | Horizontal bar charts per tree model |
| RQ2b | Did not exist | New horizontal bar chart by ECO family |

All plots saved as PDF to `outputs/figures/`.

---

## 10. Runner Script

**New file:** `chess_analysis.py` — End-to-end runner that executes all 7 research questions sequentially.

- Auto-creates output directories
- Prints formatted results to console
- Saves all tables as CSV to `outputs/tables/`
- Saves all figures as PDF to `outputs/figures/`
- Saves per-model classification reports as CSV
- Run with: `python chess_analysis.py`

---

## 11. Jupyter Notebook

**New file:** `notebooks/chess_analysis.ipynb` — Interactive notebook version.

- Imports from the `chess_ml` package (no inline analysis code)
- Markdown cells explain each research question and methodology
- Uses `display()` for rich table rendering
- Covers all RQ1–RQ7 plus confusion matrices and feature importances
- Compatible with both local Jupyter and Kaggle environments

---

## 12. Repository Hygiene

### `.gitignore` — New file

Excludes from version control:
- `outputs/`, `Output Figures/`, `Output tables/` — generated artifacts
- `*.pdf`, `*.csv` — output file types
- `Dataset/` — large data file
- `__pycache__/`, `*.pyc`, `*.egg-info/`, `dist/`, `build/` — Python artifacts
- `.ipynb_checkpoints/` — Jupyter checkpoints
- `.vscode/`, `.idea/` — IDE settings
- `.DS_Store`, `Thumbs.db` — OS files

### `requirements.txt` — New file

Pinned minimum dependencies:

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
jupyter>=1.0.0
```

### `README.md` — Rewritten

- Updated project structure diagram
- Three setup options (local CLI, Jupyter, Kaggle)
- Mermaid workflow diagram
- Detailed description of each RQ including statistical methods
- Complete model comparison table
- Summary of all improvements over the original notebook

---

## 13. Dependencies

### Added

| Package | Version | Purpose |
|---|---|---|
| `scipy` | >=1.10.0 | Statistical tests (binomtest, ttest_ind, mannwhitneyu, chi2_contingency) |

### Existing (now pinned)

| Package | Version | Purpose |
|---|---|---|
| `pandas` | >=2.0.0 | Data manipulation |
| `numpy` | >=1.24.0 | Numerical operations |
| `matplotlib` | >=3.7.0 | Visualization |
| `scikit-learn` | >=1.3.0 | ML models, pipelines, metrics, CV |
| `jupyter` | >=1.0.0 | Notebook environment |

---

## Summary of New Output Files

### Tables (`outputs/tables/`)

| File | Description |
|---|---|
| `RQ1_table.csv` | Rating bins with win rates, CIs, and game counts |
| `RQ2_openings.csv` | Top 10 openings with win rates, p-values, significance |
| `RQ2_eco.csv` | ECO family win rates |
| `RQ3_duration.csv` | Duration by victory type with std and SE |
| `RQ4_rated.csv` | Rated vs unrated turns with t-test and U-test p-values |
| `RQ4_win_distribution.csv` | Win distribution proportions |
| `RQ4_chi_square.csv` | Chi-square test result |
| `RQ5_time_control.csv` | Top 10 time controls by game length |
| `RQ6_opening_ply.csv` | Opening depth with win rates, CIs, and game counts |
| `RQ7_model.csv` | Model comparison with CV mean/std and test accuracy |
| `RQ7_report_Logistic_Regression.csv` | Per-class precision, recall, F1 |
| `RQ7_report_Random_Forest.csv` | Per-class precision, recall, F1 |
| `RQ7_report_Gradient_Boosting.csv` | Per-class precision, recall, F1 |

### Figures (`outputs/figures/`)

| File | Description |
|---|---|
| `RQ1_figure.pdf` | Rating difference vs win rate with 95% CI error bars |
| `RQ2_figure.pdf` | Top openings with significance color coding |
| `RQ2_eco.pdf` | ECO family win rates (new) |
| `RQ3_figure.pdf` | Duration by victory type with SE error bars |
| `RQ4_figure.pdf` | Rated vs unrated game length |
| `RQ4_win_dist.pdf` | Win distribution comparison |
| `RQ5_duration.pdf` | Time control vs game duration |
| `RQ6_figure.pdf` | Opening depth with 95% CI band |
| `RQ7_figure.pdf` | Model comparison with CV error bars |
| `RQ7_confusion_matrices.pdf` | Side-by-side confusion matrices (new) |
| `RQ7_feature_importance.pdf` | Feature importance per tree model (new) |
