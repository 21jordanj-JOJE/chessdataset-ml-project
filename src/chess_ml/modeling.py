import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from .config import RANDOM_STATE, TEST_SIZE, CV_FOLDS

FEATURE_NAMES = ["white_rating", "black_rating", "rating_diff", "opening_ply"]


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    X = df[FEATURE_NAMES].copy()
    y = df["winner"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X.values, y_encoded, le


def train_and_evaluate(
    X: np.ndarray, y: np.ndarray, le: LabelEncoder
) -> tuple[pd.DataFrame, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, random_state=RANDOM_STATE
            )),
        ]),
    }

    rows = []
    detailed = {}
    for name, pipe in models.items():
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        rows.append({
            "Model": name,
            "CV Mean Accuracy": round(cv_scores.mean(), 4),
            "CV Std": round(cv_scores.std(), 4),
            "Test Accuracy": round(test_acc, 4),
        })

        report = classification_report(
            y_test, y_pred, target_names=le.classes_, output_dict=True
        )
        cm = confusion_matrix(y_test, y_pred)

        importances = None
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = dict(zip(FEATURE_NAMES, clf.feature_importances_.round(4)))

        detailed[name] = {
            "classification_report": pd.DataFrame(report).T.round(4),
            "confusion_matrix": cm,
            "feature_importances": importances,
        }

    return pd.DataFrame(rows), detailed


def hyperparameter_tuning(
    X: np.ndarray, y: np.ndarray
) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    pipelines = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }

    param_grids = {
        "Random Forest": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [5, 10, 20, None],
            "clf__min_samples_split": [2, 5, 10],
        },
        "Gradient Boosting": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1],
        },
    }

    rows = []
    for name in pipelines:
        grid = GridSearchCV(
            pipelines[name],
            param_grids[name],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        test_acc = grid.score(X_test, y_test)
        rows.append({
            "Model": name,
            "Best CV Accuracy": round(grid.best_score_, 4),
            "Test Accuracy": round(test_acc, 4),
            "Best Params": str(grid.best_params_),
        })

    return pd.DataFrame(rows)
