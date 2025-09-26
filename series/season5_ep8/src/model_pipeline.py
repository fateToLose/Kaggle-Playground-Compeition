import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from typing import Optional, Any

from .config import random_state, n_job, verbose


def build_pipeline(
    X: pd.DataFrame, Y: pd.Series, preprocessor: ColumnTransformer
) -> tuple[dict[str, Any], dict[str, Any], str]:
    models = {
        "Logistic": LogisticRegression(random_state=random_state, max_iter=1000, n_jobs=n_job, verbose=verbose),
        "Random Forest": RandomForestClassifier(
            random_state=random_state, n_estimators=100, n_jobs=n_job, verbose=verbose
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state, verbose=verbose),
    }

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    results = {}
    trained_pipeline = {}

    for name, model in models.items():
        print(f"Training: {name} - {model=}")
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifer", model),
            ]
        )
        pipeline.fit(x_train, y_train)

        y_pred_prob = pipeline.predict_proba(x_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_prob)
        cv_scores = cross_val_score(pipeline, X, Y, cv=5, scoring="roc_auc", verbose=True, n_jobs=-1)

        results[name] = {
            "auc_score": auc_score,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }
        trained_pipeline[name] = pipeline

        print(f"AUC Score: {auc_score:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    best_model_name = max(results, key=lambda x: results[x]["auc_score"])
    print(f"\n=== BEST MODEL: {best_model_name} ===")
    print(f"Best Validation Accuracy: {results[best_model_name]['auc_score']:.4f}")

    return results, trained_pipeline, best_model_name


def finetune_model(
    X: pd.DataFrame, Y: pd.Series, *, params_grids: dict[str, Any], best_model: Any, preprocessor: ColumnTransformer
):
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", best_model)])
    grid_search = GridSearchCV(pipeline, params_grids, cv=5, scoring="accuracy", n_jobs=-1, verbose=True)
    grid_search.fit(X, Y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def kaggle_test_model(data: pd.DataFrame, model_pipeline: Pipeline, target_cols: str = "y") -> pd.DataFrame:
    if "id" in data.columns:
        data_clean = data.drop("id", axis=1)
    else:
        data_clean = data.loc[:]

    result = model_pipeline.predict(data_clean)

    df_result = pd.DataFrame(result, columns=[target_cols])
    df_final = pd.concat([data_clean["id"], df_result], ignore_index=False, axis=1)

    return df_final
