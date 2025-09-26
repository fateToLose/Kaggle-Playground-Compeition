import sys
from pathlib import Path

from src.preprocess import prepare_data, feature_engineering
from src.model_pipeline import build_pipeline, finetune_model, kaggle_test_model
from src.config import DATA_PATH, MODEL_PATH

sys.path.append(str(Path.cwd().parent.parent))
from common.helper import read_kaggle_data, save_kaggle_result


# ----- Global Variable ----- #
param_grids = {
    "Logistic": {
        "classifier__C": [0.1, 1, 10],
        "classifier__penalty": ["l1", "l2"],
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 5],
    },
}


# ----- Functions ----- #
def train_model():
    train = read_kaggle_data(DATA_PATH / "train.csv")
    X, Y, num_cols, cat_cols = prepare_data(train)
    preprocess = feature_engineering(num_cols, cat_cols)
    result, pipeline_result, best_model_name = build_pipeline(X, Y, preprocess)

    best_param_girid = param_grids[best_model_name]
    best_model = result[best_model_name]

    best_model = finetune_model(X, Y, params_grids=best_param_girid, best_model=best_model, preprocessor=preprocess)

    return best_model


def test_model(model):
    data = read_kaggle_data(DATA_PATH / "test.csv")
    result = kaggle_test_model(data, model)
    save_kaggle_result(result, "test_result.csv", MODEL_PATH)


# ----- Main ---- #
if __name__ == "__main__":
    test_model(train_model())
