from pathlib import Path


# ----- Directory ----- #
BASE_PATH = Path.cwd()
DATA_PATH = BASE_PATH / "data"
MODEL_PATH = BASE_PATH / "model"
RESULT_PATH = BASE_PATH / "result"

DATA_PATH.mkdir(exist_ok=True)
MODEL_PATH.mkdir(exist_ok=True)
RESULT_PATH.mkdir(exist_ok=True)


# ----- Scikit Setting ----- #
random_state = 303
n_job = 12
verbose = True
