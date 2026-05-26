import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_CSV_PATH = "/kaggle/input/datasets/jordanjesudas/chess-game-dataset/games.csv"
LOCAL_CSV_PATH = os.path.join(BASE_DIR, "Dataset", "games.csv")

DATA_PATH = DEFAULT_CSV_PATH if os.path.exists(DEFAULT_CSV_PATH) else LOCAL_CSV_PATH

OUTPUT_FIGURES = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUT_TABLES = os.path.join(BASE_DIR, "outputs", "tables")

RATING_BINS = [-float("inf"), -200, 0, 200, float("inf")]
RATING_LABELS = ["<-200", "-200 to 0", "0 to 200", ">200"]

MIN_GAMES_THRESHOLD = 30

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
