import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import CLEANED_CSV_PATH, PROCESSED_DATA_DIR
from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints
from src.data.validate_data import validate_dataset


def preprocess_and_save():
    print("Loading raw customer complaints...")
    df = load_complaints()

    print("Validating dataset...")
    validate_dataset(df)

    print("Cleaning complaints data...")
    cleaned_df = clean_complaints(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"Cleaned data successfully saved to: {CLEANED_CSV_PATH} ({len(cleaned_df):,} rows)")


if __name__ == "__main__":
    preprocess_and_save()
