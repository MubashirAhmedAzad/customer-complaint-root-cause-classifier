import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

REQUIRED_COLUMNS = [
    "complaint_id",
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "complaint_narrative",
    "company",
    "channel",
    "state",
    "sentiment_label",
    "resolution_status",
]

def validate_dataset(df: pd.DataFrame) -> bool:
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df["complaint_id"].duplicated().any():
        print("Warning: Duplicate complaint IDs found")

    print("Dataset validation completed successfully.")
    return True


if __name__ == "__main__":
    from src.data.load_data import load_complaints
    df = load_complaints()
    validate_dataset(df)
