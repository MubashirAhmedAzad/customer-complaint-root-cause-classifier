import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import RAW_CSV_PATH

DATA_PATH = RAW_CSV_PATH

def load_complaints(filepath: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        # Fallback to local relative path if run from alternative working directory
        alt_path = _PROJECT_ROOT / "data" / "raw" / "customer_complaints.csv"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {path} or {alt_path}")

    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_complaints()
    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    print(df.head(2))
