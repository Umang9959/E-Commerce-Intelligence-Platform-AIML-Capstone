from __future__ import annotations

import urllib.request
import zipfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ecommerce_intelligence import config
from src.ecommerce_intelligence.data_processing import ensure_directories


DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def main() -> None:
    ensure_directories()
    zip_path = config.RAW_DIR / "online_retail_II.zip"
    print(f"Downloading {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, zip_path)

    print("Extracting workbook")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(config.RAW_DIR)

    if config.RAW_WORKBOOK.exists():
        print(f"Dataset ready: {config.RAW_WORKBOOK}")
    else:
        print("Download finished, but online_retail_II.xlsx was not found. Check data/raw/.")


if __name__ == "__main__":
    main()
