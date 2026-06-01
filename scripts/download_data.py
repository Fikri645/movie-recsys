"""Download and extract MovieLens 1M dataset."""
import io
import zipfile
import urllib.request
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_RAW, MOVIELENS_URL, MOVIELENS_DIR


def download_movielens() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_RAW / "ml-1m.zip"

    if MOVIELENS_DIR.exists() and (MOVIELENS_DIR / "ratings.dat").exists():
        print("MovieLens 1M already extracted.")
        return

    if not zip_path.exists():
        print(f"Downloading MovieLens 1M from {MOVIELENS_URL} ...")
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        print(f"  Saved: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Zip already present: {zip_path}")

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_RAW)
    print(f"  Extracted to: {MOVIELENS_DIR}")

    ratings = MOVIELENS_DIR / "ratings.dat"
    movies  = MOVIELENS_DIR / "movies.dat"
    users   = MOVIELENS_DIR / "users.dat"
    print(f"\nFiles:")
    for f in [ratings, movies, users]:
        size = f.stat().st_size / 1e6
        print(f"  {f.name}: {size:.1f} MB")


if __name__ == "__main__":
    download_movielens()
