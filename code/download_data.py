"""Download training and validation data for Assignment 3.

Run once:
    python download_data.py

This downloads train.npz and val.npz into the data/ directory.
"""

import os
import urllib.request

DATA_DIR = "data"
BASE_URL = "https://github.com/bu-ds595/assignment03-starter/releases/download/v1.0"
FILES = ["train.npz", "val.npz"]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname in FILES:
        dest = os.path.join(DATA_DIR, fname)
        if os.path.exists(dest):
            print(f"  {dest} already exists, skipping")
            continue
        url = f"{BASE_URL}/{fname}"
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  Saved {dest} ({size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
