"""
download_dataset.py
--------------------
Downloads the jm1 dataset from the OpenML repository (a publicly accessible
mirror of the NASA PROMISE data) and saves it as jm1.csv in the current folder.

Run this once from the repo root before executing the pipeline:
    python scripts/download_dataset.py
"""

import urllib.request
import os
import sys

# ── Try OpenML's JSON API — no extra libraries required ──────────────────────
# OpenML dataset ID for jm1 is 1053.
# The API returns the raw ARFF content which we parse manually.

OPENML_URL = "https://www.openml.org/data/download/53936/jm1.arff"

def download_arff(url: str, dest: str) -> None:
    print(f"[INFO] Downloading jm1 dataset from OpenML ...")
    print(f"       URL: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[INFO] Saved ARFF file → {dest}")
    except Exception as e:
        sys.exit(f"[ERROR] Download failed: {e}\n"
                 "        Please download jm1.csv manually from:\n"
                 "        https://www.openml.org/d/1053  (Export -> CSV)")


def arff_to_csv(arff_path: str, csv_path: str) -> None:
    """
    Parse a simple ARFF file and write it out as a CSV.
    Handles @relation, @attribute, and @data sections.
    """
    import csv

    print(f"[INFO] Converting ARFF → CSV ...")
    attributes = []   # column names in order
    data_rows   = []
    in_data     = False

    with open(arff_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()

            # Skip blank lines and comments
            if not stripped or stripped.startswith("%"):
                continue

            upper = stripped.upper()

            if upper.startswith("@ATTRIBUTE"):
                # e.g.  @attribute loc numeric
                parts = stripped.split()
                # parts[1] is the attribute name (may be quoted)
                attr_name = parts[1].strip("'\"")
                attributes.append(attr_name)

            elif upper.startswith("@DATA"):
                in_data = True

            elif in_data:
                # Each data line is a comma-separated record
                # Values may contain spaces; strip each token
                tokens = [t.strip().strip("'\"") for t in stripped.split(",")]
                data_rows.append(tokens)

    if not attributes:
        sys.exit("[ERROR] No @attribute declarations found in ARFF file. "
                 "The file may be corrupt or in an unexpected format.")

    # Rename the last attribute (class/target) to 'defects' so it matches
    # what defect_prediction.py expects.
    attributes[-1] = "defects"

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(attributes)
        writer.writerows(data_rows)

    print(f"[INFO] CSV saved → {csv_path}")
    print(f"       Rows: {len(data_rows)}  |  Columns: {len(attributes)}\n")


def main():
    # Resolve paths relative to repo root (one level up from scripts/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir  = os.path.join(repo_root, "data")
    os.makedirs(data_dir, exist_ok=True)   # create data/ if it doesn't exist
    arff_path = os.path.join(data_dir, "jm1.arff")
    csv_path  = os.path.join(data_dir, "jm1.csv")

    if os.path.exists(csv_path):
        print(f"[INFO] jm1.csv already exists at:\n       {csv_path}")
        print("       Delete it and re-run this script to re-download.\n")
        return

    download_arff(OPENML_URL, arff_path)
    arff_to_csv(arff_path, csv_path)

    # Clean up the intermediate ARFF file
    if os.path.exists(arff_path):
        os.remove(arff_path)
        print("[INFO] Temporary ARFF file removed.\n")

    print("[DONE] data/jm1.csv is ready. You can now run:\n"
          "       python src/defect_prediction.py\n")


if __name__ == "__main__":
    main()
