# jobs/build_parquest_snapshot.py
# Only run when parquet for new images runs out

import os
import argparse
import pandas as pd
from huggingface_hub import hf_hub_download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="BigBang/galaxyzoo-decals")
    ap.add_argument("--filename", default="annotations/gz_decals_volunteers_5.parquet")
    ap.add_argument("--out", default="data/parquet/gz_decals_volunteers_5_filtered.parquet")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    parquet_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename=args.filename,
    )
    print("Downloaded parquet to:", parquet_path)

    df = pd.read_parquet(parquet_path, columns=["iauname", "ra", "dec", "wrong_size_warning"])

    df["iauname"] = df["iauname"].astype(str)
    df = df.dropna(subset=["ra", "dec"])

    if "wrong_size_warning" in df.columns:
        df = df[df["wrong_size_warning"].fillna(False) == False]

    print("rows after filtering:", len(df))

    df.to_parquet(args.out, index=False)
    print("Saved filtered parquet to:", args.out)

    # quick sanity
    df2 = pd.read_parquet(args.out, columns=["iauname", "ra", "dec"])
    print("reloaded rows:", len(df2))

if __name__ == "__main__":
    main()
