# jobs/download_cutouts.py
# Downloads images from parquet and appends to manifest

import os, time, random, shutil
import argparse
import pandas as pd
import requests

BASE = "https://www.legacysurvey.org/viewer/cutout.jpg"

def cutout_url(ra, dec, layer="ls-dr10-grz", bands="grz", pixscale=0.262, size=256):
    return (f"{BASE}?ra={ra}&dec={dec}&layer={layer}"
            f"&bands={bands}&pixscale={pixscale}&size={size}")

def download_with_backoff(url, out_path, timeout=30, max_retries=8, base_sleep=0.5):
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 429 or (500 <= r.status_code < 600):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True, None
        except Exception as e:
            last_err = str(e)
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.25)
            time.sleep(min(sleep_s, 20))
    return False, last_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/parquet/gz_decals_volunteers_5_filtered.parquet")
    ap.add_argument("--out_dir", default="data/new_drop")
    ap.add_argument("--processed_dir", default="data/processed_images")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--layer", default="ls-dr10-grz")
    ap.add_argument("--bands", default="grz")
    ap.add_argument("--pixscale", type=float, default=0.262)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sleep_ok", type=float, default=0.2)
    ap.add_argument("--never_retry_same_iau", action="store_true")  # optional stricter mode
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)

    manifest_path = os.path.join(args.out_dir, "manifest.csv")

    # 0) Move old .jpg out of new_drop (keep manifest.csv)
    moved = 0
    for fname in os.listdir(args.out_dir):
        if not fname.lower().endswith(".jpg"):
            continue
        src = os.path.join(args.out_dir, fname)
        dst = os.path.join(args.processed_dir, fname)

        if os.path.exists(dst):
            base, ext = os.path.splitext(fname)
            k = 1
            while True:
                dst2 = os.path.join(args.processed_dir, f"{base}__dup{k}{ext}")
                if not os.path.exists(dst2):
                    dst = dst2
                    break
                k += 1

        shutil.move(src, dst)
        moved += 1
    print(f"Moved {moved} old images from {args.out_dir} -> {args.processed_dir}")

    # 1) Load parquet
    df = pd.read_parquet(args.parquet, columns=["iauname", "ra", "dec"])
    df["iauname"] = df["iauname"].astype(str)

    # 2) Load manifest to avoid repeats
    done_ok = set()
    attempted = set()
    if os.path.exists(manifest_path) and os.path.getsize(manifest_path) > 0:
        m = pd.read_csv(manifest_path)
        if "iauname" in m.columns:
            m["iauname"] = m["iauname"].astype(str)
            attempted = set(m["iauname"].dropna().unique())
            if "status" in m.columns:
                done_ok = set(m[m["status"].isin(["ok", "exists"])]["iauname"].dropna().unique())
            else:
                done_ok = attempted

    if args.never_retry_same_iau:
        done_ok = attempted

    # 3) Remaining candidates
    remaining = df[~df["iauname"].isin(done_ok)].copy()
    if len(remaining) == 0:
        print("All rows have already been processed (ok/exists).")
        raise SystemExit

    n_take = min(args.n, len(remaining))
    sample = remaining.sample(n_take).reset_index(drop=True)

    rows = []
    ok = 0

    for _, r in sample.iterrows():
        iau = str(r["iauname"])
        ra, dec = float(r["ra"]), float(r["dec"])
        url = cutout_url(ra, dec, layer=args.layer, bands=args.bands, pixscale=args.pixscale, size=args.size)
        out = os.path.join(args.out_dir, f"{iau}.jpg")

        if os.path.exists(out) and os.path.getsize(out) > 0:
            rows.append({"iauname": iau, "ra": ra, "dec": dec, "url": url, "path": out, "status": "exists"})
            ok += 1
            continue

        success, err = download_with_backoff(url, out)
        if success:
            rows.append({"iauname": iau, "ra": ra, "dec": dec, "url": url, "path": out, "status": "ok"})
            ok += 1
            time.sleep(args.sleep_ok)
        else:
            rows.append({"iauname": iau, "ra": ra, "dec": dec, "url": url, "path": out, "status": f"fail: {err}"})
            print("FAIL", iau, err)

    # 4) Append to manifest
    new_df = pd.DataFrame(rows)
    write_header = (not os.path.exists(manifest_path)) or (os.path.getsize(manifest_path) == 0)
    new_df.to_csv(manifest_path, mode="a", header=write_header, index=False)

    print(f"Downloaded/available: {ok}/{n_take}")
    print("Appended to:", manifest_path)
    print(f"Remaining after this run (approx): {len(remaining) - n_take}")

if __name__ == "__main__":
    main()
