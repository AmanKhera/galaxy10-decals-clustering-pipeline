# galaxy_pipeline/cluster_map.py
# Updates galaxy10.clustered.csv

import pandas as pd

LABEL_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", "Cigar",
    "Barred Spiral", "Tight Spiral", "Loose Spiral", "Edge-on (no bulge)", "Edge-on (with bulge)"
]

def build_cluster_map(clustered_csv: str, out_csv: str) -> pd.DataFrame:
    df = pd.read_csv(clustered_csv)
    df = df[df["cluster_id"] != -1].copy()

    rows = []
    for cid in sorted(df["cluster_id"].unique()):
        sub = df[df["cluster_id"] == cid]
        vc = sub["true_label"].value_counts(normalize=True)

        dom_label = int(vc.index[0])
        purity = float(vc.iloc[0])

        top3 = vc.head(3)
        top3_str = "; ".join([f"{LABEL_NAMES[int(k)]}:{v:.3f}" for k, v in top3.items()])

        rows.append({
            "cluster_id": int(cid),
            "size": int(len(sub)),
            "dominant_label": dom_label,
            "dominant_name": LABEL_NAMES[dom_label],
            "purity": purity,
            "top3_breakdown": top3_str,
        })

    out = pd.DataFrame(rows).sort_values("cluster_id")
    out.to_csv(out_csv, index=False)
    return out
