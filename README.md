# galaxy10-decals-clustering-pipeline
End-to-end Galaxy10 DECaLS morphology pipeline: CNN encoder creates embeddings, then StandardScaler+PCA+HDBSCAN cluster galaxies. Includes kNN online cluster assignment, anomaly scoring/routing, cutout ingestion, visualizations, and reproducible artifacts/logs for rebuildable results.

**Performance note:** optimized the training/inference input pipeline to reduce CPU bottlenecks from **~22% host** to **~8.1% host**, increasing GPU utilization to **~91.9% device**.


---

## Features

- **Training**
  - Transfer learning backbone (e.g., EfficientNet / SwinV2) + lightweight head
  - Mixed precision + XLA (jit_compile) friendly setup
  - Optional fine-tuning stage and regularization (AdamW, L2, dropout)
- **Embeddings**
  - Extract embeddings for Galaxy10 and new images
  - Save reusable artifacts (encoder, scaler, PCA, HDBSCAN, reference embeddings)
- **Clustering**
  - StandardScaler → PCA (e.g., 20–50 dims) → HDBSCAN clustering
  - Cluster summaries (size, dominant label, purity)
- **Incremental / Online Use**
  - kNN-based assignment of new embeddings to existing clusters
  - Anomaly scoring + routing (distance / low-confidence / noise handling)
- **New Image Ingestion (Galaxy Zoo / DECaLS)**
  - Downloads a Galaxy Zoo DECaLS parquet snapshot from Hugging Face, filters valid targets, and saves a local parquet copy
  - Samples new (ra, dec) targets and downloads cutout JPGs from Legacy Surveys (`legacysurvey.org`)
  - Maintains a `manifest.csv` to avoid re-downloading previously successful images
  - Automatically moves previously downloaded images from `data/new_drop/` → `data/processed_images/` before fetching a new batch
- **Outputs**
  - CSV logs of predictions + cluster IDs + anomaly scores
  - Visualization helpers (cluster grids, anomaly galleries, confidence histograms)

---
