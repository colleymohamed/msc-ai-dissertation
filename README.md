# AI-Driven Defect Detection in Composite Materials

A semi-supervised pipeline that detects defects in thermographic images of composite materials **without any manually labelled data**. Unsupervised clustering finds anomalous thermal states, an automated pseudo-labelling step turns those clusters into bounding boxes, and deep learning models are trained on the result.

> MSc Artificial Intelligence dissertation, Northumbria University (2025)

## Why this matters

Non-destructive testing (NDT) of composite materials is safety-critical, but training defect detectors normally needs large sets of expert-annotated images, which are slow and expensive to produce. This project asks: **can a model bootstrap from noisy, auto-generated labels and still be useful?**

## Pipeline

```
Thermographic frames
        │
        ▼
 K-Means clustering ──► separates normal vs. anomalous thermal states
        │
        ▼
 Pseudo-labelling ──► pixel-level difference maps + contour detection
        │              → YOLO-format bounding box annotations
        ▼
 Model training ──► YOLOv8n (detection) and ResNet (classification)
        │
        ▼
 Evaluation ──► mAP, accuracy, and annotation-cost vs. accuracy trade-offs
```

## Results

| Stage | Metric | Result |
|---|---|---|
| Unsupervised clustering (200 frames, K-Means) | Silhouette score | **0.52** |
| YOLOv8n detection (pseudo-labels) | mAP@50-95 | **0.153** |
| ResNet classification (pseudo-labels) | Accuracy | **97.5%** |

The detection score is modest, as expected when training on noisy labels. The main finding is that the approach works end to end with zero manual annotation, and the dissertation documents where the pseudo-label noise limits performance.

## Key contributions

- End-to-end semi-supervised workflow from raw thermal frames to a trained detector
- Automated pseudo-labelling that converts cluster outputs into YOLO-format annotations
- Comparison of detection (YOLOv8n) and classification (ResNet) models on auto-generated labels
- Analysis of trade-offs between annotation cost, model accuracy, and pseudo-label noise for low-label industrial inspection

## Tech stack

Python · PyTorch · Ultralytics YOLOv8 · scikit-learn · OpenCV · NumPy · Pandas · Matplotlib · Jupyter

## Repository structure

<!-- Update to match your actual files -->
```
├── notebooks/      # clustering, pseudo-labelling, training, evaluation
├── data/           # thermographic frames (see note below)
├── labels/         # generated YOLO-format annotations
├── results/        # metrics, plots, trained weights
└── README.md
```

Run the notebooks in order: clustering → pseudo-labelling → training → evaluation.

## Data

<!-- State clearly whether the dataset is included, and its source/licence. If it is not public, say so here. -->
The thermographic dataset is not redistributed here. Add your own frames to `data/` to reproduce the pipeline.

## Limitations and future work

- Pseudo-label quality caps detection performance (mAP@50-95 of 0.153)
- Clustering was run on a small set of 200 frames
- Future work: iterative label refinement, larger datasets, and testing other detectors

## Author

**Muhammed Colley** · [LinkedIn](https://linkedin.com/in/colleymuhammed) · [Portfolio](http://muhammedcolley.com/)
