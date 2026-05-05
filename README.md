# Material & Colour Annotation Pipeline

A production-grade YOLOv8 segmentation pipeline that automatically updates **Material** and **Colour** attributes inside COCO-format annotation JSON files.

---

## 🚀 Features

- **Dual-model inference** — runs separate YOLOv8 segmentation models for material and colour prediction
- **Weighted IoU matching** — scores detections as `0.7 × IoU + 0.3 × confidence`
- **Batch GPU inference** — `stream=True` with configurable batch size for memory efficiency
- **Async I/O** — `ThreadPoolExecutor` for parallel image and JSON loading/saving
- **Atomic JSON writes** — writes to `.tmp` then renames to prevent corruption
- **Resume support** — skips already-inferred images with `--resume`
- **Flat polygon support** — handles `[x0,y0,x1,y1,…]` and `[[x,y],…]` formats
- **Attribute preservation** — only `Material` and `Colour` are overwritten; all other attributes are kept intact
- **Stats report** — per-run CSV summary + console output

---

## 📦 Requirements

```bash
pip install ultralytics opencv-python numpy tqdm torch
```

---

## 🛠️ Usage

### Full inference mode

```bash
python material_color_models_pipeline.py \
    --base_coco   /path/to/input_coco.json \
    --output_dir  /path/to/output \
    --images      /path/to/images \
    --material_model /path/to/material.pt \
    --color_model    /path/to/color.pt \
    --device   0 \
    --batch    16 \
    --imgsz    640 \
    --iou_thresh 0.1 \
    --workers  4
```

### Skip-inference mode (use pre-generated predictions)

```bash
python material_color_models_pipeline.py \
    --base_coco  /path/to/input_coco.json \
    --output_dir /path/to/output \
    --mat_pred   /path/to/material_predictions.json \
    --col_pred   /path/to/color_predictions.json
```

---

## ⚙️ Arguments

| Argument | Default | Description |
|---|---|---|
| `--base_coco` | required | Input COCO JSON file |
| `--output_dir` | required | Output directory |
| `--images` | — | Images folder (required for inference) |
| `--material_model` | — | Path to material `.pt` model |
| `--color_model` | — | Path to colour `.pt` model |
| `--mat_pred` | — | Pre-generated material predictions JSON |
| `--col_pred` | — | Pre-generated colour predictions JSON |
| `--device` | `0` | GPU index or `cpu` |
| `--batch` | `16` | Inference batch size |
| `--imgsz` | `640` | Inference image size |
| `--iou_thresh` | `0.1` | Minimum IoU threshold for mask matching |
| `--workers` | `4` | I/O threads |
| `--resume` | `False` | Skip already-processed images |
| `--chunk` | `0` | Limit to first N images (`0` = all) |
| `--log_level` | `INFO` | Logging verbosity |
| `--output_name` | `coco_updated.json` | Output filename |

---

## 📁 Output

- `coco_updated.json` — Updated COCO JSON with `Material` and `Colour` attributes filled
- `material_predictions.json` — Raw material model predictions (cached)
- `color_predictions.json` — Raw colour model predictions (cached)

---

## 🔄 Pipeline Stages

```
Stage 1a → Material model inference (YOLOv8 segmentation)
Stage 1b → Colour model inference  (YOLOv8 segmentation)
Stage 2  → Build prediction lookup indexes
Stage 3  → Match predictions to COCO annotations via IoU
         → Write updated COCO JSON (atomic)
```

---

## 📄 License

MIT
