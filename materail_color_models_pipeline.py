# #!/usr/bin/env python3
# """
# Annotation Attribute Update Pipeline  ─ Industrial Grade
# =========================================================
# Updates **Material** and **Colour** attributes inside
#   data["items"][N]["annotations"][M]["attributes"]
# using two YOLOv8 segmentation models.

# Key improvements over v1
# ─────────────────────────
# • Correct JSON schema  : items[].annotations[].attributes
# • Flat polygon support : [x0,y0,x1,y1,…]  →  [[x,y],…]
# • Batch GPU inference  : stream=True + configurable batch size
# • cudnn.benchmark      : faster for fixed-size inputs
# • Weighted IoU score   : 0.7*IoU + 0.3*conf
# • Async I/O            : ThreadPoolExecutor for load/save
# • Memory-safe          : processes work in chunks, no full list in RAM
# • Stats report         : per-run CSV + console summary
# • Atomic JSON writes   : tmp → rename
# • Full attribute pass  : preserves ALL existing attributes; only
#                          Material & Colour are overwritten

# Usage
# ─────
# python pipeline.py \\
#     --images   /path/to/images \\
#     --jsons    /path/to/jsons \\
#     --material_model /path/to/material.pt \\
#     --color_model    /path/to/color.pt \\
#     --output   /path/to/output \\
#     --device   0 \\
#     --batch    8 \\
#     --imgsz    640 \\
#     --iou_thresh 0.1 \\
#     --workers  4
# """

# from __future__ import annotations

# import argparse
# import copy
# import csv
# import json
# import logging
# import os
# import sys
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import cv2
# import numpy as np

# # ── tqdm ──────────────────────────────────────────────────────────────────────
# try:
#     from tqdm import tqdm
# except ImportError:
#     print("[WARN] tqdm not installed – pip install tqdm")

#     class tqdm:  # type: ignore[no-reuse-declaration]
#         def __init__(self, iterable=None, **kw):
#             self._it = iterable
#             desc = kw.get("desc", "")
#             if desc:
#                 print(f"{desc} ...")
#             self.total = kw.get("total")

#         def __iter__(self):
#             return iter(self._it)

#         def __enter__(self):
#             return self

#         def __exit__(self, *_):
#             pass

#         def update(self, n=1):
#             pass

#         def set_postfix(self, **_):
#             pass

#         def write(self, s):
#             print(s)


# # ── ultralytics ───────────────────────────────────────────────────────────────
# try:
#     from ultralytics import YOLO
# except ImportError:
#     print("[ERROR] ultralytics not installed – pip install ultralytics")
#     sys.exit(1)

# # ── torch ─────────────────────────────────────────────────────────────────────
# try:
#     import torch

#     _TORCH_AVAILABLE = True
# except ImportError:
#     _TORCH_AVAILABLE = False

# # ─────────────────────────────────────────────────────────────────────────────
# # Logging
# # ─────────────────────────────────────────────────────────────────────────────
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)-8s | %(message)s",
#     datefmt="%H:%M:%S",
# )
# log = logging.getLogger("pipeline")

# # ─────────────────────────────────────────────────────────────────────────────
# # Constants
# # ─────────────────────────────────────────────────────────────────────────────
# DEFAULT_MATERIAL = "PP"
# DEFAULT_COLOUR = "Unknown"
# IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# # Weighted score blend  (IoU weight + conf weight must sum to 1.0)
# IOU_WEIGHT = 0.7
# CONF_WEIGHT = 0.3


# # ─────────────────────────────────────────────────────────────────────────────
# # Device / hardware helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def parse_device(device_str: str) -> str:
#     s = str(device_str).strip().lower()
#     if s == "cpu":
#         return "cpu"
#     try:
#         idx = int(s)
#         if _TORCH_AVAILABLE and torch.cuda.is_available():
#             if _TORCH_AVAILABLE:
#                 torch.backends.cudnn.benchmark = True  # faster fixed-size inference
#             return str(idx)
#         log.warning("CUDA not available – falling back to CPU")
#         return "cpu"
#     except ValueError:
#         return s


# def supports_half(device: str) -> bool:
#     if not _TORCH_AVAILABLE or device == "cpu":
#         return False
#     try:
#         return torch.cuda.is_available()
#     except Exception:
#         return False


# # ─────────────────────────────────────────────────────────────────────────────
# # Geometry helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def flat_to_pairs(flat: List[float]) -> List[List[float]]:
#     """Convert [x0,y0,x1,y1,…] to [[x0,y0],[x1,y1],…]."""
#     return [[flat[i], flat[i + 1]] for i in range(0, len(flat) - 1, 2)]


# def parse_points(ann: Dict[str, Any]) -> Optional[List[List[float]]]:
#     """
#     Extract polygon points from an annotation dict.
#     Handles:
#       • "points": [x,y,x,y,…]        ← your actual format (flat list)
#       • "points": [[x,y],[x,y],…]     ← list-of-pairs
#       • "segmentation": [[x,y,…]]     ← COCO flat
#       • "polygon": [[x,y],…]
#     Returns list of [x,y] pairs or None.
#     """
#     # ── your actual schema: annotation["points"] = flat list ──────────────
#     if "points" in ann:
#         raw = ann["points"]
#         if isinstance(raw, list) and len(raw) >= 6:
#             # already pairs?
#             if isinstance(raw[0], (list, tuple)):
#                 return [list(p) for p in raw]
#             # flat float/int list
#             if len(raw) % 2 == 0:
#                 return flat_to_pairs(raw)
#             # odd-length: drop last
#             return flat_to_pairs(raw[:-1])

#     # ── COCO segmentation ──────────────────────────────────────────────────
#     if "segmentation" in ann:
#         seg = ann["segmentation"]
#         if isinstance(seg, list) and len(seg) > 0:
#             flat = seg[0] if isinstance(seg[0], list) else seg
#             if isinstance(flat, list) and len(flat) >= 6:
#                 if isinstance(flat[0], (list, tuple)):
#                     return [list(p) for p in flat]
#                 if len(flat) % 2 == 0:
#                     return flat_to_pairs(flat)

#     # ── explicit polygon key ───────────────────────────────────────────────
#     if "polygon" in ann:
#         raw = ann["polygon"]
#         if isinstance(raw, list) and len(raw) >= 3:
#             if isinstance(raw[0], (list, tuple)):
#                 return [list(p) for p in raw]
#             if len(raw) >= 6 and len(raw) % 2 == 0:
#                 return flat_to_pairs(raw)

#     return None


# def polygon_to_mask(points: List[List[float]], h: int, w: int) -> np.ndarray:
#     pts = np.array(points, dtype=np.float32).reshape(-1, 2)
#     pts_int = pts.astype(np.int32)
#     mask = np.zeros((h, w), dtype=np.uint8)
#     cv2.fillPoly(mask, [pts_int], 1)
#     return mask


# def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
#     inter = np.logical_and(a, b).sum()
#     union = np.logical_or(a, b).sum()
#     return float(inter) / float(union) if union else 0.0


# # ─────────────────────────────────────────────────────────────────────────────
# # I/O helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def load_image_safe(path: Path) -> Optional[np.ndarray]:
#     try:
#         img = cv2.imread(str(path))
#         if img is None:
#             raise ValueError("cv2.imread returned None")
#         return img
#     except Exception as exc:
#         log.warning("Unreadable image %s – %s", path.name, exc)
#         return None


# def find_json_for_image(img_path: Path, json_dir: Path) -> Optional[Path]:
#     stem = img_path.stem
#     candidate = json_dir / (stem + ".json")
#     if candidate.exists():
#         return candidate
#     stem_lower = stem.lower()
#     for p in json_dir.glob("*.json"):
#         if p.stem.lower() == stem_lower:
#             return p
#     return None


# def load_json_safe(path: Path) -> Optional[Dict]:
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception as exc:
#         log.warning("Bad JSON %s – %s", path.name, exc)
#         return None


# def save_json(data: Dict, path: Path) -> None:
#     """Atomic write: write to .tmp then rename."""
#     path.parent.mkdir(parents=True, exist_ok=True)
#     tmp = path.with_suffix(".tmp")
#     try:
#         with open(tmp, "w", encoding="utf-8") as f:
#             json.dump(data, f, ensure_ascii=False, indent=2)
#         tmp.replace(path)
#     except Exception:
#         tmp.unlink(missing_ok=True)
#         raise


# # ─────────────────────────────────────────────────────────────────────────────
# # YOLOv8 model wrapper
# # ─────────────────────────────────────────────────────────────────────────────

# Detection = Tuple[np.ndarray, str, float]   # (binary_mask, class_name, conf)


# class SegModel:
#     """Thin wrapper around a YOLOv8 segmentation model."""

#     def __init__(self, model_path: str, device: str, half: bool, imgsz: int):
#         log.info("Loading model: %s | device=%s | half=%s", model_path, device, half)
#         self.model = YOLO(model_path)
#         self.device = device
#         self.half = half
#         self.imgsz = imgsz
#         self.names: Dict[int, str] = self.model.names
#         self._warmup()

#     def _warmup(self) -> None:
#         dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
#         try:
#             self.model.predict(
#                 dummy, device=self.device, half=self.half,
#                 imgsz=self.imgsz, verbose=False,
#             )
#         except Exception as exc:
#             log.warning("Warmup failed (non-fatal): %s", exc)

#     # ── Single-image inference ─────────────────────────────────────────────
#     def predict_one(self, img: np.ndarray) -> List[Detection]:
#         results = self.model.predict(
#             img, device=self.device, half=self.half,
#             imgsz=self.imgsz, verbose=False,
#         )
#         return self._parse_results(results, img.shape[0], img.shape[1])

#     # ── Batch inference (list of BGR images) ──────────────────────────────
#     def predict_batch(self, imgs: List[np.ndarray]) -> List[List[Detection]]:
#         """
#         Run inference on a list of BGR images.
#         Returns one list of Detections per image.
#         stream=True keeps memory low for large batches.
#         """
#         results_per_image: List[List[Detection]] = []
#         # ultralytics accepts a list of arrays directly
#         gen = self.model.predict(
#             imgs,
#             device=self.device,
#             half=self.half,
#             imgsz=self.imgsz,
#             verbose=False,
#             stream=True,    # generator → low peak memory
#         )
#         for i, r in enumerate(gen):
#             h, w = imgs[i].shape[:2]
#             results_per_image.append(self._parse_results([r], h, w))
#         return results_per_image

#     # ── Parse a list[Result] → List[Detection] ────────────────────────────
#     @staticmethod
#     def _parse_results(results, h: int, w: int) -> List[Detection]:
#         detections: List[Detection] = []
#         if not results:
#             return detections
#         r = results[0]
#         if r.masks is None:
#             return detections
#         masks_data = r.masks.data.cpu().numpy()   # (N, Hm, Wm)
#         boxes = r.boxes
#         for i, raw_mask in enumerate(masks_data):
#             mask_resized = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
#             bin_mask = (mask_resized > 0.5).astype(np.uint8)
#             cls_id = int(boxes.cls[i].item())
#             conf = float(boxes.conf[i].item())
#             cls_name = r.names.get(cls_id, str(cls_id))
#             detections.append((bin_mask, cls_name, conf))
#         return detections


# # ─────────────────────────────────────────────────────────────────────────────
# # Matching
# # ─────────────────────────────────────────────────────────────────────────────

# def best_match(
#     poly_mask: np.ndarray,
#     detections: List[Detection],
#     iou_thresh: float,
# ) -> Optional[str]:
#     """
#     Return class_name with highest weighted score (0.7*IoU + 0.3*conf)
#     among detections whose IoU >= iou_thresh.
#     """
#     best_score = -1.0
#     best_cls: Optional[str] = None
#     for det_mask, cls_name, conf in detections:
#         iou = mask_iou(poly_mask, det_mask)
#         if iou < iou_thresh:
#             continue
#         score = IOU_WEIGHT * iou + CONF_WEIGHT * conf
#         if score > best_score:
#             best_score = score
#             best_cls = cls_name
#     return best_cls


# # ─────────────────────────────────────────────────────────────────────────────
# # Annotation update  (preserves all existing attributes)
# # ─────────────────────────────────────────────────────────────────────────────

# def update_annotation(
#     ann: Dict[str, Any],
#     mat_dets: List[Detection],
#     col_dets: List[Detection],
#     img_h: int,
#     img_w: int,
#     iou_thresh: float,
# ) -> Dict[str, Any]:
#     """
#     Deep-copy ann, match masks, update Material + Colour only.
#     All other attributes are preserved exactly.
#     """
#     ann = copy.deepcopy(ann)

#     points = parse_points(ann)
#     if points is None or len(points) < 3:
#         _write_attrs(ann, None, None)
#         return ann

#     try:
#         poly_mask = polygon_to_mask(points, img_h, img_w)
#     except Exception as exc:
#         log.debug("polygon_to_mask failed: %s", exc)
#         _write_attrs(ann, None, None)
#         return ann

#     matched_mat = best_match(poly_mask, mat_dets, iou_thresh)
#     matched_col = best_match(poly_mask, col_dets, iou_thresh)
#     _write_attrs(ann, matched_mat, matched_col)
#     return ann


# def _write_attrs(
#     ann: Dict[str, Any],
#     mat: Optional[str],
#     col: Optional[str],
# ) -> None:
#     """Write Material + Colour into ann["attributes"], leaving rest intact."""
#     if "attributes" not in ann or not isinstance(ann["attributes"], dict):
#         ann["attributes"] = {}
#     ann["attributes"]["Material"] = mat if mat is not None else DEFAULT_MATERIAL
#     ann["attributes"]["Colour"]   = col if col is not None else DEFAULT_COLOUR


# # ─────────────────────────────────────────────────────────────────────────────
# # Per-image processing  ← handles your exact JSON schema
# # ─────────────────────────────────────────────────────────────────────────────

# def process_image(
#     image_path: Path,
#     json_path: Path,
#     output_path: Path,
#     mat_model: SegModel,
#     col_model: SegModel,
#     iou_thresh: float,
# ) -> Tuple[bool, str, int, int]:
#     """
#     Process one image + JSON file pair.

#     JSON schema handled (in priority order):
#       1. data["items"][N]["annotations"]   ← YOUR FORMAT
#       2. data["annotations"] / ["objects"] / ["shapes"] / ["labels"]
#       3. The root dict itself if it has polygon keys

#     Returns (success, message, ann_total, ann_updated).
#     """
#     img = load_image_safe(image_path)
#     if img is None:
#         return False, f"Corrupt image: {image_path.name}", 0, 0

#     img_h, img_w = img.shape[:2]

#     data = load_json_safe(json_path)
#     if data is None:
#         return False, f"Bad JSON: {json_path.name}", 0, 0

#     # Run models
#     try:
#         mat_dets = mat_model.predict_one(img)
#     except Exception as exc:
#         log.warning("Material model failed on %s – %s", image_path.name, exc)
#         mat_dets = []

#     try:
#         col_dets = col_model.predict_one(img)
#     except Exception as exc:
#         log.warning("Colour model failed on %s – %s", image_path.name, exc)
#         col_dets = []

#     out_data = copy.deepcopy(data)
#     ann_total = 0
#     ann_updated = 0

#     # ── Schema 1: data["items"][N]["annotations"]  (YOUR FORMAT) ──────────
#     if "items" in out_data and isinstance(out_data["items"], list):
#         for item in out_data["items"]:
#             if not isinstance(item, dict):
#                 continue
#             anns = item.get("annotations")
#             if not isinstance(anns, list):
#                 continue
#             updated = []
#             for ann in anns:
#                 ann_total += 1
#                 new_ann = update_annotation(ann, mat_dets, col_dets, img_h, img_w, iou_thresh)
#                 updated.append(new_ann)
#                 ann_updated += 1
#             item["annotations"] = updated
#         save_json(out_data, output_path)
#         return True, "ok", ann_total, ann_updated

#     # ── Schema 2: flat top-level annotation list ───────────────────────────
#     for key in ("annotations", "objects", "shapes", "labels"):
#         if key in out_data and isinstance(out_data[key], list):
#             updated = []
#             for ann in out_data[key]:
#                 ann_total += 1
#                 new_ann = update_annotation(ann, mat_dets, col_dets, img_h, img_w, iou_thresh)
#                 updated.append(new_ann)
#                 ann_updated += 1
#             out_data[key] = updated
#             save_json(out_data, output_path)
#             return True, "ok", ann_total, ann_updated

#     # ── Schema 3: root dict is itself an annotation ────────────────────────
#     if any(k in out_data for k in ("segmentation", "polygon", "points")):
#         ann_total = 1
#         updated = update_annotation(out_data, mat_dets, col_dets, img_h, img_w, iou_thresh)
#         save_json(updated, output_path)
#         return True, "single-annotation root", 1, 1

#     # No annotations found – save as-is
#     save_json(out_data, output_path)
#     return True, "no annotations found – saved as-is", 0, 0


# # ─────────────────────────────────────────────────────────────────────────────
# # Batch processing  (GPU batching for throughput)
# # ─────────────────────────────────────────────────────────────────────────────

# def process_batch(
#     batch: List[Tuple[Path, Path, Path]],
#     mat_model: SegModel,
#     col_model: SegModel,
#     iou_thresh: float,
#     workers: int,
# ) -> List[Tuple[bool, str, int, int]]:
#     """
#     Load images in parallel, run GPU batch inference, update JSONs in parallel.

#     Returns list of (success, msg, ann_total, ann_updated) per item.
#     """
#     n = len(batch)
#     img_paths  = [t[0] for t in batch]
#     json_paths = [t[1] for t in batch]
#     out_paths  = [t[2] for t in batch]

#     # ── Parallel image load ────────────────────────────────────────────────
#     imgs: List[Optional[np.ndarray]] = [None] * n
#     with ThreadPoolExecutor(max_workers=min(workers, n)) as ex:
#         futs = {ex.submit(load_image_safe, p): i for i, p in enumerate(img_paths)}
#         for f in as_completed(futs):
#             imgs[futs[f]] = f.result()

#     # ── Parallel JSON load ─────────────────────────────────────────────────
#     jsons: List[Optional[Dict]] = [None] * n
#     with ThreadPoolExecutor(max_workers=min(workers, n)) as ex:
#         futs = {ex.submit(load_json_safe, p): i for i, p in enumerate(json_paths)}
#         for f in as_completed(futs):
#             jsons[futs[f]] = f.result()

#     # ── Filter valid images for batch inference ────────────────────────────
#     valid_idx = [i for i, im in enumerate(imgs) if im is not None]
#     valid_imgs = [imgs[i] for i in valid_idx]

#     mat_batch: Dict[int, List[Detection]] = {}
#     col_batch: Dict[int, List[Detection]] = {}

#     if valid_imgs:
#         try:
#             mat_results = mat_model.predict_batch(valid_imgs)
#             for k, i in enumerate(valid_idx):
#                 mat_batch[i] = mat_results[k]
#         except Exception as exc:
#             log.warning("Batch material inference failed – %s", exc)
#             for i in valid_idx:
#                 mat_batch[i] = []

#         try:
#             col_results = col_model.predict_batch(valid_imgs)
#             for k, i in enumerate(valid_idx):
#                 col_batch[i] = col_results[k]
#         except Exception as exc:
#             log.warning("Batch colour inference failed – %s", exc)
#             for i in valid_idx:
#                 col_batch[i] = []

#     # ── Update JSONs ───────────────────────────────────────────────────────
#     results: List[Tuple[bool, str, int, int]] = []

#     def _process_one(idx: int) -> Tuple[bool, str, int, int]:
#         img = imgs[idx]
#         data = jsons[idx]
#         out_path = out_paths[idx]
#         img_path = img_paths[idx]

#         if img is None:
#             return False, f"Corrupt image: {img_path.name}", 0, 0
#         if data is None:
#             return False, f"Bad JSON: {json_paths[idx].name}", 0, 0

#         img_h, img_w = img.shape[:2]
#         mat_dets = mat_batch.get(idx, [])
#         col_dets = col_batch.get(idx, [])

#         out_data = copy.deepcopy(data)
#         ann_total = 0
#         ann_updated = 0

#         # Schema 1: items[].annotations
#         if "items" in out_data and isinstance(out_data["items"], list):
#             for item in out_data["items"]:
#                 if not isinstance(item, dict):
#                     continue
#                 anns = item.get("annotations")
#                 if not isinstance(anns, list):
#                     continue
#                 updated = []
#                 for ann in anns:
#                     ann_total += 1
#                     new_ann = update_annotation(ann, mat_dets, col_dets, img_h, img_w, iou_thresh)
#                     updated.append(new_ann)
#                     ann_updated += 1
#                 item["annotations"] = updated
#             save_json(out_data, out_path)
#             return True, "ok", ann_total, ann_updated

#         # Schema 2: top-level list
#         for key in ("annotations", "objects", "shapes", "labels"):
#             if key in out_data and isinstance(out_data[key], list):
#                 updated = []
#                 for ann in out_data[key]:
#                     ann_total += 1
#                     new_ann = update_annotation(ann, mat_dets, col_dets, img_h, img_w, iou_thresh)
#                     updated.append(new_ann)
#                     ann_updated += 1
#                 out_data[key] = updated
#                 save_json(out_data, out_path)
#                 return True, "ok", ann_total, ann_updated

#         # Schema 3: root annotation
#         if any(k in out_data for k in ("segmentation", "polygon", "points")):
#             updated = update_annotation(out_data, mat_dets, col_dets, img_h, img_w, iou_thresh)
#             save_json(updated, out_path)
#             return True, "single-annotation root", 1, 1

#         save_json(out_data, out_path)
#         return True, "no annotations – saved as-is", 0, 0

#     # Run JSON updates in parallel (I/O bound)
#     with ThreadPoolExecutor(max_workers=min(workers, n)) as ex:
#         futs = {ex.submit(_process_one, i): i for i in range(n)}
#         for f in as_completed(futs):
#             results.append(f.result())

#     return results


# # ─────────────────────────────────────────────────────────────────────────────
# # Stats report
# # ─────────────────────────────────────────────────────────────────────────────

# def write_stats_csv(
#     output_dir: Path,
#     work: List[Tuple[Path, Path, Path]],
#     outcomes: List[Tuple[bool, str, int, int]],
# ) -> Path:
#     """Write a CSV report of per-file results."""
#     report_path = output_dir / "_pipeline_report.csv"
#     with open(report_path, "w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["image", "json", "success", "message", "annotations_found", "annotations_updated"])
#         for (img_p, json_p, _out), (ok, msg, total, updated) in zip(work, outcomes):
#             w.writerow([img_p.name, json_p.name, ok, msg, total, updated])
#     return report_path


# # ─────────────────────────────────────────────────────────────────────────────
# # CLI
# # ─────────────────────────────────────────────────────────────────────────────

# def build_parser() -> argparse.ArgumentParser:
#     p = argparse.ArgumentParser(
#         description="Annotation Attribute Update Pipeline (industrial grade)",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     p.add_argument("--images",          required=True, help="Dir of input images")
#     p.add_argument("--jsons",           required=True, help="Dir of input JSON annotations")
#     p.add_argument("--material_model",  required=True, help="Path to material .pt model")
#     p.add_argument("--color_model",     required=True, help="Path to colour .pt model")
#     p.add_argument("--output",          required=True, help="Dir for output JSONs")
#     p.add_argument("--device",          default="0",   help="0 / 1 / cpu")
#     p.add_argument("--batch",           type=int, default=8,   help="GPU batch size")
#     p.add_argument("--imgsz",           type=int, default=640, help="Inference image size")
#     p.add_argument("--iou_thresh",      type=float, default=0.1, help="Min IoU for mask match")
#     p.add_argument("--workers",         type=int, default=4,   help="I/O threads")
#     p.add_argument("--overwrite",       action="store_true",   help="Reprocess existing outputs")
#     p.add_argument("--suffix",          default="",            help="Output filename suffix before .json")
#     p.add_argument("--report",          action="store_true",   help="Write CSV stats report")
#     p.add_argument("--chunk",           type=int, default=0,
#                    help="Process only first N images (0 = all; useful for dry-runs)")
#     p.add_argument("--log_level",       default="INFO",
#                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
#     return p


# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     parser = build_parser()
#     args = parser.parse_args()

#     logging.getLogger().setLevel(getattr(logging, args.log_level))

#     images_dir    = Path(args.images)
#     jsons_dir     = Path(args.jsons)
#     output_dir    = Path(args.output)
#     mat_model_path = Path(args.material_model)
#     col_model_path = Path(args.color_model)

#     for label, path in [
#         ("--images",         images_dir),
#         ("--jsons",          jsons_dir),
#         ("--material_model", mat_model_path),
#         ("--color_model",    col_model_path),
#     ]:
#         if not path.exists():
#             log.error("%s path not found: %s", label, path)
#             sys.exit(1)

#     output_dir.mkdir(parents=True, exist_ok=True)

#     device = parse_device(args.device)
#     half   = supports_half(device)
#     log.info("Device: %s | FP16: %s | batch: %d | imgsz: %d",
#              device, half, args.batch, args.imgsz)

#     # ── Discover images ────────────────────────────────────────────────────
#     image_files: List[Path] = sorted(
#         p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS
#     )
#     if not image_files:
#         log.error("No images found in %s", images_dir)
#         sys.exit(1)
#     log.info("Found %d image(s)", len(image_files))

#     # ── Load models ────────────────────────────────────────────────────────
#     try:
#         mat_model = SegModel(str(mat_model_path), device, half, args.imgsz)
#         col_model = SegModel(str(col_model_path), device, half, args.imgsz)
#     except Exception as exc:
#         log.error("Model load failed: %s", exc)
#         sys.exit(1)

#     # ── Build work list ────────────────────────────────────────────────────
#     work: List[Tuple[Path, Path, Path]] = []
#     skipped_no_json = 0
#     skipped_exists  = 0

#     for img_path in image_files:
#         json_path = find_json_for_image(img_path, jsons_dir)
#         if json_path is None:
#             skipped_no_json += 1
#             log.debug("No JSON for %s", img_path.name)
#             continue

#         out_name = img_path.stem + args.suffix + ".json"
#         out_path = output_dir / out_name

#         if out_path.exists() and not args.overwrite:
#             skipped_exists += 1
#             continue

#         work.append((img_path, json_path, out_path))

#     if args.chunk > 0:
#         work = work[: args.chunk]
#         log.info("Dry-run chunk: processing first %d items", len(work))

#     log.info(
#         "Work: %d | No JSON: %d | Already done (skipped): %d",
#         len(work), skipped_no_json, skipped_exists,
#     )

#     if not work:
#         log.info("Nothing to do. Use --overwrite to reprocess existing outputs.")
#         return

#     # ── Main processing loop (batched) ────────────────────────────────────
#     batch_size = max(1, args.batch)
#     stats   = {"ok": 0, "fail": 0, "ann_total": 0, "ann_updated": 0}
#     all_outcomes: List[Tuple[bool, str, int, int]] = []
#     t_start = time.perf_counter()

#     with tqdm(total=len(work), desc="Processing", unit="img") as pbar:
#         for start in range(0, len(work), batch_size):
#             batch = work[start : start + batch_size]
#             try:
#                 outcomes = process_batch(batch, mat_model, col_model,
#                                          args.iou_thresh, args.workers)
#             except Exception as exc:
#                 log.error("Batch [%d:%d] crashed: %s", start, start + len(batch), exc)
#                 outcomes = [(False, str(exc), 0, 0)] * len(batch)

#             for (img_p, _, _), (ok, msg, total, updated) in zip(batch, outcomes):
#                 all_outcomes.append((ok, msg, total, updated))
#                 if ok:
#                     stats["ok"] += 1
#                     stats["ann_total"]   += total
#                     stats["ann_updated"] += updated
#                     if msg != "ok":
#                         log.debug("%s – %s", img_p.name, msg)
#                 else:
#                     stats["fail"] += 1
#                     pbar.write(f"[WARN] {img_p.name}: {msg}")

#             pbar.update(len(batch))
#             pbar.set_postfix(
#                 ok=stats["ok"],
#                 fail=stats["fail"],
#                 ann=stats["ann_updated"],
#             )

#     elapsed    = time.perf_counter() - t_start
#     throughput = stats["ok"] / elapsed if elapsed > 0 else 0.0

#     # ── Summary ───────────────────────────────────────────────────────────
#     log.info("=" * 60)
#     log.info("PIPELINE COMPLETE")
#     log.info("  Images processed : %d", stats["ok"])
#     log.info("  Images failed    : %d", stats["fail"])
#     log.info("  Annotations found: %d", stats["ann_total"])
#     log.info("  Annotations updated: %d", stats["ann_updated"])
#     log.info("  Throughput       : %.1f img/s", throughput)
#     log.info("  Total time       : %.1fs", elapsed)
#     log.info("  Output dir       : %s", output_dir)
#     log.info("=" * 60)

#     # ── Optional CSV report ───────────────────────────────────────────────
#     if args.report:
#         report_path = write_stats_csv(output_dir, work, all_outcomes)
#         log.info("Stats report written to: %s", report_path)


# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
"""
COCO Annotation Attribute Update Pipeline  ─ Production Grade
==============================================================
Updates Material and Colour attributes in a base COCO JSON file
using predictions from two YOLOv8 segmentation models.

Architecture (3-file design):
  1. Base COCO JSON     → source of truth for annotations
  2. Material predictions COCO  → generated by running material model on images
  3. Color predictions COCO     → generated by running color model on images

  For each annotation in Base COCO:
    • Find material prediction on same image_id with best IoU overlap
    • Find color prediction on same image_id with best IoU overlap
    • Update annotation["attributes"]["Material"] and ["Colour"]

Usage:
------
  # Full run (inference + attribute update)
  python3 materail_color_models_pipeline.py \\
      --images        /home/wi/Avinash_Works/waste-masknet/waste/data/images \\
      --base_coco     /media/wi/ssd_hub/Ishika_works/coco_combined.json \\
      --material_model /media/wi/ssd_hub/training_runs/yolov8m_seg_real_only_20260423_175631/weights/best.pt \\
      --color_model    /media/wi/ssd_hub/Ishika_works/training_runs/yolov8m_color_seg/weights/best.pt \\
      --output_dir     /media/wi/ssd_hub/Ishika_works/pipeline_outputs/output \\
      --device 0 \\
      --batch  16

  # Resume interrupted run (skips images already predicted)
  python coco_attribute_updater.py ... --resume

  # Dry run: only first 100 images
  python coco_attribute_updater.py ... --chunk 100

  # Skip inference (reuse existing prediction files)
  python coco_attribute_updater.py ... \\
      --mat_pred  /path/to/material_predictions.json \\
      --col_pred  /path/to/color_predictions.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    print("[WARN] tqdm not installed: pip install tqdm")
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kw):
            self._it = iterable
            if kw.get("desc"):
                print(f"{kw['desc']} ...")
        def __iter__(self): return iter(self._it)
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def update(self, n=1): pass
        def set_postfix(self, **_): pass
        def write(self, s): print(s)

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed: pip install ultralytics")
    sys.exit(1)

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("coco_updater")

# ── constants ─────────────────────────────────────────────────────────────────
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
DEFAULT_MATERIAL = "PP"
DEFAULT_COLOUR   = "Unknown"
IOU_WEIGHT       = 0.7
CONF_WEIGHT      = 0.3

# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def flat_to_pairs(flat: List[float]) -> np.ndarray:
    arr = np.array(flat, dtype=np.float32).reshape(-1, 2)
    return arr


def seg_to_mask(segmentation: List, h: int, w: int) -> np.ndarray:
    """Convert COCO segmentation (list of flat polygons) to a binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in segmentation:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def bbox_to_mask(bbox: List[float], h: int, w: int) -> np.ndarray:
    """Fallback: convert COCO bbox [x,y,w,h] to binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = [int(v) for v in bbox]
    x2, y2 = min(x + bw, w), min(y + bh, h)
    mask[max(y, 0):y2, max(x, 0):x2] = 1
    return mask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def mask_iou_batch(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Vectorised IoU: query (H,W) vs targets (N,H,W).
    Returns array of N floats.
    """
    inter = np.logical_and(query[None], targets).sum(axis=(1, 2))
    union = np.logical_or(query[None], targets).sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Device helpers
# ══════════════════════════════════════════════════════════════════════════════

def resolve_device(device_str: str) -> str:
    s = str(device_str).strip().lower()
    if s == "cpu":
        return "cpu"
    try:
        idx = int(s)
        if _TORCH and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            return str(idx)
        log.warning("CUDA not available — falling back to CPU")
        return "cpu"
    except ValueError:
        return s


def supports_half(device: str) -> bool:
    return _TORCH and device != "cpu" and torch.cuda.is_available()


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(data: Dict, path: Path) -> None:
    """Write to .tmp then rename — safe against partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_image(path: Path) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError("cv2.imread returned None")
        return img
    except Exception as exc:
        log.warning("Cannot read image %s: %s", path.name, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# YOLOv8 segmentation model wrapper
# ══════════════════════════════════════════════════════════════════════════════

class SegModel:
    def __init__(self, model_path: str, device: str, half: bool, imgsz: int):
        log.info("Loading model: %s  [device=%s  fp16=%s  imgsz=%d]",
                 model_path, device, half, imgsz)
        self.model  = YOLO(model_path)
        self.device = device
        self.half   = half
        self.imgsz  = imgsz
        self.names: Dict[int, str] = self.model.names
        self._warmup()

    def _warmup(self):
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(dummy, device=self.device, half=self.half,
                               imgsz=self.imgsz, verbose=False)
        except Exception as exc:
            log.warning("Warmup non-fatal: %s", exc)

    def predict_batch(
        self, imgs: List[np.ndarray]
    ) -> List[List[Dict]]:
        """
        Run batch inference.
        Returns per-image list of dicts:
          {"segmentation": [[x,y,...]], "bbox": [x,y,w,h],
           "category_name": str, "score": float}
        """
        per_image: List[List[Dict]] = []
        gen = self.model.predict(
            imgs,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
            verbose=False,
            stream=True,
        )
        for i, r in enumerate(gen):
            h, w = imgs[i].shape[:2]
            dets = []
            if r.masks is not None:
                for j, raw_mask in enumerate(r.masks.xy):
                    # raw_mask: (N_pts, 2) float array
                    flat = raw_mask.flatten().tolist()
                    bbox_xyxy = r.boxes.xyxy[j].cpu().numpy()
                    bx = float(bbox_xyxy[0])
                    by = float(bbox_xyxy[1])
                    bw = float(bbox_xyxy[2] - bbox_xyxy[0])
                    bh = float(bbox_xyxy[3] - bbox_xyxy[1])
                    cls_id = int(r.boxes.cls[j].item())
                    conf   = float(r.boxes.conf[j].item())
                    dets.append({
                        "segmentation":   [flat],
                        "bbox":           [bx, by, bw, bh],
                        "category_name":  r.names.get(cls_id, str(cls_id)),
                        "score":          conf,
                        "_h": h, "_w": w,
                    })
            per_image.append(dets)
        return per_image


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Run inference and save prediction COCO files
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(
    images_dir: Path,
    base_coco: Dict,
    model: SegModel,
    label: str,
    output_dir: Path,
    batch_size: int,
    workers: int,
    resume: bool,
    chunk: int,
) -> Path:
    """
    Run model on all images, match each image to its base COCO image_id by
    file_name stem, and save a COCO-format predictions JSON.

    Returns path to the saved prediction file.
    """
    pred_path = output_dir / f"_predictions_{label}.json"

    # ── Build stem → image_id index from base COCO ────────────────────────
    stem_to_image_id: Dict[str, int] = {}
    for img_entry in base_coco["images"]:
        stem = Path(img_entry["file_name"]).stem
        stem_to_image_id[stem] = img_entry["id"]

    # ── Discover images on disk ───────────────────────────────────────────
    all_images: List[Path] = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
    )
    log.info("[%s] Found %d image files on disk", label, len(all_images))

    # Filter to only images that exist in base COCO
    work_images = [p for p in all_images if p.stem in stem_to_image_id]
    log.info("[%s] %d images match base COCO", label, len(work_images))

    if chunk > 0:
        work_images = work_images[:chunk]
        log.info("[%s] Chunk mode: processing first %d images", label, len(work_images))

    # ── Resume: load existing predictions ────────────────────────────────
    existing_preds: List[Dict] = []
    done_image_ids: set = set()
    if resume and pred_path.exists():
        try:
            existing_data = load_json(pred_path)
            existing_preds = existing_data.get("predictions", [])
            done_image_ids = {p["image_id"] for p in existing_preds}
            log.info("[%s] Resume: %d images already predicted", label, len(done_image_ids))
        except Exception as exc:
            log.warning("[%s] Could not load existing predictions: %s", label, exc)

    # Filter already-done
    todo = [p for p in work_images
            if stem_to_image_id.get(p.stem) not in done_image_ids]
    log.info("[%s] Remaining to infer: %d", label, len(todo))

    all_preds = list(existing_preds)

    # ── Batch inference loop ──────────────────────────────────────────────
    t0 = time.perf_counter()
    processed = 0

    with tqdm(total=len(todo), desc=f"[{label}] inference", unit="img") as pbar:
        for start in range(0, len(todo), batch_size):
            batch_paths = todo[start: start + batch_size]

            # Parallel image load
            imgs: List[Optional[np.ndarray]] = [None] * len(batch_paths)
            with ThreadPoolExecutor(max_workers=min(workers, len(batch_paths))) as ex:
                futs = {ex.submit(load_image, p): i for i, p in enumerate(batch_paths)}
                for f in as_completed(futs):
                    imgs[futs[f]] = f.result()

            valid_idx  = [i for i, im in enumerate(imgs) if im is not None]
            valid_imgs = [imgs[i] for i in valid_idx]

            if not valid_imgs:
                pbar.update(len(batch_paths))
                continue

            try:
                results = model.predict_batch(valid_imgs)
            except Exception as exc:
                log.warning("[%s] Batch inference failed: %s", label, exc)
                pbar.update(len(batch_paths))
                continue

            for k, i in enumerate(valid_idx):
                img_path  = batch_paths[i]
                image_id  = stem_to_image_id[img_path.stem]
                dets      = results[k]
                for det in dets:
                    pred_entry = {
                        "image_id":     image_id,
                        "category_name": det["category_name"],
                        "score":        det["score"],
                        "segmentation": det["segmentation"],
                        "bbox":         det["bbox"],
                        "_h":           det["_h"],
                        "_w":           det["_w"],
                    }
                    all_preds.append(pred_entry)

            processed += len(valid_idx)
            pbar.update(len(batch_paths))

            # Periodic checkpoint every 1000 images
            if processed % 1000 < batch_size:
                _save_predictions(all_preds, pred_path)
                log.debug("[%s] Checkpoint saved (%d images done)", label, processed)

    # Final save
    _save_predictions(all_preds, pred_path)
    elapsed = time.perf_counter() - t0
    log.info("[%s] Inference done: %d predictions in %.1fs (%.1f img/s)",
             label, len(all_preds), elapsed,
             processed / elapsed if elapsed > 0 else 0)
    return pred_path


def _save_predictions(preds: List[Dict], path: Path) -> None:
    save_json_atomic({"predictions": preds}, path)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Build lookup index for fast IoU matching
# ══════════════════════════════════════════════════════════════════════════════

class PredictionIndex:
    """
    Holds predictions indexed by image_id.
    Lazily converts segmentation → masks on first query per image.
    Masks are cached in a size-bounded LRU dict to avoid OOM on lakh+ images.
    """

    def __init__(self, predictions: List[Dict], cache_size: int = 500):
        # image_id → list of prediction dicts
        self._preds: Dict[int, List[Dict]] = defaultdict(list)
        for p in predictions:
            self._preds[p["image_id"]].append(p)
        self._cache: Dict[int, Tuple[np.ndarray, List[str], List[float]]] = {}
        self._cache_order: List[int] = []
        self._cache_size = cache_size
        log.info("PredictionIndex: %d image_ids, %d total predictions",
                 len(self._preds), len(predictions))

    def _get_masks(
        self, image_id: int, h: int, w: int
    ) -> Tuple[np.ndarray, List[str], List[float]]:
        """Return (masks_NHW, category_names, scores)."""
        if image_id in self._cache:
            return self._cache[image_id]

        preds = self._preds.get(image_id, [])
        if not preds:
            result = (np.empty((0, h, w), dtype=np.uint8), [], [])
        else:
            masks = []
            names = []
            scores = []
            for p in preds:
                ph, pw = p.get("_h", h), p.get("_w", w)
                seg = p.get("segmentation")
                if seg:
                    m = seg_to_mask(seg, ph, pw)
                    if ph != h or pw != w:
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    bbox = p.get("bbox")
                    m = bbox_to_mask(bbox, h, w) if bbox else np.zeros((h, w), np.uint8)
                masks.append(m)
                names.append(p["category_name"])
                scores.append(float(p["score"]))
            result = (np.stack(masks, axis=0), names, scores)

        # LRU eviction
        if len(self._cache_order) >= self._cache_size:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        self._cache[image_id] = result
        self._cache_order.append(image_id)
        return result

    def best_match(
        self,
        image_id: int,
        ann_seg: List,
        ann_bbox: List[float],
        img_h: int,
        img_w: int,
        iou_thresh: float,
    ) -> Optional[str]:
        """
        Match an annotation's mask against all predictions for image_id.
        Returns category_name of the best scoring prediction, or None.
        """
        masks, names, scores = self._get_masks(image_id, img_h, img_w)
        if masks.shape[0] == 0:
            return None

        # Build annotation mask
        if ann_seg:
            query_mask = seg_to_mask(ann_seg, img_h, img_w)
        elif ann_bbox:
            query_mask = bbox_to_mask(ann_bbox, img_h, img_w)
        else:
            return None

        if query_mask.sum() == 0:
            return None

        # Vectorised IoU against all predictions
        ious = mask_iou_batch(query_mask, masks)
        scores_arr = np.array(scores, dtype=np.float32)
        weighted = IOU_WEIGHT * ious + CONF_WEIGHT * scores_arr

        # Filter by threshold
        valid = np.where(ious >= iou_thresh)[0]
        if len(valid) == 0:
            return None

        best_idx = valid[np.argmax(weighted[valid])]
        return names[best_idx]


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Attribute update pass
# ══════════════════════════════════════════════════════════════════════════════

def build_image_size_index(base_coco: Dict) -> Dict[int, Tuple[int, int]]:
    """image_id → (height, width)"""
    idx = {}
    for img in base_coco["images"]:
        idx[img["id"]] = (img.get("height", 0), img.get("width", 0))
    return idx


def update_attributes(
    base_coco: Dict,
    mat_index: PredictionIndex,
    col_index: PredictionIndex,
    iou_thresh: float,
    workers: int,
) -> Dict:
    """
    Deep-copy base_coco, update Material + Colour on every annotation.
    Returns the updated COCO dict.
    """
    out_coco = copy.deepcopy(base_coco)
    img_sizes = build_image_size_index(base_coco)

    annotations = out_coco["annotations"]
    total = len(annotations)
    matched_mat = 0
    matched_col = 0

    log.info("Updating attributes on %d annotations ...", total)

    def _update_one(ann: Dict) -> Dict:
        nonlocal matched_mat, matched_col
        image_id = ann["image_id"]
        h, w = img_sizes.get(image_id, (0, 0))

        # Use height/width = 1 as sentinel when unknown (IoU will still work
        # if both masks are proportionally correct)
        if h == 0 or w == 0:
            h, w = 1080, 1920  # safe default

        seg  = ann.get("segmentation", [])
        bbox = ann.get("bbox", [])

        mat_cls = mat_index.best_match(image_id, seg, bbox, h, w, iou_thresh)
        col_cls = col_index.best_match(image_id, seg, bbox, h, w, iou_thresh)

        if "attributes" not in ann or not isinstance(ann["attributes"], dict):
            ann["attributes"] = {}

        ann["attributes"]["Material"] = mat_cls if mat_cls else DEFAULT_MATERIAL
        ann["attributes"]["Colour"]   = col_cls if col_cls else DEFAULT_COLOUR
        return ann

    # Process with progress bar (single-threaded to avoid GIL issues with numpy)
    with tqdm(total=total, desc="Updating attributes", unit="ann") as pbar:
        for i, ann in enumerate(annotations):
            old_mat = ann.get("attributes", {}).get("Material")
            old_col = ann.get("attributes", {}).get("Colour")

            _update_one(ann)

            new_mat = ann["attributes"]["Material"]
            new_col = ann["attributes"]["Colour"]

            if new_mat != DEFAULT_MATERIAL or (old_mat and new_mat != old_mat):
                matched_mat += 1
            if new_col != DEFAULT_COLOUR or (old_col and new_col != old_col):
                matched_col += 1

            if (i + 1) % 50_000 == 0:
                log.info("  ... %d / %d annotations processed", i + 1, total)

            pbar.update(1)

    log.info("  Material matched: %d / %d", matched_mat, total)
    log.info("  Colour matched  : %d / %d", matched_col, total)
    return out_coco


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="COCO Annotation Attribute Updater (Material + Colour)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--base_coco",      required=True, help="Input base COCO JSON file")
    p.add_argument("--output_dir",     required=True, help="Output directory")

    # Inference (required unless --mat_pred and --col_pred are given)
    grp = p.add_argument_group("inference (skip if --mat_pred/--col_pred provided)")
    grp.add_argument("--images",        default=None, help="Images folder for inference")
    grp.add_argument("--material_model",default=None, help="Material model .pt path")
    grp.add_argument("--color_model",   default=None, help="Color model .pt path")

    # Skip-inference mode
    skip = p.add_argument_group("skip inference (provide pre-generated prediction files)")
    skip.add_argument("--mat_pred", default=None, help="Existing material predictions JSON")
    skip.add_argument("--col_pred", default=None, help="Existing color predictions JSON")

    # Runtime
    p.add_argument("--device",     default="0",   help="GPU index or 'cpu'")
    p.add_argument("--batch",      type=int, default=16,  help="Inference batch size")
    p.add_argument("--imgsz",      type=int, default=640, help="Inference image size")
    p.add_argument("--iou_thresh", type=float, default=0.1, help="Min IoU for match")
    p.add_argument("--workers",    type=int, default=4,   help="I/O threads")
    p.add_argument("--resume",     action="store_true",   help="Skip already-inferred images")
    p.add_argument("--chunk",      type=int, default=0,
                   help="Limit to first N images (0=all; for dry runs)")
    p.add_argument("--log_level",  default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--output_name", default="coco_updated.json",
                   help="Output COCO filename")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = build_parser().parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_coco_path = Path(args.base_coco)
    if not base_coco_path.exists():
        log.error("base_coco not found: %s", base_coco_path)
        sys.exit(1)

    # ── Validate output won't overwrite input ─────────────────────────────
    output_path = output_dir / args.output_name
    if output_path.resolve() == base_coco_path.resolve():
        log.error("Output path is the same as input! Use a different --output_dir or --output_name.")
        sys.exit(1)

    log.info("Loading base COCO: %s", base_coco_path)
    base_coco = load_json(base_coco_path)
    log.info("  images=%d  annotations=%d  categories=%d",
             len(base_coco.get("images", [])),
             len(base_coco.get("annotations", [])),
             len(base_coco.get("categories", [])))

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1: Inference (or load existing predictions)
    # ══════════════════════════════════════════════════════════════════════

    if args.mat_pred and args.col_pred:
        # Skip-inference mode
        log.info("Skipping inference — loading existing prediction files")
        mat_pred_path = Path(args.mat_pred)
        col_pred_path = Path(args.col_pred)
        for p in [mat_pred_path, col_pred_path]:
            if not p.exists():
                log.error("Prediction file not found: %s", p)
                sys.exit(1)
        mat_preds = load_json(mat_pred_path)["predictions"]
        col_preds = load_json(col_pred_path)["predictions"]

    else:
        # Inference mode
        if not args.images or not args.material_model or not args.color_model:
            log.error(
                "Provide --images, --material_model, --color_model "
                "OR skip inference with --mat_pred and --col_pred."
            )
            sys.exit(1)

        images_dir     = Path(args.images)
        mat_model_path = Path(args.material_model)
        col_model_path = Path(args.color_model)

        for label, path in [
            ("--images",         images_dir),
            ("--material_model", mat_model_path),
            ("--color_model",    col_model_path),
        ]:
            if not path.exists():
                log.error("%s not found: %s", label, path)
                sys.exit(1)

        device = resolve_device(args.device)
        half   = supports_half(device)
        log.info("Device=%s  FP16=%s  batch=%d  imgsz=%d",
                 device, half, args.batch, args.imgsz)

        log.info("=" * 60)
        log.info("STAGE 1a: Material model inference")
        log.info("=" * 60)
        mat_model = SegModel(str(mat_model_path), device, half, args.imgsz)
        mat_pred_path = run_inference(
            images_dir, base_coco, mat_model, "material",
            output_dir, args.batch, args.workers, args.resume, args.chunk,
        )
        del mat_model  # free GPU memory before loading next model

        log.info("=" * 60)
        log.info("STAGE 1b: Color model inference")
        log.info("=" * 60)
        col_model = SegModel(str(col_model_path), device, half, args.imgsz)
        col_pred_path = run_inference(
            images_dir, base_coco, col_model, "color",
            output_dir, args.batch, args.workers, args.resume, args.chunk,
        )
        del col_model

        mat_preds = load_json(mat_pred_path)["predictions"]
        col_preds = load_json(col_pred_path)["predictions"]

    log.info("Loaded %d material predictions", len(mat_preds))
    log.info("Loaded %d color predictions",    len(col_preds))

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2: Build lookup indexes
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 2: Building prediction indexes")
    log.info("=" * 60)
    mat_index = PredictionIndex(mat_preds)
    col_index = PredictionIndex(col_preds)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 3: Update annotations
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 3: Updating annotation attributes")
    log.info("=" * 60)
    t0 = time.perf_counter()
    updated_coco = update_attributes(
        base_coco, mat_index, col_index, args.iou_thresh, args.workers,
    )
    elapsed = time.perf_counter() - t0

    # ══════════════════════════════════════════════════════════════════════
    # Save output
    # ══════════════════════════════════════════════════════════════════════
    log.info("Saving updated COCO → %s", output_path)
    save_json_atomic(updated_coco, output_path)

    # ── Final summary ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("DONE")
    log.info("  Output file     : %s", output_path)
    log.info("  Images          : %d", len(updated_coco["images"]))
    log.info("  Annotations     : %d", len(updated_coco["annotations"]))
    log.info("  Categories      : %d", len(updated_coco["categories"]))
    log.info("  Attribute update: %.1fs", elapsed)
    log.info("  Input untouched : %s", base_coco_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()


