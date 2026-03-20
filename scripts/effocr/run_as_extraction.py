#!/usr/bin/env python3
"""
Run American Stories YOLO line detection on full-resolution JP2 newspaper scans.

Uses the AS pipeline's YOLOv8 ONNX models for layout detection and line detection,
producing line-level crops and bounding boxes for each page. Optionally runs EffOCR
recognition on each detected line.

Usage:
    source effocr_env/bin/activate
    python scripts/effocr/run_as_extraction.py [--pages N] [--skip-ocr]
"""

import sys
import json
import time
import argparse
import gc
from pathlib import Path
from math import floor, ceil

import numpy as np
import cv2
import onnx
import onnxruntime as ort
import torch
from torchvision.ops import nms
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "effocr"
PILOT_PAGES_JSON = DATA_DIR / "pilot_pages.json"
OUTPUT_DIR = DATA_DIR / "as_extractions"

AS_MODELS_DIR = Path("/Users/nealcaren/Dropbox/american-stories/american_stories_models")
AS_SRC_DIR = Path("/Users/nealcaren/Dropbox/american-stories/AmericanStories/src")

LAYOUT_MODEL_PATH = AS_MODELS_DIR / "layout_model_new.onnx"
LINE_MODEL_PATH = AS_MODELS_DIR / "line_model_new.onnx"
LAYOUT_LABEL_MAP_PATH = AS_SRC_DIR / "label_maps" / "label_map_layout.json"

LAYOUT_TYPES_FOR_LINES = ['article', 'author', 'headline', 'image_caption']


# ---------------------------------------------------------------------------
# YOLO helper functions (extracted from AS pipeline)
# ---------------------------------------------------------------------------

def get_onnx_input_name(model_path):
    """Get the input tensor name for an ONNX model."""
    model = onnx.load(model_path)
    input_all = [node.name for node in model.graph.input]
    input_initializer = [node.name for node in model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    del model
    return net_feed_input[0]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False,
              scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms_yolov8(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300,
               agnostic=False):
    """YOLOv8-format NMS: prediction shape is (batch, num_classes+4, num_boxes)."""
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    bs = prediction.shape[0]
    nc = prediction.shape[1] - 4  # number of classes
    xc = prediction[:, 4:4 + nc].amax(1) > conf_thres

    max_wh = 7680
    max_nms = 30000
    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        x = x.transpose(0, -1)[xc[xi]]
        if not x.shape[0]:
            continue

        box, cls = x[:, :4], x[:, 4:4 + nc]
        box = xywh2xyxy(box)
        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]

    return output


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def run_layout_detection(session, input_name, ca_img, label_map):
    """Run layout detection on a full page image.

    Returns list of (class_label, (x0, y0, x1, y1), PIL_crop) for each region.
    """
    im = letterbox(ca_img, (1280, 1280), auto=False)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.expand_dims(np.ascontiguousarray(im), axis=0).astype(np.float32) / 255.0

    predictions = session.run(None, {input_name: im})
    predictions = torch.from_numpy(predictions[0])
    predictions = nms_yolov8(predictions, conf_thres=0.01, iou_thres=0.1,
                             max_det=2000, agnostic=True)[0]

    bboxes = predictions[:, :4]
    labels = predictions[:, -1]

    layout_img = Image.fromarray(cv2.cvtColor(ca_img, cv2.COLOR_BGR2RGB))
    im_width, im_height = layout_img.size

    # Compute rescaling from 1280x1280 letterboxed coords back to original
    if im_width > im_height:
        w_ratio = 1280
        h_ratio = (im_width / im_height) * 1280
        w_trans = 0
        h_trans = 1280 * ((1 - (im_height / im_width)) / 2)
    else:
        h_trans = 0
        h_ratio = 1280
        w_trans = 1280 * ((1 - (im_width / im_height)) / 2)
        w_ratio = 1280 * (im_width / im_height)

    layout_crops = []
    for i, (bbox, pred_class) in enumerate(zip(bboxes, labels)):
        x0, y0, x1, y1 = torch.round(bbox)
        x0 = int(floor((x0.item() - w_trans) * im_width / w_ratio))
        y0 = int(floor((y0.item() - h_trans) * im_height / h_ratio))
        x1 = int(ceil((x1.item() - w_trans) * im_width / w_ratio))
        y1 = int(ceil((y1.item() - h_trans) * im_height / h_ratio))

        # Clamp to image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(im_width, x1), min(im_height, y1)

        if x1 <= x0 or y1 <= y0:
            continue

        crop = layout_img.crop((x0, y0, x1, y1))
        class_label = label_map.get(int(pred_class.item()), "unknown")
        layout_crops.append((class_label, (x0, y0, x1, y1), crop))

    return layout_crops


# ---------------------------------------------------------------------------
# Line detection
# ---------------------------------------------------------------------------

def get_crops_from_layout_image(image):
    """Chunk tall layout regions into overlapping crops for line detection."""
    im_width, im_height = image.size
    if im_height <= im_width * 2:
        return [image]
    else:
        y0 = 0
        y1 = im_width * 2
        crops = []
        while y1 <= im_height:
            crops.append(image.crop((0, y0, im_width, y1)))
            y0 += int(im_width * 1.5)
            y1 += int(im_width * 1.5)
        crops.append(image.crop((0, y0, im_width, im_height)))
        return crops


def readjust_line_detections(line_preds, orig_img_width):
    """Merge line detections from overlapping crops back to layout coords."""
    y0 = 0
    dif = int(orig_img_width * 1.5)
    all_preds = []

    for preds, probs, _labels in line_preds:
        for i, pred in enumerate(preds):
            all_preds.append((pred[0], pred[1] + y0, pred[2], pred[3] + y0, probs[i]))
        y0 += dif

    all_preds = torch.tensor(all_preds)
    final_preds = []
    if all_preds.dim() > 1 and all_preds.shape[0] > 0:
        keep = nms(all_preds[:, :4], all_preds[:, -1], iou_threshold=0.15)
        filtered = all_preds[keep, :4]
        filtered = filtered[filtered[:, 1].sort()[1]]  # sort by y
        for pred in filtered:
            x0, y0, x1, y1 = torch.round(pred)
            final_preds.append((x0.item(), y0.item(), x1.item(), y1.item()))

    return final_preds


def run_line_detection(session, input_name, layout_crops, layout_label_map=None):
    """Run line detection on layout crops that contain text.

    Returns list of (layout_idx, class_label, layout_bbox, line_bbox_in_page, PIL_line_crop)
    for every detected line.
    """
    all_lines = []

    for layout_idx, (class_label, layout_bbox, layout_crop) in enumerate(layout_crops):
        # Only detect lines in text-bearing regions
        if class_label not in LAYOUT_TYPES_FOR_LINES:
            continue

        im_width, im_height = layout_crop.size
        chunks = get_crops_from_layout_image(layout_crop)

        chunk_preds = []
        for chunk in chunks:
            gc.collect()
            chunk_cv = cv2.cvtColor(np.array(chunk), cv2.COLOR_RGB2BGR)
            im = letterbox(chunk_cv, (640, 640), auto=False)[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = np.expand_dims(np.ascontiguousarray(im), axis=0).astype(np.float32) / 255.0

            preds = session.run(None, {input_name: im})
            preds = torch.from_numpy(preds[0])
            preds = nms_yolov8(preds, conf_thres=0.2, iou_thres=0.15, max_det=200)[0]

            preds = preds[preds[:, 1].sort()[1]]  # sort by y
            line_bboxes = preds[:, :4]
            line_confs = preds[:, -2]
            line_labels = preds[:, -1]

            chunk_w, chunk_h = chunk.size
            if chunk_w > chunk_h:
                h_ratio = (chunk_h / chunk_w) * 640
                h_trans = 640 * ((1 - (chunk_h / chunk_w)) / 2)
            else:
                h_trans = 0
                h_ratio = 640

            line_proj_crops = []
            for bbox in line_bboxes:
                bx0, by0, bx1, by1 = torch.round(bbox)
                # Use full width of the layout crop, only vertical from detection
                lx0 = 0
                ly0 = int(floor((by0.item() - h_trans) * chunk_h / h_ratio))
                lx1 = chunk_w
                ly1 = int(ceil((by1.item() - h_trans) * chunk_h / h_ratio))
                line_proj_crops.append((lx0, ly0, lx1, ly1))

            chunk_preds.append((line_proj_crops, line_confs, line_labels))

        # Readjust overlapping chunk predictions
        line_bboxes_in_layout = readjust_line_detections(chunk_preds, im_width)

        lx0_page, ly0_page = layout_bbox[0], layout_bbox[1]
        for line_bbox in line_bboxes_in_layout:
            x0, y0, x1, y1 = line_bbox
            # Clamp to layout bounds
            x0 = max(0, int(x0))
            y0 = max(0, int(y0))
            x1 = min(im_width, int(x1))
            y1 = min(im_height, int(y1))

            if x1 <= x0 or y1 <= y0:
                continue

            line_crop = layout_crop.crop((x0, y0, x1, y1))
            if line_crop.size[0] == 0 or line_crop.size[1] == 0:
                continue

            # Convert to page-level coordinates
            page_bbox = (lx0_page + x0, ly0_page + y0,
                         lx0_page + x1, ly0_page + y1)

            all_lines.append((layout_idx, class_label, layout_bbox,
                              page_bbox, line_crop))

    return all_lines


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def make_page_id(entry):
    lccn = entry["lccn"]
    date = entry["date"]
    seq = Path(entry["path"]).stem
    return f"{lccn}_{date}_{seq}"


def process_page(entry, layout_session, layout_input_name, layout_label_map,
                 line_session, line_input_name, effocr=None):
    """Process a single page: layout -> lines -> optional OCR."""
    page_id = make_page_id(entry)
    jp2_path = entry["path"]
    page_dir = OUTPUT_DIR / page_id

    # Resume check
    metadata_path = page_dir / "metadata.json"
    if metadata_path.exists():
        print(f"  Already processed, skipping: {page_id}")
        return None

    print(f"\nProcessing: {page_id}")
    print(f"  JP2: {jp2_path}")

    # Load image
    t0 = time.time()
    ca_img = cv2.imread(jp2_path, cv2.IMREAD_COLOR)
    if ca_img is None:
        print(f"  ERROR: Could not read image {jp2_path}")
        return None
    h, w = ca_img.shape[:2]
    print(f"  Image size: {w}x{h}")

    # --- Layout detection ---
    t1 = time.time()
    layout_crops = run_layout_detection(layout_session, layout_input_name,
                                        ca_img, layout_label_map)
    layout_time = time.time() - t1
    print(f"  Layout detection: {len(layout_crops)} regions in {layout_time:.1f}s")

    # Count by class
    class_counts = {}
    for cls, bbox, crop in layout_crops:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    for cls, count in sorted(class_counts.items()):
        print(f"    {cls}: {count}")

    # --- Line detection ---
    t2 = time.time()
    all_lines = run_line_detection(line_session, line_input_name, layout_crops)
    line_time = time.time() - t2
    print(f"  Line detection: {len(all_lines)} lines in {line_time:.1f}s")

    # --- Save outputs ---
    lines_dir = page_dir / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)

    # Save layout visualization
    vis_img = Image.fromarray(cv2.cvtColor(ca_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(vis_img)
    LAYOUT_COLORS = {
        'article': 'blue', 'headline': 'red',
        'cartoon_or_advertisement': 'orange', 'author': 'green',
        'image_caption': 'purple', 'masthead': 'cyan',
        'photograph': 'yellow', 'table': 'brown',
    }
    for cls, bbox, crop in layout_crops:
        color = LAYOUT_COLORS.get(cls, 'white')
        draw.rectangle(bbox, outline=color, width=3)
    vis_img.save(str(page_dir / "layout_viz.jpg"), quality=85)

    # Save line crops and build metadata
    line_records = []
    ocr_texts = []

    for line_idx, (layout_idx, cls, layout_bbox, page_bbox, line_crop) in enumerate(all_lines):
        line_id = f"line_{line_idx:04d}"
        crop_path = lines_dir / f"{line_id}.png"
        line_crop.save(str(crop_path))

        # Draw line bbox on viz
        draw.rectangle(page_bbox, outline='red', width=1)

        record = {
            "line_id": line_id,
            "line_index": line_idx,
            "layout_index": layout_idx,
            "layout_class": cls,
            "layout_bbox": list(layout_bbox),
            "page_bbox": list(page_bbox),
            "crop_path": f"lines/{line_id}.png",
            "crop_size": [line_crop.size[0], line_crop.size[1]],
        }

        # Optional OCR
        if effocr is not None:
            try:
                # Save temp file for EffOCR
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    line_crop.save(tmp.name)
                    results = effocr.infer(tmp.name)
                    Path(tmp.name).unlink(missing_ok=True)

                # Extract text from results
                text_parts = []
                for bbox_result in results:
                    for k in sorted(bbox_result.preds.keys()):
                        line_data = bbox_result.preds[k]
                        word_preds = line_data.get("word_preds", [])
                        final_puncs = line_data.get("final_puncs", [])
                        parts = []
                        for i, word in enumerate(word_preds):
                            w = word if word else ""
                            if i < len(final_puncs) and final_puncs[i]:
                                w += final_puncs[i]
                            parts.append(w)
                        text_parts.append(" ".join(parts))
                ocr_text = " ".join(text_parts)
                record["ocr_text"] = ocr_text
                ocr_texts.append(ocr_text)
            except Exception as e:
                record["ocr_text"] = f"ERROR: {e}"
                ocr_texts.append("")
        else:
            ocr_texts.append("")

        line_records.append(record)

    # Save line+layout viz
    vis_img.save(str(page_dir / "lines_viz.jpg"), quality=85)

    elapsed = time.time() - t0
    metadata = {
        "page_id": page_id,
        "jp2_path": jp2_path,
        "image_size": [w, h],
        "num_layout_regions": len(layout_crops),
        "layout_class_counts": class_counts,
        "num_lines": len(all_lines),
        "layout_time_s": round(layout_time, 2),
        "line_time_s": round(line_time, 2),
        "total_time_s": round(elapsed, 2),
        "lines": line_records,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Output: {page_dir}")

    # Print sample OCR if available
    if any(ocr_texts):
        sample = [t for t in ocr_texts if t.strip()][:5]
        if sample:
            print(f"\n  Sample OCR lines:")
            for t in sample:
                print(f"    {t[:100]}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Run AS line detection on pilot newspaper pages")
    parser.add_argument("--pages", type=int, default=1,
                        help="Number of pages to process (default: 1)")
    parser.add_argument("--skip-ocr", action="store_true",
                        help="Skip EffOCR recognition (just detect lines)")
    args = parser.parse_args()

    # Load pilot pages
    with open(PILOT_PAGES_JSON) as f:
        pilot_pages = json.load(f)

    pages_to_process = pilot_pages[:args.pages]
    print(f"Will process {len(pages_to_process)} page(s)")

    # Load layout model
    print("Loading layout model...")
    layout_input_name = get_onnx_input_name(str(LAYOUT_MODEL_PATH))
    layout_session = ort.InferenceSession(str(LAYOUT_MODEL_PATH))
    with open(LAYOUT_LABEL_MAP_PATH) as f:
        layout_label_map = {int(k): v for k, v in json.load(f).items()}
    print(f"  Layout classes: {layout_label_map}")

    # Load line model
    print("Loading line model...")
    line_input_name = get_onnx_input_name(str(LINE_MODEL_PATH))
    line_session = ort.InferenceSession(str(LINE_MODEL_PATH))

    # Optionally load EffOCR
    effocr = None
    if not args.skip_ocr:
        try:
            print("Loading EffOCR...")
            from efficient_ocr import EffOCR
            hf_repo = "dell-research-harvard/effocr_en"
            effocr = EffOCR(config={
                "Recognizer": {
                    "char": {
                        "model_backend": "onnx",
                        "hf_repo_id": f"{hf_repo}/char_recognizer",
                    },
                    "word": {
                        "model_backend": "onnx",
                        "hf_repo_id": f"{hf_repo}/word_recognizer",
                    },
                },
                "Localizer": {"model_backend": "onnx", "hf_repo_id": hf_repo},
                "Line": {"model_backend": "onnx", "hf_repo_id": hf_repo},
            })
            print("  EffOCR loaded.")
        except Exception as e:
            print(f"  WARNING: Could not load EffOCR, skipping OCR: {e}")
            effocr = None

    # Process pages
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for entry in pages_to_process:
        try:
            process_page(entry, layout_session, layout_input_name,
                         layout_label_map, line_session, line_input_name,
                         effocr=effocr)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
