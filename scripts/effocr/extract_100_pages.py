#!/usr/bin/env python3
"""
extract_100_pages.py — Run AS YOLO layout + line detection on 100 sampled pages.

For each page, detects layout regions, picks the 1-2 largest article/headline
regions by pixel area, and extracts line crops for those regions only.

Usage:
    source effocr_env/bin/activate
    python scripts/effocr/extract_100_pages.py
"""

import sys
import json
import time
import gc
from pathlib import Path
from math import floor, ceil

import numpy as np
import cv2
import onnx
import onnxruntime as ort
import torch
from torchvision.ops import nms
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "effocr"
SAMPLE_JSON = DATA_DIR / "100page_sample.json"
OUTPUT_DIR = DATA_DIR / "100page_extractions"

AS_MODELS_DIR = Path("/Users/nealcaren/Dropbox/american-stories/american_stories_models")
AS_SRC_DIR = Path("/Users/nealcaren/Dropbox/american-stories/AmericanStories/src")

LAYOUT_MODEL_PATH = AS_MODELS_DIR / "layout_model_new.onnx"
LINE_MODEL_PATH = AS_MODELS_DIR / "line_model_new.onnx"
LAYOUT_LABEL_MAP_PATH = AS_SRC_DIR / "label_maps" / "label_map_layout.json"

# Region types eligible for line extraction
TEXT_REGION_TYPES = {'article', 'headline', 'author', 'image_caption'}
# Region types to skip entirely
SKIP_REGION_TYPES = {'cartoon_or_advertisement', 'photograph', 'table'}
# Max article regions to pick per page
MAX_REGIONS_PER_PAGE = 2


# ---------------------------------------------------------------------------
# YOLO helpers (from run_as_extraction.py)
# ---------------------------------------------------------------------------

def get_onnx_input_name(model_path):
    model = onnx.load(model_path)
    input_all = [node.name for node in model.graph.input]
    input_initializer = [node.name for node in model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    del model
    return net_feed_input[0]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False,
              scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
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
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms_yolov8(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300,
               agnostic=False):
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    bs = prediction.shape[0]
    nc = prediction.shape[1] - 4
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
    im = letterbox(ca_img, (1280, 1280), auto=False)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.expand_dims(np.ascontiguousarray(im), axis=0).astype(np.float32) / 255.0

    predictions = session.run(None, {input_name: im})
    predictions = torch.from_numpy(predictions[0])
    predictions = nms_yolov8(predictions, conf_thres=0.01, iou_thres=0.1,
                             max_det=2000, agnostic=True)[0]

    bboxes = predictions[:, :4]
    labels = predictions[:, -1]

    layout_img = Image.fromarray(cv2.cvtColor(ca_img, cv2.COLOR_BGR2RGB))
    im_width, im_height = layout_img.size

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
    y0 = 0
    dif = int(orig_img_width * 1.5)
    all_preds = []
    for preds, probs, _labels in line_preds:
        for i, pred in enumerate(preds):
            all_preds.append((pred[0], pred[1] + y0, pred[2], pred[3] + y0, probs[i].item()))
        y0 += dif
    all_preds = torch.tensor(all_preds)
    final_preds = []
    if all_preds.dim() > 1 and all_preds.shape[0] > 0:
        keep = nms(all_preds[:, :4], all_preds[:, -1], iou_threshold=0.15)
        filtered = all_preds[keep, :4]
        filtered = filtered[filtered[:, 1].sort()[1]]
        for pred in filtered:
            px0, py0, px1, py1 = torch.round(pred)
            final_preds.append((px0.item(), py0.item(), px1.item(), py1.item()))
    return final_preds


def run_line_detection_for_region(session, input_name, layout_crop, layout_bbox):
    """Run line detection on a single layout region. Returns list of
    (page_bbox, line_crop) tuples."""
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
        preds = preds[preds[:, 1].sort()[1]]

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
            lx0 = 0
            ly0 = int(floor((by0.item() - h_trans) * chunk_h / h_ratio))
            lx1 = chunk_w
            ly1 = int(ceil((by1.item() - h_trans) * chunk_h / h_ratio))
            line_proj_crops.append((lx0, ly0, lx1, ly1))

        chunk_preds.append((line_proj_crops, line_confs, line_labels))

    line_bboxes_in_layout = readjust_line_detections(chunk_preds, im_width)

    lx0_page, ly0_page = layout_bbox[0], layout_bbox[1]
    results = []
    for line_bbox in line_bboxes_in_layout:
        x0, y0, x1, y1 = line_bbox
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(im_width, int(x1))
        y1 = min(im_height, int(y1))
        if x1 <= x0 or y1 <= y0:
            continue
        line_crop = layout_crop.crop((x0, y0, x1, y1))
        if line_crop.size[0] == 0 or line_crop.size[1] == 0:
            continue
        page_bbox = (lx0_page + x0, ly0_page + y0,
                     lx0_page + x1, ly0_page + y1)
        results.append((page_bbox, line_crop))

    return results


# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------

def select_top_regions(layout_crops, max_regions=MAX_REGIONS_PER_PAGE):
    """Pick the largest text-bearing regions by pixel area, skipping
    non-text types (ads, photos, tables)."""
    candidates = []
    for idx, (class_label, bbox, crop) in enumerate(layout_crops):
        if class_label in SKIP_REGION_TYPES:
            continue
        if class_label not in TEXT_REGION_TYPES:
            continue
        x0, y0, x1, y1 = bbox
        area = (x1 - x0) * (y1 - y0)
        candidates.append((area, idx, class_label, bbox, crop))

    # Sort by area descending, take top N
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [(idx, cls, bbox, crop) for _area, idx, cls, bbox, crop
            in candidates[:max_regions]]


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def make_page_id(entry):
    lccn = entry["lccn"]
    date = entry["date"]
    seq = Path(entry["path"]).stem
    return f"{lccn}_{date}_{seq}"


def process_page(entry, split, layout_session, layout_input_name,
                 layout_label_map, line_session, line_input_name):
    """Process a single page: layout -> select regions -> extract lines."""
    page_id = make_page_id(entry)
    page_dir = OUTPUT_DIR / split / page_id
    metadata_path = page_dir / "metadata.json"

    # Resume check
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        return meta.get("num_lines", 0)

    jp2_path = entry["path"]
    t0 = time.time()
    ca_img = cv2.imread(jp2_path, cv2.IMREAD_COLOR)
    if ca_img is None:
        print(f"    ERROR: Could not read {jp2_path}")
        return 0
    h, w = ca_img.shape[:2]

    # Layout detection
    layout_crops = run_layout_detection(layout_session, layout_input_name,
                                        ca_img, layout_label_map)

    # Select top regions
    selected_regions = select_top_regions(layout_crops)
    if not selected_regions:
        print(f"    No text regions found, skipping")
        return 0

    # Line detection on selected regions only
    lines_dir = page_dir / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)

    line_records = []
    region_records = []
    line_idx = 0

    for region_idx, cls, bbox, crop in selected_regions:
        lines = run_line_detection_for_region(
            line_session, line_input_name, crop, bbox)

        region_records.append({
            "layout_index": region_idx,
            "class": cls,
            "bbox": list(bbox),
            "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            "num_lines": len(lines),
        })

        for page_bbox, line_crop in lines:
            line_id = f"line_{line_idx:04d}"
            crop_path = lines_dir / f"{line_id}.png"
            line_crop.save(str(crop_path))
            line_records.append({
                "line_id": line_id,
                "line_index": line_idx,
                "layout_class": cls,
                "layout_bbox": list(bbox),
                "page_bbox": list(page_bbox),
                "crop_size": [line_crop.size[0], line_crop.size[1]],
            })
            line_idx += 1

    elapsed = time.time() - t0

    metadata = {
        "page_id": page_id,
        "split": split,
        "jp2_path": jp2_path,
        "lccn": entry["lccn"],
        "date": entry["date"],
        "year": entry["year"],
        "decade": entry["decade"],
        "image_size": [w, h],
        "num_layout_regions": len(layout_crops),
        "num_selected_regions": len(selected_regions),
        "selected_regions": region_records,
        "num_lines": len(line_records),
        "total_time_s": round(elapsed, 2),
        "lines": line_records,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return len(line_records)


def main():
    # Load sample
    with open(SAMPLE_JSON) as f:
        splits = json.load(f)

    total_pages = sum(len(v) for v in splits.values())
    print(f"Loaded {total_pages} pages from {SAMPLE_JSON}")
    for split_name, pages in splits.items():
        print(f"  {split_name}: {len(pages)} pages")

    # Load models
    print("\nLoading layout model...")
    layout_input_name = get_onnx_input_name(str(LAYOUT_MODEL_PATH))
    layout_session = ort.InferenceSession(str(LAYOUT_MODEL_PATH))
    with open(LAYOUT_LABEL_MAP_PATH) as f:
        layout_label_map = {int(k): v for k, v in json.load(f).items()}

    print("Loading line model...")
    line_input_name = get_onnx_input_name(str(LINE_MODEL_PATH))
    line_session = ort.InferenceSession(str(LINE_MODEL_PATH))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process all pages
    split_line_counts = {}
    processed = 0

    for split_name in ["train", "val", "test"]:
        pages = splits[split_name]
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_lines = 0
        print(f"\n{'='*60}")
        print(f"Processing {split_name} ({len(pages)} pages)")
        print(f"{'='*60}")

        for i, entry in enumerate(pages):
            page_id = make_page_id(entry)
            print(f"  [{processed+1}/{total_pages}] {page_id}", end="", flush=True)

            n_lines = process_page(
                entry, split_name,
                layout_session, layout_input_name, layout_label_map,
                line_session, line_input_name)

            split_lines += n_lines
            processed += 1
            print(f" -> {n_lines} lines")

        split_line_counts[split_name] = split_lines
        print(f"  {split_name} total: {split_lines} lines")

    # Summary
    total_lines = sum(split_line_counts.values())
    print(f"\n{'='*60}")
    print(f"DONE: {processed} pages, {total_lines} total lines")
    print(f"{'='*60}")
    for split_name, count in split_line_counts.items():
        print(f"  {split_name}: {count} lines")


if __name__ == "__main__":
    main()
