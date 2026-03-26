import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf


def load_prices(prices_path: str = "config/prices.json") -> Dict[str, float]:
    with open(prices_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_class_mapping(classes_path: str = "config/classes.json") -> Dict[int, str]:
    with open(classes_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_trained_model(model_path: str = "models/snack_classifier.keras") -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def predict_snack(
    img_bgr: np.ndarray,
    model: tf.keras.Model,
    class_mapping: Dict[int, str],
    img_size: Tuple[int, int] = (224, 224),
) -> Tuple[str, float, float]:
    """
    Predict snack class, confidence, and decision margin for a single cropped image.

    img_size must match the size used during training (224x224 by default).
    The model contains mobilenet_v2.preprocess_input which normalises pixels
    from [0, 255] to [-1, 1] internally, so raw uint8 values must be passed in
    (do NOT divide by 255 here).

    Returns:
        label       – predicted class name
        confidence  – softmax probability of the top class
        margin      – gap between the top and second-highest softmax probability;
                      a low margin means the model is uncertain between two classes
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype("float32"), axis=0)
    preds = model.predict(batch, verbose=0)[0]
    sorted_preds = np.sort(preds)[::-1]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    margin = float(sorted_preds[0] - sorted_preds[1]) if len(sorted_preds) > 1 else confidence
    label = class_mapping.get(idx, "unknown")
    return label, confidence, margin


def _watershed_split(
    blob_mask: np.ndarray,
    min_area: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """
    Split a binary mask that may contain multiple touching objects into
    separate bounding boxes using the distance-transform watershed algorithm.

    HOW IT WORKS:
    1. Distance transform: each foreground pixel gets a value equal to its
       distance to the nearest background pixel.  Object centres score highest.
    2. Threshold at 40 % of the peak distance → "sure foreground" seeds.
       If two objects touch, their centres still form two distinct peaks, so
       connectedComponents finds two seed regions even when the outer contours merge.
    3. Watershed expands each seed until it hits the boundary between the two
       objects (the saddle point of the distance map).

    offset_x / offset_y: top-left position of blob_mask in the full image,
        used to return boxes in full-image coordinates.

    Returns a list of (x, y, w, h) boxes, or [] when no valid split is found
    (caller should fall back to the original bounding rect in that case).
    """
    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []
    dist_norm = dist / dist.max()

    _, sure_fg = cv2.threshold(dist_norm, 0.4, 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    if num_labels <= 1:
        return []

    # Background → label 1, objects → labels 2..N; unknown boundary → 0.
    markers = markers + 1
    sure_bg = cv2.dilate(blob_mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers[unknown == 255] = 0

    img_3ch = cv2.merge([blob_mask, blob_mask, blob_mask])
    cv2.watershed(img_3ch, markers)

    boxes: List[Tuple[int, int, int, int]] = []
    for lbl in range(2, num_labels + 2):
        region = np.uint8(markers == lbl) * 255
        if cv2.countNonZero(region) == 0:
            continue
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw * bh >= min_area:
                boxes.append((offset_x + bx, offset_y + by, bw, bh))

    return boxes if len(boxes) >= 2 else []


def simple_region_proposals(
    img_bgr: np.ndarray,
    min_area_fraction: float = 0.005,
    max_aspect_ratio: float = 2.0,
) -> List[Tuple[int, int, int, int]]:
    """
    Color-saturation segmentation with per-hue-bin bounding boxes.

    WHY not edge-based: Canny + dilation merges physically overlapping packets
    into a single blob, losing the individual packet boundaries.  Two packets
    of DIFFERENT colours that physically overlap still occupy entirely separate
    hue bins in HSV space, so colour-based segmentation can isolate each one.

    Algorithm:
    1. Convert to HSV.  Build a foreground mask: keep only pixels whose
       saturation exceeds a threshold (snack packaging is colourful; plain
       table surfaces have near-zero saturation and are excluded).
    2. Divide the 0-179° hue axis into 12 bins of 15° each.  For each bin,
       intersect the bin mask with the foreground mask, then dilate so that
       sparsely-coloured pixels belonging to the same packet fuse into one
       contiguous blob.  Finer bins keep snacks with adjacent hues (e.g.
       red-orange vs orange-yellow) in separate bins, preventing them from
       merging.  Suspiciously large blobs (>10× the minimum snack area) are
       sent to _watershed_split() to recover individual snacks that were
       fused because they were placed close together.
    3. Find bounding rects, filter by area and aspect ratio, and pad by 5 %.

    Fallback: if saturation-based foreground is empty (very low-saturation or
    greyscale image), fall back to edge detection so the pipeline still works.

    min_area_fraction: minimum blob area as a fraction of total image pixels.
    max_aspect_ratio: discard extremely elongated blobs (thin strips / bars).
    """
    img_h, img_w = img_bgr.shape[:2]
    img_area = img_h * img_w
    min_area = int(min_area_fraction * img_area)
    max_area = int(0.60 * img_area)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # Pixels with saturation > 50 belong to the snack packets; the plain
    # table/background has very low saturation and is excluded.
    fg_mask = (saturation > 50).astype(np.uint8) * 255

    # Dilation kernel: 2.5 % of the shorter image dimension, 2 iterations.
    # This fuses scattered pixels of the same colour on one packet into one
    # solid blob while keeping separately-placed packets of the same colour
    # distinct (the gap between physically separated packets exceeds the
    # dilation radius).
    dilation_px = max(20, int(0.025 * min(img_h, img_w)))
    dil_kernel = np.ones((dilation_px, dilation_px), np.uint8)

    # Blobs larger than this fraction of the image are suspicious (likely merged
    # snacks) and will be sent to watershed for splitting before falling back to
    # a plain bounding rect.  10× the minimum snack area is a safe threshold that
    # won't falsely split a single large snack photographed up close.
    watershed_threshold = min_area * 10

    # Each side of a valid snack box must be at least 4 % of the shorter image
    # dimension.  This rejects thin strips, small texture patches, and other
    # background fragments that pass the area filter only because they are very
    # elongated in one direction.
    min_side_px = max(30, int(0.04 * min(img_h, img_w)))

    def _boxes_from_mask(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        result: List[Tuple[int, int, int, int]] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < min_area or area > max_area:
                continue
            if w < min_side_px or h < min_side_px:
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > max_aspect_ratio:
                continue

            # For suspiciously large blobs, attempt watershed splitting first.
            # This recovers individual snacks that were fused by dilation when
            # placed close together (same-hue snacks side by side).
            if area > watershed_threshold:
                blob_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(blob_mask, [cnt], -1, 255, cv2.FILLED)
                sub_boxes = _watershed_split(
                    blob_mask[y : y + h, x : x + w], min_area,
                    offset_x=x, offset_y=y,
                )
                if sub_boxes:
                    for sx, sy, sw, sh in sub_boxes:
                        # Watershed sub-boxes must pass the same aspect-ratio
                        # guard as ordinary blobs (they bypass the earlier check).
                        sub_asp = max(sw, sh) / max(min(sw, sh), 1)
                        if sub_asp > max_aspect_ratio:
                            continue
                        px, py = int(0.05 * sw), int(0.05 * sh)
                        sx = max(0, sx - px)
                        sy = max(0, sy - py)
                        sw = min(img_w - sx, sw + 2 * px)
                        sh = min(img_h - sy, sh + 2 * py)
                        result.append((sx, sy, sw, sh))
                    continue  # split succeeded — skip the merged bounding rect

            pad_x = int(0.05 * w)
            pad_y = int(0.05 * h)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(img_w - x, w + 2 * pad_x)
            h = min(img_h - y, h + 2 * pad_y)
            result.append((x, y, w, h))
        return result

    hue = hsv[:, :, 0]  # 0-179 in OpenCV
    # 12 bins of 15° each (was 6 × 30°).  Finer quantization keeps snacks with
    # adjacent but distinct hues (e.g. red-orange vs orange-yellow) in separate
    # bins, preventing them from merging during dilation.
    hue_bin = (hue // 15).astype(np.uint8)  # 12 bins: 0-11

    boxes: List[Tuple[int, int, int, int]] = []
    for bin_id in range(12):
        bin_mask = ((hue_bin == bin_id).astype(np.uint8) * 255)
        # Keep only foreground pixels in this hue bin, then dilate.
        combined = cv2.bitwise_and(fg_mask, bin_mask)
        if cv2.countNonZero(combined) == 0:
            continue
        dilated = cv2.dilate(combined, dil_kernel, iterations=2)
        boxes.extend(_boxes_from_mask(dilated))

    # Fallback for very low-saturation or near-greyscale images.
    if not boxes:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        dilated = cv2.dilate(edges, dil_kernel, iterations=2)
        boxes.extend(_boxes_from_mask(dilated))

    return boxes


def _nms(
    detections: List[Dict[str, Any]],
    containment_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Cross-class Non-Maximum Suppression using bidirectional containment ratio.

    Standard IoU-based NMS fails when a small box is fully inside a large one
    because the union is dominated by the large box, keeping IoU artificially low.
    Instead we compute:

        containment = intersection_area / area_of_the_smaller_box

    A candidate is suppressed when it overlaps a previously kept (higher-confidence)
    box such that the intersection covers more than `containment_threshold` of
    *either* box's area. This bidirectional check handles the case where two
    similarly-sized boxes heavily overlap but neither fully contains the other.

    Detections must be sorted by descending confidence before calling this function.
    """
    kept: List[Dict[str, Any]] = []

    for det in detections:
        bx = det["bbox"]
        x1, y1 = bx["x"], bx["y"]
        x2, y2 = x1 + bx["w"], y1 + bx["h"]
        area = bx["w"] * bx["h"]

        suppressed = False
        for keeper in kept:
            kb = keeper["bbox"]
            kx1, ky1 = kb["x"], kb["y"]
            kx2, ky2 = kx1 + kb["w"], ky1 + kb["h"]
            k_area = kb["w"] * kb["h"]

            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h

            smaller_area = min(area, k_area)
            # Suppress if the intersection covers a large fraction of either box.
            if inter_area / smaller_area > containment_threshold:
                suppressed = True
                break

        if not suppressed:
            kept.append(det)

    return kept


def _same_label_nms(
    detections: List[Dict[str, Any]],
    proximity_fraction: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Suppress duplicate detections of the SAME label that are spatially close.

    A single physical snack (e.g. a cylindrical can) can generate multiple
    region proposals from different visible faces or colour regions.  Those
    proposals may be vertically/horizontally separated by a gap, so standard
    containment NMS never fires on them.

    Each bounding box is expanded by `proximity_fraction` × its own width /
    height in all four directions.  Two same-label detections whose EXPANDED
    boxes overlap are treated as detecting the same physical object; the
    lower-confidence one is suppressed.

    Detections must already be sorted by descending confidence so the most
    representative (highest-confidence) box is always the keeper.

    NOTE: two physically separate snacks of the same type will be counted as
    one if the gap between them is smaller than proximity_fraction × snack size.
    Leave a visible gap between same-type snacks when photographing.
    """
    kept: List[Dict[str, Any]] = []

    for det in detections:
        b = det["bbox"]
        px = int(proximity_fraction * b["w"])
        py = int(proximity_fraction * b["h"])
        ex1, ey1 = b["x"] - px, b["y"] - py
        ex2, ey2 = b["x"] + b["w"] + px, b["y"] + b["h"] + py

        suppressed = False
        for keeper in kept:
            if keeper["label"] != det["label"]:
                continue  # only deduplicate same-label pairs
            kb = keeper["bbox"]
            kpx = int(proximity_fraction * kb["w"])
            kpy = int(proximity_fraction * kb["h"])
            kex1, key1 = kb["x"] - kpx, kb["y"] - kpy
            kex2, key2 = kb["x"] + kb["w"] + kpx, kb["y"] + kb["h"] + kpy

            x_overlap = min(ex2, kex2) > max(ex1, kex1)
            y_overlap = min(ey2, key2) > max(ey1, key1)
            if x_overlap and y_overlap:
                suppressed = True
                break

        if not suppressed:
            kept.append(det)

    return kept


def analyze_image(
    img_bgr: np.ndarray,
    model: tf.keras.Model,
    class_mapping: Dict[int, str],
    prices: Dict[str, float],
    confidence_threshold: float = 0.7,
    margin_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    post_nms_containment_threshold: float = 0.30,
    same_label_proximity_fraction: float = 0.50,
) -> Dict[str, Any]:
    """
    Full pipeline — three-stage approach:

    Stage 1  Geometric deduplication with IoU-based NMS (no classification):
        All region proposals are sorted by area descending and deduplicated
        using Intersection-over-Union (IoU).  Near-duplicate proposals from
        adjacent hue bins (IoU > 0.5) are collapsed into one; genuinely
        separate snacks — even when one sits inside another's bounding box —
        have very low IoU and are kept.  (Containment-based NMS was the previous
        approach but it incorrectly suppressed small snacks whose bounding box
        fell inside a large bag's box, since containment=100 % even though the
        IoU is only 5–10 %.)

    Stage 2  Classify surviving boxes:
        Only the geometrically non-redundant boxes are sent to the model.
        Confidence and margin thresholds are applied here.

    Stage 3  Post-classification NMS:
        After classification, detections are sorted by confidence descending
        and run through containment-based NMS a second time.  Stage 1 uses a
        permissive threshold (nms_containment_threshold) so that adjacent
        snacks with touching borders are not suppressed; Stage 3 uses a tighter
        threshold (post_nms_containment_threshold) to remove sub-region crops
        that slipped through Stage 1 because their bounding box happened to
        barely miss the larger box.  A sub-region crop typically covers 30–50 %
        of its own area inside the full-packet box; two genuinely separate snacks
        side by side typically share only 5–25 % of the smaller box's area.

    Stage 4  Same-label proximity deduplication:
        A single physical snack (e.g. a cylindrical can) can produce multiple
        region proposals from its different visible faces.  Those proposals may
        be vertically or horizontally separated by a gap, so Stage 3 containment
        NMS never fires on them.  Each same-label detection pair is tested after
        expanding both boxes by same_label_proximity_fraction × their own size;
        if the expanded boxes overlap, the lower-confidence duplicate is removed.

    margin_threshold: minimum gap between the top and second-best softmax
        probability.  A low margin means the model is uncertain between two
        classes (typical for partial/fragment crops) and the detection is
        discarded.
    """
    raw_boxes = simple_region_proposals(img_bgr)

    # --- Stage 1: geometry-only NMS ------------------------------------------
    # Sort by area descending so larger whole-packet boxes take priority.
    raw_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

    kept_boxes: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in raw_boxes:
        area = w * h
        suppressed = False
        for (kx, ky, kw, kh) in kept_boxes:
            k_area = kw * kh
            ix1, iy1 = max(x, kx), max(y, ky)
            ix2, iy2 = min(x + w, kx + kw), min(y + h, ky + kh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = area + k_area - inter
            if union > 0 and inter / union > nms_iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept_boxes.append((x, y, w, h))

    # --- Stage 2: classify surviving boxes -----------------------------------
    detections: List[Dict[str, Any]] = []
    for (x, y, w, h) in kept_boxes:
        crop = img_bgr[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        label, conf, margin = predict_snack(crop, model, class_mapping)
        if conf < confidence_threshold:
            continue
        if margin < margin_threshold:
            continue
        detections.append(
            {
                "label": label,
                "confidence": conf,
                "margin": margin,
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            }
        )

    # --- Stage 3: post-classification NMS ------------------------------------
    # Sort by confidence so the most confident (usually full-packet) detection
    # wins when a smaller sub-region box overlaps with it.  A sub-region crop
    # that survived Stage 1 because its bounding box barely missed the larger
    # box is suppressed here.
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    detections = _nms(detections, containment_threshold=post_nms_containment_threshold)

    # --- Stage 4: same-label proximity deduplication -------------------------
    # Suppress duplicate detections of the same class that are spatially close.
    # A cylindrical can or similarly shaped package can generate separate region
    # proposals for its top and bottom faces; these have a vertical gap so Stage
    # 3 misses them.  Expanding each box by 50 % of its own size before the
    # overlap test bridges that gap without affecting different-label pairs.
    detections = _same_label_nms(
        detections, proximity_fraction=same_label_proximity_fraction
    )

    counts: Counter = Counter(d["label"] for d in detections)

    breakdown: Dict[str, Any] = {}
    total_price_rm: float = 0.0

    for label, count in counts.items():
        unit_price = float(prices.get(label, 0.0))
        subtotal = unit_price * count
        total_price_rm += subtotal
        breakdown[label] = {
            "count": int(count),
            "unit_price_rm": unit_price,
            "subtotal_rm": subtotal,
        }

    return {
        "detections": detections,
        "breakdown": breakdown,
        "total_price_rm": total_price_rm,
    }


def analyze_image_file(
    image_path: str,
    model_path: str = "models/snack_classifier.keras",
    classes_path: str = "config/classes.json",
    prices_path: str = "config/prices.json",
    confidence_threshold: float = 0.7,
    margin_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    post_nms_containment_threshold: float = 0.30,
    same_label_proximity_fraction: float = 0.50,
) -> Dict[str, Any]:
    """
    Convenience function to run the entire pipeline on an image path.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    model = load_trained_model(model_path)
    class_mapping = load_class_mapping(classes_path)
    prices = load_prices(prices_path)

    return analyze_image(
        img_bgr=img_bgr,
        model=model,
        class_mapping=class_mapping,
        prices=prices,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        nms_iou_threshold=nms_iou_threshold,
        post_nms_containment_threshold=post_nms_containment_threshold,
        same_label_proximity_fraction=same_label_proximity_fraction,
    )


def visualize_detections(
    img_bgr: np.ndarray,
    result: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 9),
) -> None:
    """
    Display the image with bounding boxes and labels drawn over each detected snack region.
    Each unique snack class gets a distinct colour. The figure title shows the total item
    count and price.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    palette = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
        "#fabed4", "#469990",
    ]
    label_color: Dict[str, str] = {}
    color_idx = 0

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img_rgb)

    for det in result["detections"]:
        label = det["label"]
        conf = det["confidence"]
        x, y, w, h = det["bbox"]["x"], det["bbox"]["y"], det["bbox"]["w"], det["bbox"]["h"]

        if label not in label_color:
            label_color[label] = palette[color_idx % len(palette)]
            color_idx += 1
        color = label_color[label]

        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x + 4, y + 4,
            f"{label}  {conf:.0%}",
            color="white",
            fontsize=9,
            fontweight="bold",
            va="top",
            bbox=dict(facecolor=color, alpha=0.75, pad=2, linewidth=0),
        )

    legend_handles = [
        mpatches.Patch(color=c, label=lbl)
        for lbl, c in label_color.items()
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
                  framealpha=0.85, title="Detected snacks")

    n = len(result["detections"])
    total = result["total_price_rm"]
    ax.set_title(
        f"{n} detection{'s' if n != 1 else ''} found  |  Total: RM {total:.2f}",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()

