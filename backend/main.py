import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageOps
import time

app = FastAPI(title="VisionAnalytica YOLO Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models ──────────────────────────────────────────────────────────────
LOCAL_HELMET_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "yolov11m(100epochs).pt")
REMOTE_HELMET_URL = "https://huggingface.co/nnsohamnn/helmet-detection-yolo11/resolve/main/yolov11m%28100epochs%29.pt"

helmet_model_source = LOCAL_HELMET_PATH if os.path.exists(LOCAL_HELMET_PATH) else REMOTE_HELMET_URL
print(f"📥 Loading helmet model from {helmet_model_source}...")

try:
    helmet_model = YOLO(helmet_model_source)
    # yolo11s.pt — better accuracy than yolo11n.pt, still fast enough for real-time
    base_model = YOLO("yolo11s.pt")
    print("✅ Models loaded successfully!")
    print(f"   Base model: yolo11s.pt (COCO)")
    print(f"   Helmet model classes: {helmet_model.names}")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    helmet_model = None
    base_model = None

# ── Configuration ────────────────────────────────────────────────────────────

# Asymmetric confidence: bias toward catching "Without Helmet"
CONF_WITH_HELMET = 0.35       # higher bar to confirm helmet present
CONF_WITHOUT_HELMET = 0.12    # low bar — catch violations aggressively
HELMET_MODEL_CONF = 0.08      # model runs at very low conf; we filter per-class after

# ROI Zooming configuration
ROI_EXPAND_RATIO = 0.60       # expand motorcycle bbox by 60% in all directions for the crop
ROI_TARGET_SIZE = 640         # resize each ROI crop to this size for helmet model
ROI_ABOVE_EXPAND = 1.0        # expand MORE above the motorcycle (riders' heads are above)
BASE_MODEL_IMGSZ = 640        # input size for base model
HELMET_FULL_IMGSZ = 640       # input size for full-image helmet pass


# ── Utility Functions ────────────────────────────────────────────────────────

def calculate_iou(box1, box2):
    """Calculate IoU of two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def apply_nms(detections, iou_threshold=0.45):
    """Non-Maximum Suppression. On ties, prefers 'no_helmet' labels (safety-first)."""
    if not detections:
        return []
    # Sort: no_helmet detections first (at same confidence), then by confidence desc
    sorted_dets = sorted(
        detections,
        key=lambda x: (x.get('is_no_helmet', False), x['confidence']),
        reverse=True,
    )
    keep = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        sorted_dets = [d for d in sorted_dets if calculate_iou(best['box'], d['box']) < iou_threshold]
    return keep


def box_center(box):
    """Return (cx, cy) center of a box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def point_in_box(point, box, margin=0):
    """Check if a point is inside a box (with optional margin expansion)."""
    x, y = point
    return (box[0] - margin) <= x <= (box[2] + margin) and \
           (box[1] - margin) <= y <= (box[3] + margin)


def expand_box(box, img_w, img_h, ratio, ratio_above=None):
    """
    Expand a bounding box by `ratio` in all directions.
    If `ratio_above` is set, use a larger expansion above the box
    (useful for motorcycles — riders' heads extend upward).
    Returns clipped [x1, y1, x2, y2] in integer pixel coords.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    expand_above = ratio_above if ratio_above is not None else ratio

    nx1 = max(0, int(x1 - w * ratio))
    ny1 = max(0, int(y1 - h * expand_above))  # more space above
    nx2 = min(img_w, int(x2 + w * ratio))
    ny2 = min(img_h, int(y2 + h * ratio))
    return [nx1, ny1, nx2, ny2]


def filter_head_detections(raw_detections):
    """
    Apply asymmetric per-class confidence filtering.
    'Without Helmet' → low threshold (catch violations).
    'With Helmet' → higher threshold (reduce false positives like caps).
    """
    filtered = []
    for det in raw_detections:
        is_no_helmet = det.get("is_no_helmet", False)
        min_conf = CONF_WITHOUT_HELMET if is_no_helmet else CONF_WITH_HELMET
        if det["confidence"] >= min_conf:
            filtered.append(det)
    return filtered


def run_helmet_on_region(img_np, roi_box, img_w, img_h, source_tag=""):
    """
    Run the helmet model on a cropped ROI region.
    - Crops img_np to roi_box
    - Runs helmet model at ROI_TARGET_SIZE
    - Maps detections back to original image coordinates
    Returns list of raw head detections in original image coords.
    """
    rx1, ry1, rx2, ry2 = roi_box
    crop = img_np[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return []

    results = helmet_model(crop, conf=HELMET_MODEL_CONF, imgsz=ROI_TARGET_SIZE)[0]

    detections = []
    for h_box in results.boxes:
        hx1, hy1, hx2, hy2 = h_box.xyxy[0].tolist()
        conf = float(h_box.conf[0])
        cls_id = int(h_box.cls[0])
        label = helmet_model.names[cls_id]
        is_no_helmet = cls_id == 1 or "without" in label.lower()

        # Map ROI-local coords → original image coords
        detections.append({
            "box": [hx1 + rx1, hy1 + ry1, hx2 + rx1, hy2 + ry1],
            "confidence": conf,
            "label": label,
            "class_id": cls_id,
            "is_no_helmet": is_no_helmet,
            "source": source_tag,
        })

    return detections


# ── Main Detection Endpoint ─────────────────────────────────────────────────

@app.post("/detect")
async def detect_violations(file: UploadFile = File(...)):
    if not helmet_model or not base_model:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        t_start = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image).convert("RGB")
        img_np = np.array(image)
        h_orig, w_orig = img_np.shape[:2]
        print(f"\n{'='*60}")
        print(f"📸 Processing image: {w_orig}x{h_orig}")

        # ═══════════════════════════════════════════════════════════════
        # STAGE 1: Detect persons & motorcycles with YOLO11s
        #
        # yolo11s.pt is more accurate than yolo11n.pt (higher mAP)
        # while still being fast enough for real-time use.
        # We also detect bicycles (class 1) for completeness.
        # ═══════════════════════════════════════════════════════════════
        base_results = base_model(img_np, conf=0.25, imgsz=BASE_MODEL_IMGSZ)[0]

        persons = []
        motorcycles = []
        for box in base_results.boxes:
            cls = int(box.cls[0])
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            if cls == 0:
                persons.append({"box": coords, "conf": conf, "id": len(persons)})
            elif cls == 3:  # motorcycle
                motorcycles.append({"box": coords, "conf": conf, "id": len(motorcycles)})

        print(f"🔍 Stage 1: {len(persons)} persons, {len(motorcycles)} motorcycles")

        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: Multi-scale Helmet Detection (ROI Zooming + Full Image)
        #
        # PROBLEM: In a typical traffic image (1920x1080), a rider's
        # head might be only 15-25px after resize to 640. The helmet
        # model can miss these tiny heads entirely.
        #
        # SOLUTION: ROI Zooming — for each motorcycle detected, we:
        #   1. Expand the motorcycle bbox by 60% (more above — heads
        #      extend upward from the bike)
        #   2. Crop that expanded region from the original image
        #   3. Resize the crop to 640x640 → the head now occupies
        #      maybe 80-120px instead of 15-25px
        #   4. Run helmet model on this zoomed crop
        #   5. Map detections back to original image coordinates
        #
        # We ALSO run on the full image to catch any heads not near
        # a detected motorcycle, then MERGE all detections with NMS
        # to remove duplicates from overlapping crops.
        # ═══════════════════════════════════════════════════════════════

        all_raw_heads = []

        # Pass A: Full-image detection (catches everything, but lower res on small heads)
        print(f"\n🔬 Stage 2a: Full-image helmet detection at {HELMET_FULL_IMGSZ}px...")
        full_results = helmet_model(img_np, conf=HELMET_MODEL_CONF, imgsz=HELMET_FULL_IMGSZ)[0]
        for h_box in full_results.boxes:
            hx1, hy1, hx2, hy2 = h_box.xyxy[0].tolist()
            conf = float(h_box.conf[0])
            cls_id = int(h_box.cls[0])
            label = helmet_model.names[cls_id]
            is_no_helmet = cls_id == 1 or "without" in label.lower()
            all_raw_heads.append({
                "box": [hx1, hy1, hx2, hy2],
                "confidence": conf,
                "label": label,
                "class_id": cls_id,
                "is_no_helmet": is_no_helmet,
                "source": "full_image",
            })
        print(f"   Found {len(all_raw_heads)} raw detections from full image")

        # Pass B: ROI Zoom on each motorcycle (high-res on the area that matters)
        print(f"🔬 Stage 2b: ROI Zoom on {len(motorcycles)} motorcycle region(s)...")
        for m_idx, moto in enumerate(motorcycles):
            roi = expand_box(
                moto["box"], w_orig, h_orig,
                ratio=ROI_EXPAND_RATIO,
                ratio_above=ROI_ABOVE_EXPAND,  # expand MORE above (riders' heads)
            )

            # Calculate the effective zoom factor
            roi_w = roi[2] - roi[0]
            roi_h = roi[3] - roi[1]
            zoom_x = ROI_TARGET_SIZE / max(roi_w, 1)
            zoom_y = ROI_TARGET_SIZE / max(roi_h, 1)
            print(f"   Motorcycle {m_idx}: ROI {roi} (zoom ~{min(zoom_x,zoom_y):.1f}x)")

            roi_heads = run_helmet_on_region(img_np, roi, w_orig, h_orig, source_tag=f"roi_moto_{m_idx}")
            print(f"   → {len(roi_heads)} detections from ROI crop")
            all_raw_heads.extend(roi_heads)

        # Pass C: ROI Zoom on each person (catches riders on bikes not detected as motorcycles)
        # Only for persons not already covered by a motorcycle ROI
        print(f"🔬 Stage 2c: ROI Zoom on person head regions...")
        for p_idx, person in enumerate(persons):
            px1, py1, px2, py2 = person["box"]
            p_w = px2 - px1
            p_h = py2 - py1

            # Check if this person is already inside a motorcycle ROI
            p_center = box_center(person["box"])
            covered = False
            for moto in motorcycles:
                moto_roi = expand_box(moto["box"], w_orig, h_orig, ratio=ROI_EXPAND_RATIO, ratio_above=ROI_ABOVE_EXPAND)
                if point_in_box(p_center, moto_roi):
                    covered = True
                    break

            if covered:
                continue  # already covered by motorcycle ROI zoom

            # Crop just the upper 45% of the person (head + shoulders)
            head_roi = [
                max(0, int(px1 - p_w * 0.10)),
                max(0, int(py1 - p_h * 0.10)),
                min(w_orig, int(px2 + p_w * 0.10)),
                min(h_orig, int(py1 + p_h * 0.45)),
            ]
            roi_heads = run_helmet_on_region(img_np, head_roi, w_orig, h_orig, source_tag=f"roi_person_{p_idx}")
            all_raw_heads.extend(roi_heads)

        # Filter per-class confidence and apply NMS to merge all sources
        filtered_heads = filter_head_detections(all_raw_heads)
        head_detections = apply_nms(filtered_heads, iou_threshold=0.40)

        print(f"\n🎯 Stage 2 FINAL: {len(head_detections)} head detections after multi-scale merge + NMS")
        for hd in head_detections:
            print(f"   {hd['label']:16s} conf={hd['confidence']:.2f}  src={hd.get('source','-'):16s}  box={[round(c) for c in hd['box']]}")

        # ═══════════════════════════════════════════════════════════════
        # STAGE 3: HYBRID Matching — heads + person fallback
        #
        # PRIMARY: head detection → motorcycle (precise helmet status)
        # FALLBACK: person box → motorcycle (catches riders whose heads
        #           were missed by the helmet model entirely)
        #
        # This is critical: YOLO11s detects persons very reliably, but
        # the helmet model can miss heads in occluded/crowded scenes.
        # Without the fallback, those riders simply don't exist in
        # our system, leading to "0 riders" on occupied motorcycles.
        # ═══════════════════════════════════════════════════════════════

        # Step 3a: Head → Person matching
        head_to_person = {}
        person_helmet_status = {p["id"]: "unknown" for p in persons}

        for h_idx, head in enumerate(head_detections):
            hcx, hcy = box_center(head["box"])
            best_person = -1
            best_dist = float('inf')

            for p_idx, person in enumerate(persons):
                px1, py1, px2, py2 = person["box"]
                p_h = py2 - py1
                # Head should be in the upper 60% of the person box
                upper_person = [px1, py1, px2, py1 + p_h * 0.60]

                margin_x = (px2 - px1) * 0.20
                margin_y = p_h * 0.20
                if point_in_box((hcx, hcy), upper_person, margin=max(margin_x, margin_y)):
                    dist = abs(hcx - (px1+px2)/2) + abs(hcy - (py1+py2)/2)
                    if dist < best_dist:
                        best_dist = dist
                        best_person = p_idx

            if best_person >= 0:
                head_to_person[h_idx] = best_person
                current_status = person_helmet_status[best_person]
                # Safety-first: if ANY detection says no_helmet, mark as no_helmet
                if head["is_no_helmet"]:
                    person_helmet_status[best_person] = "no_helmet"
                elif current_status == "unknown":
                    person_helmet_status[best_person] = "helmet"

        print(f"\n👤 Stage 3a: Head-to-person:")
        for h_idx, p_idx in head_to_person.items():
            print(f"   Head {h_idx} ({head_detections[h_idx]['label']}) → Person {p_idx}")

        # Step 3b: Head → Motorcycle matching (primary rider signal)
        head_to_moto = {}
        riders_per_bike_heads = {}  # moto_idx -> [head_indices]

        for h_idx, head in enumerate(head_detections):
            hcx, hcy = box_center(head["box"])
            h_bottom = head["box"][3]

            best_moto = -1
            best_score = 0

            for m_idx, moto in enumerate(motorcycles):
                mx1, my1, mx2, my2 = moto["box"]
                m_w = mx2 - mx1
                m_h = my2 - my1
                m_cy = (my1 + my2) / 2

                # Head must be horizontally within motorcycle span (±30% tolerance)
                h_margin = m_w * 0.30
                if hcx < mx1 - h_margin or hcx > mx2 + h_margin:
                    continue

                # Head bottom must not be far below motorcycle bottom
                if h_bottom > my2 + m_h * 0.15:
                    continue

                # Head should not be absurdly far above the motorcycle
                h_top = head["box"][1]
                if h_top < my1 - m_h * 2.0:
                    continue

                # Score: proximity-based (closer = better)
                h_dist = abs(hcx - (mx1+mx2)/2) / max(m_w, 1)
                v_dist = abs(hcy - m_cy) / max(m_h, 1)
                score = 1.0 / (1.0 + h_dist + v_dist)

                if score > best_score:
                    best_score = score
                    best_moto = m_idx

            if best_moto >= 0:
                head_to_moto[h_idx] = best_moto
                riders_per_bike_heads.setdefault(best_moto, []).append(h_idx)

        print(f"🏍️  Stage 3b: Head-to-motorcycle (primary):")
        for m_idx, h_ids in riders_per_bike_heads.items():
            labels = [head_detections[h]['label'] for h in h_ids]
            print(f"   Motorcycle {m_idx}: {len(h_ids)} rider(s) via heads = {labels}")

        # Step 3c: FALLBACK — Person → Motorcycle matching
        #
        # For persons who have NO head detection matched to them,
        # check if their body position indicates they're on a motorcycle.
        # This catches riders the helmet model completely missed.
        #
        # Criteria for person-to-motorcycle match:
        #   1. Person's center is within the motorcycle's expanded ROI
        #   2. Person's height is NOT much taller than motorcycle (filters standing people)
        #   3. Person's horizontal center overlaps the motorcycle's horizontal span
        #   4. Person is not already matched via head detection
        persons_with_head = set(head_to_person.values())

        person_bike_assignment = [-1] * len(persons)
        riders_per_bike_fallback = {}  # moto_idx -> [person_indices] (fallback-matched only)

        # First, assign from head-based matching
        for h_idx, m_idx in head_to_moto.items():
            if h_idx in head_to_person:
                p_idx = head_to_person[h_idx]
                person_bike_assignment[p_idx] = m_idx

        # Then, fallback for unmatched persons
        for p_idx, person in enumerate(persons):
            if person_bike_assignment[p_idx] >= 0:
                continue  # already matched via head

            px1, py1, px2, py2 = person["box"]
            p_h = py2 - py1
            p_w = px2 - px1
            p_cx, p_cy = box_center(person["box"])

            best_moto = -1
            best_score = 0

            for m_idx, moto in enumerate(motorcycles):
                mx1, my1, mx2, my2 = moto["box"]
                m_w = mx2 - mx1
                m_h = my2 - my1

                # CHECK 1: Person height should not be far taller than motorcycle
                # A rider appears at most ~1.5x the motorcycle height
                # A standing person next to a bike is usually 1.8-2.5x taller
                if p_h > m_h * 1.5:
                    continue

                # CHECK 2: Person's horizontal center must be within motorcycle
                # span (±30% tolerance)
                h_margin = m_w * 0.30
                if p_cx < mx1 - h_margin or p_cx > mx2 + h_margin:
                    continue

                # CHECK 3: Person's bottom should be near motorcycle's bottom
                # (riders sit on bikes, their bottom is near wheel level)
                bottom_diff = abs(py2 - my2)
                if bottom_diff > m_h * 0.50:
                    continue

                # CHECK 4: Vertical overlap — person should overlap significantly
                # with the motorcycle vertically
                v_overlap = max(0, min(py2, my2) - max(py1, my1))
                v_ratio = v_overlap / max(p_h, 1)
                if v_ratio < 0.30:
                    continue

                # Score: prefer closer match
                h_dist = abs(p_cx - (mx1+mx2)/2) / max(m_w, 1)
                v_dist = abs(p_cy - (my1+my2)/2) / max(m_h, 1)
                score = v_ratio / (1.0 + h_dist + v_dist)

                if score > best_score:
                    best_score = score
                    best_moto = m_idx

            if best_moto >= 0 and best_score > 0.1:
                person_bike_assignment[p_idx] = best_moto
                riders_per_bike_fallback.setdefault(best_moto, []).append(p_idx)
                print(f"   🔄 FALLBACK: Person {p_idx} → Motorcycle {best_moto} (score={best_score:.2f}, helmet=unknown)")

        # Combine: total riders per bike = head-matched + fallback-matched
        riders_per_bike = {}  # final combined count
        for m_idx in range(len(motorcycles)):
            head_riders = riders_per_bike_heads.get(m_idx, [])
            fallback_riders = riders_per_bike_fallback.get(m_idx, [])
            # Get person IDs from head-matched riders
            head_person_ids = set()
            for h_idx in head_riders:
                if h_idx in head_to_person:
                    head_person_ids.add(head_to_person[h_idx])
            # Fallback riders that aren't already counted via heads
            unique_fallback = [p for p in fallback_riders if p not in head_person_ids]
            total_rider_count = len(head_riders) + len(unique_fallback)
            if total_rider_count > 0:
                riders_per_bike[m_idx] = {
                    "head_indices": head_riders,
                    "fallback_person_indices": unique_fallback,
                    "total_count": total_rider_count,
                }

        print(f"\n🏍️  Stage 3 FINAL — Combined rider counts:")
        for m_idx, info in riders_per_bike.items():
            print(f"   Motorcycle {m_idx}: {info['total_count']} total riders "
                  f"({len(info['head_indices'])} via heads + {len(info['fallback_person_indices'])} via fallback)")

        # ═══════════════════════════════════════════════════════════════
        # STAGE 4: Violation Assembly
        # ═══════════════════════════════════════════════════════════════
        violations = []

        # Violation 1: No helmet on motorcycle rider (from head detections)
        for h_idx, m_idx in head_to_moto.items():
            head = head_detections[h_idx]
            if head["is_no_helmet"]:
                p_idx = head_to_person.get(h_idx)
                violations.append({
                    "type": "no_helmet",
                    "severity": "high",
                    "description": "Rider on motorcycle without helmet",
                    "person_box": persons[p_idx]["box"] if p_idx is not None else head["box"],
                    "motorcycle_box": motorcycles[m_idx]["box"],
                    "person_id": p_idx,
                    "motorcycle_id": m_idx,
                })

        # Violation 1b: Fallback riders with unknown helmet status
        # (no head detected = we can't confirm a helmet → flag as warning)
        for m_idx, info in riders_per_bike.items():
            for p_idx in info["fallback_person_indices"]:
                if person_helmet_status.get(p_idx) == "unknown":
                    violations.append({
                        "type": "no_helmet",
                        "severity": "medium",
                        "description": "Rider on motorcycle — helmet not detected (possible violation)",
                        "person_box": persons[p_idx]["box"],
                        "motorcycle_box": motorcycles[m_idx]["box"],
                        "person_id": p_idx,
                        "motorcycle_id": m_idx,
                    })

        # Violation 2: Triple riding (>2 riders per motorcycle)
        for m_idx, info in riders_per_bike.items():
            if info["total_count"] > 2:
                # Gather all rider person IDs from both sources
                all_rider_pids = []
                for h_idx in info["head_indices"]:
                    if h_idx in head_to_person:
                        all_rider_pids.append(head_to_person[h_idx])
                all_rider_pids.extend(info["fallback_person_indices"])

                violations.append({
                    "type": "triple_riding",
                    "severity": "high",
                    "description": f"{info['total_count']} persons detected on one motorcycle (max allowed: 2)",
                    "rider_count": info["total_count"],
                    "person_boxes": [persons[pid]["box"] for pid in all_rider_pids],
                    "motorcycle_box": motorcycles[m_idx]["box"],
                    "person_ids": all_rider_pids,
                    "motorcycle_id": m_idx,
                })

        t_elapsed = time.time() - t_start
        print(f"\n🚨 Stage 4: {len(violations)} violations detected")
        for v in violations:
            print(f"   {v['type']}({v['severity']}): {v['description']}")
        print(f"⏱️  Total processing time: {t_elapsed:.2f}s")

        # ── Build Response ───────────────────────────────────────────
        response_persons = []
        for p_idx, person in enumerate(persons):
            response_persons.append({
                "id": p_idx,
                "box": person["box"],
                "confidence": person["conf"],
                "helmet_status": person_helmet_status.get(p_idx, "unknown"),
                "on_motorcycle": person_bike_assignment[p_idx] >= 0,
                "motorcycle_id": person_bike_assignment[p_idx] if person_bike_assignment[p_idx] >= 0 else None,
            })

        response_motorcycles = []
        for m_idx, moto in enumerate(motorcycles):
            info = riders_per_bike.get(m_idx, {"head_indices": [], "fallback_person_indices": [], "total_count": 0})
            # Collect all rider person IDs from both head and fallback matching
            all_rider_pids = []
            for h_idx in info["head_indices"]:
                if h_idx in head_to_person:
                    all_rider_pids.append(head_to_person[h_idx])
            all_rider_pids.extend(info["fallback_person_indices"])

            response_motorcycles.append({
                "id": m_idx,
                "box": moto["box"],
                "confidence": moto["conf"],
                "rider_count": info["total_count"],
                "rider_ids": all_rider_pids,
            })

        frontend_detections = []
        for h_idx, head in enumerate(head_detections):
            frontend_detections.append({
                "box": head["box"],
                "confidence": head["confidence"],
                "label": head["label"],
                "class_id": head["class_id"],
                "person_id": head_to_person.get(h_idx),
            })

        return {
            "detections": frontend_detections,
            "persons": response_persons,
            "motorcycles": response_motorcycles,
            "violations": violations,
            "image_size": {
                "width": w_orig,
                "height": h_orig,
            },
            "processing_time_ms": round(t_elapsed * 1000),
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
