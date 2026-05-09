import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

_BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = Path(os.path.join(_BASE, "models"))

CIGARETTE_MODEL_PATH = MODELS_DIR / "cigarette_model.pt"
SMOKE_MODEL_PATH     = MODELS_DIR / "smoke_model.pt"
POSE_MODEL_PATH      = MODELS_DIR / "yolo_pose.pt"
HPE_CLASSIFIER_PATH  = MODELS_DIR / "hpe_classifier.pt"

CIG_CONF_THRESHOLD   = 0.1
FINAL_CONF_THRESHOLD = 0.2
SMOKE_IOU_BOOST_THRESHOLD = 0.1
SMOKE_BOOST_FACTOR   = 0.3
VL_CHECK_INTERVAL    = 30
SCALE_FACTOR         = 2.0
TEMPORAL_WINDOW      = 30
N_JOINTS             = 9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_cigarette_model = None
_smoke_model     = None
_pose_model      = None
_hpe_classifier  = None
_vl_model        = None
_vl_processor    = None


def _load_cigarette_model():
    global _cigarette_model
    if _cigarette_model is None:
        _cigarette_model = YOLO(str(CIGARETTE_MODEL_PATH))
    return _cigarette_model


def _load_smoke_model():
    global _smoke_model
    if _smoke_model is None:
        _smoke_model = YOLO(str(SMOKE_MODEL_PATH))
    return _smoke_model


def _load_pose_model():
    global _pose_model
    if _pose_model is None:
        _pose_model = YOLO(str(POSE_MODEL_PATH))
    return _pose_model


def _load_hpe_classifier():
    global _hpe_classifier
    if _hpe_classifier is None:
        _hpe_classifier = torch.load(str(HPE_CLASSIFIER_PATH), map_location=DEVICE)
        _hpe_classifier.eval()
    return _hpe_classifier


def _load_vl_model():
    global _vl_model, _vl_processor
    if _vl_model is None:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        _vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=DEVICE
        )
        _vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    return _vl_model, _vl_processor


def _split_frame_into_quadrants(frame):
    h, w = frame.shape[:2]
    half_h, half_w = h // 2, w // 2
    quadrants = [
        frame[0:half_h, 0:half_w],
        frame[0:half_h, half_w:w],
        frame[half_h:h, 0:half_w],
        frame[half_h:h, half_w:w],
    ]
    offsets = [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]
    return quadrants, offsets, (half_w, half_h)


def _scale_detections_to_original(detections, offset, scale_factor):
    if detections is None or len(detections) == 0:
        return []
    scaled_boxes = []
    for det in detections:
        boxes = det.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1_orig = (x1 / scale_factor) + offset[0]
                y1_orig = (y1 / scale_factor) + offset[1]
                x2_orig = (x2 / scale_factor) + offset[0]
                y2_orig = (y2 / scale_factor) + offset[1]
                conf = float(box.conf[0].cpu().numpy())
                cls  = int(box.cls[0].cpu().numpy())
                scaled_boxes.append({'box': [x1_orig, y1_orig, x2_orig, y2_orig], 'conf': conf, 'cls': cls})
    return scaled_boxes


def _compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2); yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2); yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def _compute_distance(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2; cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2; cy2 = (box2[1] + box2[3]) / 2
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def _verify_smoking_with_vl(frame, boxes):
    if not boxes:
        return []
    try:
        from qwen_vl_utils import process_vision_info
        from PIL import Image
        vl_model, vl_processor = _load_vl_model()
        results = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                results.append(True)
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "Is this person smoking a cigarette or vape? Be careful not to confuse phones or bottles with cigarettes. Answer only Yes or No."}
            ]}]
            text = vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = vl_processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = vl_model.generate(**inputs, max_new_tokens=10)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output_text = vl_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip().lower()
            results.append("yes" in output_text)
        return results
    except Exception as e:
        print(f"VL verification error: {e}")
        return [True] * len(boxes)


class DetectionTracker:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3, distance_threshold=50):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append({
                    'id': self.next_id, 'box': det['box'], 'conf': det['conf'],
                    'cls': det['cls'], 'age': 0, 'hits': 1, 'conf_history': [det['conf']]
                })
                self.next_id += 1
            return self.tracks

        if len(detections) == 0:
            for track in self.tracks:
                track['age'] += 1
            self.tracks = [t for t in self.tracks if t['age'] < self.max_age]
            return self.tracks

        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou = _compute_iou(track['box'], det['box'])
                dist = _compute_distance(track['box'], det['box'])
                if dist < self.distance_threshold:
                    iou_matrix[i, j] = iou

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_tracks = set(); matched_dets = set()

        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] > self.iou_threshold:
                self.tracks[i]['box'] = detections[j]['box']
                self.tracks[i]['conf'] = detections[j]['conf']
                self.tracks[i]['cls'] = detections[j]['cls']
                self.tracks[i]['age'] = 0
                self.tracks[i]['hits'] += 1
                self.tracks[i]['conf_history'].append(detections[j]['conf'])
                if len(self.tracks[i]['conf_history']) > 10:
                    self.tracks[i]['conf_history'].pop(0)
                matched_tracks.add(i); matched_dets.add(j)

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track['age'] += 1

        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.tracks.append({
                    'id': self.next_id, 'box': det['box'], 'conf': det['conf'],
                    'cls': det['cls'], 'age': 0, 'hits': 1, 'conf_history': [det['conf']]
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]
        return [t for t in self.tracks if t['hits'] >= self.min_hits]


def detect_frame(frame: np.ndarray, tracker: DetectionTracker, frame_count: int, verified_tracks: dict) -> list:
    cigarette_model = _load_cigarette_model()
    smoke_model     = _load_smoke_model()

    quadrants, offsets, _ = _split_frame_into_quadrants(frame)
    all_cig_dets   = []
    all_smoke_dets = []

    for quadrant, offset in zip(quadrants, offsets):
        upscaled = cv2.resize(quadrant, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
        cig_results   = cigarette_model(upscaled, conf=CIG_CONF_THRESHOLD, verbose=False)
        smoke_results = smoke_model(upscaled, conf=CIG_CONF_THRESHOLD, verbose=False)
        all_cig_dets.extend(_scale_detections_to_original(cig_results, offset, SCALE_FACTOR))
        all_smoke_dets.extend(_scale_detections_to_original(smoke_results, offset, SCALE_FACTOR))

    merged = []
    for cig_det in all_cig_dets:
        max_smoke_iou = 0.0
        for smoke_det in all_smoke_dets:
            iou = _compute_iou(cig_det['box'], smoke_det['box'])
            if iou > SMOKE_IOU_BOOST_THRESHOLD:
                max_smoke_iou = max(max_smoke_iou, iou)
        has_smoke = max_smoke_iou > 0
        boost = 1.0 + (SMOKE_BOOST_FACTOR * max_smoke_iou) if has_smoke else 1.0
        merged.append({
            'box': cig_det['box'],
            'conf': min(cig_det['conf'] * boost, 1.0),
            'cls': cig_det['cls'],
            'has_smoke': has_smoke
        })

    tracked = tracker.update(merged)
    for t in tracked:
        if 'has_smoke' not in t:
            t['has_smoke'] = False

    if frame_count % VL_CHECK_INTERVAL == 0 and tracked:
        boxes_to_verify = [t['box'] for t in tracked if t['id'] not in verified_tracks]
        if boxes_to_verify:
            vl_results = _verify_smoking_with_vl(frame, boxes_to_verify)
            idx = 0
            for t in tracked:
                if t['id'] not in verified_tracks:
                    verified_tracks[t['id']] = vl_results[idx]
                    idx += 1

    final = []
    for t in tracked:
        is_verified = verified_tracks.get(t['id'], True)
        if not is_verified:
            continue
        avg_conf = float(np.mean(t['conf_history']))
        temporal_boost = min(t['hits'] / 10.0, 0.2)
        final_conf = min(avg_conf + temporal_boost, 1.0)
        if final_conf >= FINAL_CONF_THRESHOLD:
            final.append({
                'box': [int(v) for v in t['box']],
                'conf': final_conf,
                'has_smoke': t['has_smoke'],
                'track_id': t['id']
            })
    return final


def detect_smoking(b64_image: str) -> Optional[str]:
    import base64
    try:
        img_data = base64.b64decode(b64_image)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        tracker = DetectionTracker()
        verified_tracks = {}
        detections = detect_frame(frame, tracker, 0, verified_tracks)
        return "Yes" if detections else "No"
    except Exception as e:
        print(f"detect_smoking error: {e}")
        return None


def extract_hls_url_from_page(url: str) -> Optional[str]:
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    hls_urls = []

    for script in soup.find_all('script'):
        if script.string:
            found_urls = re.findall(r'https?://[^\s"\'_]+\.m3u8', script.string)
            hls_urls.extend(found_urls)

    for video_tag in soup.find_all('video'):
        if 'src' in video_tag.attrs and '.m3u8' in video_tag['src']:
            hls_urls.append(video_tag['src'])
        for source_tag in video_tag.find_all('source'):
            if 'src' in source_tag.attrs and '.m3u8' in source_tag['src']:
                hls_urls.append(source_tag['src'])

    for tag in soup.find_all(True):
        for attr, value in tag.attrs.items():
            if isinstance(value, str) and '.m3u8' in value:
                if value.startswith('http://') or value.startswith('https://'):
                    hls_urls.append(value)

    if hls_urls:
        return hls_urls[0]
    return None