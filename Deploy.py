import time
import cv2
import numpy as np
import json
import os
import psutil
import requests
import subprocess
import uuid
from datetime import datetime
from ultralytics import YOLO

# ----------------- LOG DIRECTORY ----------------- #
log_dir = "Log"
os.makedirs(log_dir, exist_ok=True)

# ----------------- LOAD ONNX MODEL ----------------- #
onnx_model_path = "/home/thesis/Deployment/best10082025.onnx"
model = YOLO(onnx_model_path)

# ----------------- SYSTEM METRICS ----------------- #
def get_system_metrics(ping_host="8.8.8.8"):
    cpu_usage = psutil.cpu_percent(interval=1)
    temps = psutil.sensors_temperatures()
    cpu_temp = None
    if temps and "coretemp" in temps:
        cpu_temp = temps["coretemp"][0].current
    elif temps:
        first_sensor = list(temps.keys())[0]
        cpu_temp = temps[first_sensor][0].current

    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        ).decode("utf-8").strip()
        gpu_usage = float(gpu_info.split("\n")[0])
    except Exception:
        gpu_usage = None

    try:
        ping_output = subprocess.check_output(
            ["ping", "-c", "1", ping_host], stderr=subprocess.STDOUT
        ).decode("utf-8")
        latency_line = [l for l in ping_output.split("\n") if "time=" in l]
        latency = float(latency_line[0].split("time=")[-1].split(" ")[0]) if latency_line else None
    except Exception:
        latency = None

    return {
        "cpu_usage_percent": cpu_usage,
        "cpu_temp_celsius": cpu_temp,
        "gpu_usage_percent": gpu_usage,
        "network_latency_ms": latency
    }

# ----------------- TILE HANDLING ----------------- #
def tile_image(image_path, output_folder="tiles", overlap_pixels=100):
    img = cv2.imread(image_path)
    img_height, img_width, _ = img.shape
    mid = img_width // 2

    left_tile = img[:, 0:mid + overlap_pixels // 2]
    right_tile = img[:, mid - overlap_pixels // 2:img_width]

    os.makedirs(output_folder, exist_ok=True)
    base_filename = os.path.basename(image_path)
    left_tile_path = os.path.join(output_folder, base_filename.replace(".jpg", "_left.jpg"))
    right_tile_path = os.path.join(output_folder, base_filename.replace(".jpg", "_right.jpg"))

    cv2.imwrite(left_tile_path, left_tile)
    cv2.imwrite(right_tile_path, right_tile)

    print(f"Tiles saved: {left_tile_path}, {right_tile_path}")
    return left_tile_path, right_tile_path, left_tile, right_tile, img_width

# ----------------- RUN INFERENCE ----------------- #
def run_ultralytics_on_tiles(left_tile_path, right_tile_path, overlap_pixels=100, conf=0.1, iou=0.45):
    detections = []
    left_img = cv2.imread(left_tile_path)
    left_width = left_img.shape[1]
    offsets = [0, left_width - overlap_pixels]

    for tile_path, x_offset in zip([left_tile_path, right_tile_path], offsets):
        results = model.predict(source=tile_path, imgsz=640, conf=conf, iou=iou, verbose=False)
        for r in results:
            for box in r.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                conf_score = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                detections.append({
                    "class_id": cls_id,
                    "class_name": model.names[cls_id] if hasattr(model, "names") else f"class_{cls_id}",
                    "confidence": float(conf_score),
                    "x_min": float(x_min + x_offset),
                    "y_min": float(y_min),
                    "x_max": float(x_max + x_offset),
                    "y_max": float(y_max),
                    "image": tile_path
                })
    return detections

# ----------------- IOU CALCULATION ----------------- #
def compute_iou(det1, det2):
    x1 = max(det1["x_min"], det2["x_min"])
    y1 = max(det1["y_min"], det2["y_min"])
    x2 = min(det1["x_max"], det2["x_max"])
    y2 = min(det1["y_max"], det2["y_max"])
    inter_w, inter_h = max(0, x2 - x1), max(0, y2 - y1)
    intersection = inter_w * inter_h
    area1 = (det1["x_max"] - det1["x_min"]) * (det1["y_max"] - det1["y_min"])
    area2 = (det2["x_max"] - det2["x_min"]) * (det2["y_max"] - det2["y_min"])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# ----------------- CONFIDENCE-BASED NMS ----------------- #
def manual_nms(detections, conf_threshold=0.1, iou_threshold=0.2):
    # Filter out low-confidence detections
    detections = [d for d in detections if d["confidence"] >= conf_threshold]
    detections.sort(key=lambda x: x["confidence"], reverse=True)  # Sort by confidence

    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)

        new_detections = []
        for d in detections:
            iou = compute_iou(current, d)

            # 🔸 If boxes overlap heavily
            if iou >= iou_threshold:
                # Keep the one with higher confidence (current is always higher here)
                continue
            else:
                new_detections.append(d)
        detections = new_detections
    return keep

# ----------------- VISUALIZATION ----------------- #
def overlay_detections_on_image(image, detections):
    for det in detections:
        x_min, y_min, x_max, y_max = map(int, [det["x_min"], det["y_min"], det["x_max"], det["y_max"]])
        color = (0, 255, 0) if det["class_id"] == 0 else (0, 0, 255)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    return image

# ----------------- SAFE JSON ENCODER ----------------- #
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ----------------- MAIN ----------------- #
if __name__ == "__main__":
    overlap_pixels = 100
    images_dir = "/home/thesis/Deployment/DataCollection"

    last_processed = None  # 🆕 Track last processed file

    while True:
        try:
            # Get all JPGs in folder
            image_files = [
                os.path.join(images_dir, f)
                for f in os.listdir(images_dir)
                if f.lower().endswith(".jpg")
            ]

            if not image_files:
                print("⚠️ No images found, waiting...")
                time.sleep(5)
                continue

            # ✅ Get the most recent image by modification time
            latest_image = max(image_files, key=os.path.getmtime)

            # 🆕 Skip if this image was already processed
            if latest_image == last_processed:
                print(f"⏳ Waiting for new image... (Last: {os.path.basename(latest_image)})")
                time.sleep(5)
                continue

            last_processed = latest_image
            image_path = latest_image
            base_filename = os.path.basename(image_path)

            # ✅ Extract UUID from filename if pattern matches
            try:
                file_uuid = base_filename.split('-')[-1].replace('.jpg', '')
            except Exception:
                file_uuid = str(uuid.uuid4())

            print(f"🆔 New image detected: {base_filename} | UUID: {file_uuid}")

            start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            start_time = time.time()

            left_tile_path, right_tile_path, left_tile, right_tile, img_width = tile_image(
                image_path, overlap_pixels=overlap_pixels
            )
            detections = run_ultralytics_on_tiles(
                left_tile_path, right_tile_path, overlap_pixels=overlap_pixels
            )
            detections = manual_nms(detections, conf_threshold=0.1, iou_threshold=0.2)

            end_time = time.time()
            inference_time = end_time - start_time
            end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            metrics = get_system_metrics()
            right_tile_cropped = right_tile[:, overlap_pixels:]
            merged_image = np.hstack((left_tile, right_tile_cropped))
            merged_with_dets = overlay_detections_on_image(merged_image, detections)

            # ✅ Save merged image with same UUID
            output_overlay_dir = "/home/thesis/Deployment/Images"
            os.makedirs(output_overlay_dir, exist_ok=True)
            merged_filename = f"Merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}-{file_uuid}.jpg"
            overlay_save_path = os.path.join(output_overlay_dir, merged_filename)
            cv2.imwrite(overlay_save_path, merged_with_dets)
            print(f"✅ Merged image saved at: {overlay_save_path}")

            # ✅ Save JSON with same UUID
            json_dir = "/home/thesis/Deployment/Json"
            os.makedirs(json_dir, exist_ok=True)
            json_filename = f"Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}-{file_uuid}.json"
            json_path = os.path.join(json_dir, json_filename)

            results_json = {
                "uuid": file_uuid,
                "image_name": base_filename,
                "merged_image": merged_filename,
                "start_time": start_timestamp,
                "end_time": end_timestamp,
                "inference_time_sec": float(inference_time),
                "system_metrics": metrics,
                "detection_results": detections
            }

            with open(json_path, "w") as f:
                json.dump(results_json, f, indent=4, cls=NumpyEncoder)
            print(f"✅ JSON saved: {json_path}")

            # ✅ POST to API
            BEARER_TOKEN = "7a450d69-8ef6-4249-87cf-70cf7ce0d621"
            headers = {"Authorization": f"Bearer {BEARER_TOKEN}", "Content-Type": "application/json"}
            url = "https://api-push-2oek.vercel.app/api/v1/push"
            response = requests.post(url, json=results_json, headers=headers)
            print(f"🌐 POST {response.status_code}: {response.text}")

            # ✅ Log
            log_filename = os.path.join("Log", f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}-{file_uuid}.txt")
            with open(log_filename, "w") as log_file:
                log_file.write(f"UUID: {file_uuid}\n")
                log_file.write(f"Start Time: {start_timestamp}\n")
                log_file.write(f"Inference Time (sec): {inference_time:.4f}\n")
                log_file.write(f"CPU Usage (%): {metrics['cpu_usage_percent']}\n")
                log_file.write(f"CPU Temp (°C): {metrics['cpu_temp_celsius']}\n")
                log_file.write(f"Status Code: {response.status_code}\n")
                log_file.write(f"Response: {response.text}\n")
            print(f"🪶 Log saved: {log_filename}")

        except Exception as e:
            print(f"❌ Error: {e}")

        time.sleep(5)
