from ultralytics import YOLO

# Load your trained YOLO model (.pt file)
model = YOLO("best.pt")

# Export to ONNX
model.export(format="onnx", imgsz=640)
