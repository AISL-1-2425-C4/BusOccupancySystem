from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# Initialize the YOLO model
model = YOLO('yolov11n.yaml') 


# Train the model
model.train(
    data='data.yaml',       # Dataset YAML file
    epochs=300,                # Number of epochs
    imgsz=640,                # Image size
    batch=16,                 # Batch size
    workers=0,
    cache='disk',
    cos_lr=True,
    box=8,
    dfl=3,
    optimizer='AdamW',
    device=0,
    name='bus_seat_detection_yolo11n', # Experiment name
    project='runs/train801010/tiled/e300',    # Save directory
    patience=30
)

# Evaluate the model
metrics = model.val(split='test', data='data.yaml')
print(metrics)