from ultralytics import YOLO

# Load base YOLOv8 model
model = YOLO("yolov8n.pt")

# Train on your sack dataset
model.train(
    data="sack-2/data.yaml",
    epochs=50,
    imgsz=640,
    device="cpu"   # change to 0 if you have GPU
)

print("Training complete.")