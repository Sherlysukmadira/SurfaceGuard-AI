from ultralytics import YOLO

# Load trained model
model = YOLO("runs/classify/train-2/weights/best.pt")

# Predict image
results = model.predict(
    source="test.jpg",
    imgsz=256
)

# Get prediction
probs = results[0].probs

# Class names
names = results[0].names

# Top prediction
top1 = probs.top1
confidence = probs.top1conf.item()

print("Predicted Class:", names[top1])
print("Confidence:", confidence)