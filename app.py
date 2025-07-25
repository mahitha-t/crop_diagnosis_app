from flask import Flask, request, render_template
import torch
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from ultralytics import YOLO

app = Flask(__name__)

# Correct loading of the YOLOv8 model
model = YOLO("best.pt")
model.fuse()  # Optional: optimizes model for inference

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB").resize((224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST' and 'file' in request.files:
        img = request.files['file']
        results = model.predict(source=img, imgsz=640)
        r = results[0]
        prediction = [{
            'label': r.names[int(cls)],
            'confidence': float(conf),
            'box': box.xyxy.tolist()
        } for cls, conf, box in zip(r.boxes.cls, r.boxes.conf, r.boxes)]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
