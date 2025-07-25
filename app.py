from flask import Flask, request, render_template
import torch
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from ultralytics import YOLO



app = Flask(__name__)
model = YOLO("best.pt", map_location=torch.device("cpu"), weights_only=False)
model = model['model']
model.eval()
model.float()

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB").resize((224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1]
    ])
    return transform(image).unsqueeze(0)  

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST' and 'file' in request.files:
        img = request.files['file']
        results = model.predict(source=img, imgsz=640)  # or a size you trained with
        # results is an ultralytics Results object list
        r = results[0]
        prediction = [{'label': cls, 'confidence': float(conf), 'box': box.tolist()}
                      for cls, conf, box in zip(r.names.values(), r.conf, r.boxes.xyxy)]
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port)
