from flask import Flask, request, render_template
from PIL import Image
import os
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("best.pt")
model.fuse()  

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file:
            try:
                image = Image.open(file).convert("RGB")
                results = model.predict(source=image, imgsz=640)

                r = results[0]
                prediction = [{
                    'label': r.names[int(cls)],
                    'confidence': float(conf),
                    'box': box.xyxy.tolist()
                } for cls, conf, box in zip(r.boxes.cls, r.boxes.conf, r.boxes)]

            except Exception as e:
                prediction = [{'label': 'Error', 'confidence': 0.0, 'box': str(e)}]
    label = prediction[0]['label'] if prediction else "Pest not detected."
    return render_template('index.html', prediction=f'This is a {label}.')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
