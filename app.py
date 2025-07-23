from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model("best.pt")

def preprocess_image(image_file):
    image=Image.open(image_file).resize((224,224))
    image=np.Array(image)/255.0
    return np.expand_dims(image,axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = preprocess_image(file)
            pred = model.predict (img)
            prediction = np.argmax(pred)
    
    return render_template('index.html', prediction= prediction)


if __name__ == '__main__':
    app.run(debug=True)