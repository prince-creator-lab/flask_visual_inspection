from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model/casting_inspection_model.h5")

IMG_SIZE = 128

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join("static", file.filename)
            file.save(image_path)
            img_array = preprocess_image(image_path)
            pred = model.predict(img_array)[0][0]
            prediction = "Defective ❌" if pred > 0.5 else "OK ✅"
    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
