import numpy as np
import pickle
import joblib  
import flask
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,models
app = Flask(__name__)


model = joblib.load('mnist_model.pkl')  

@app.route("/")
def index():
    return render_template("drawing.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    image_data = base64.b64decode(data.split(",")[1])

    # Convert image to grayscale
    image = Image.open(io.BytesIO(image_data)).convert("L")
    image = image.resize((28, 28))  # Resize to MNIST size
    image = np.array(image)

    
    image = image / 255.0
    image = image.reshape(1, 28,28)  

    prediction = model.predict(image)
    predicted_value=np.argmax(prediction[0])
    return jsonify({"prediction": int(predicted_value)})

if __name__ == "__main__":
    app.run(debug=True)  
