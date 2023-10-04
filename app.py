from flask import Flask, jsonify, request
import cv2
import pytesseract
import numpy as np  
import tensorflow as tf 
import requests
from io import BytesIO

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = 'pytesseract/tesseract.exe'

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="image_classifier_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image_from_url(image_url):
    response = requests.get(image_url)
    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.float32(img)
    img = np.expand_dims(img, axis=0)
    return img

def predictByImageFromURL(image_url):
    # Preprocess the input image
    input_image = preprocess_image_from_url(image_url)
    class_labels = ["Aadhaar", "PAN", "Driving Licence", "Voter ID", "Passport", "Utility", "credit_cards"]

    # Run the inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    if predicted_class == "Aadhaar":
        response = requests.get(image_url)
        img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(image_rgb)
        return jsonify({"predicted_class": predicted_class, "data": text})
    else:
        return jsonify({"predicted_class": "Not an Aadhar", "data": "unavailable"})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        image_url = data.get('image_url')
        return predictByImageFromURL(image_url)
    elif request.method == 'GET':
        image_url = request.args.get('image_url')
        return predictByImageFromURL(image_url)
    else:
        return jsonify({"error": "Invalid request method"}), 405


if __name__ == '__main__':
    app.run(debug=True)
