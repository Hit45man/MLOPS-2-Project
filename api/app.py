# api/app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/registered_model.pkl")

# Species mapping
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_web', methods=['POST'])
def predict_web():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    species_name = species_map[prediction]
    
    return render_template('index.html', prediction=prediction, species_name=species_name)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
