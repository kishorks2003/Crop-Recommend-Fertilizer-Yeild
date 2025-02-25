from flask import Flask, request, render_template
""" Flask Modules:
Flask: Creates the web application.
request: Handles user input from forms.
render_template: Renders HTML pages."""

import numpy as np
#Used for handling numerical data and feature arrays
import pickle
#Loads pre-trained machine learning models stored as .pkl files 
import os
#Helps manage file paths (check if models exist)
import pandas as pd
#Used for reading and processing datasets (csv)

app = Flask(__name__)

# Function to safely load model files
def load_model(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    print(f"Warning: Model file '{filename}' not found!")
    return None

# Load trained models
crop_model = load_model('models/crop_model.pkl')
fertilizer_model = load_model('models/fertilizer_model.pkl')
yield_model = load_model('models/yield_model.pkl')
scaler = load_model('models/scaler.pkl')

# Load dataset for crop and soil types
df = pd.read_csv('/datasets/agriculture_dataset.csv')
crop_types = df['Crop_Type'].unique().tolist()
soil_types = df['Soil_Type'].unique().tolist()
soil_mapping = {soil: idx for idx, soil in enumerate(soil_types)}

# Image directory for crops
IMAGE_FOLDER = "static/crop_images/"

# Home route
@app.route('/')
def home():
    return render_template('index.html', crop_types=crop_types, soil_types=soil_types)

# Prediction Logic
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data with reasonable defaults
        N = float(request.form.get('Nitrogen', 50))  
        P = float(request.form.get('Phosphorus', 50))  
        K = float(request.form.get('Potassium', 50))  
        temp = float(request.form.get('Temperature', 25))  
        humidity = float(request.form.get('Humidity', 50))  
        ph = float(request.form.get('pH', 6.5))  
        rainfall = float(request.form.get('Rainfall', 100))  
        soil_type = request.form.get('Soil_Type', 'Alluvial').strip()
        crop_type = request.form.get('Crop_Type', 'Rice').strip()

        # Validate soil type input
        if soil_type not in soil_mapping:
            return render_template('index.html', error=f"Invalid soil type: {soil_type}", crop_types=crop_types, soil_types=soil_types)

        soil_num = soil_mapping[soil_type]

        # Check if models are loaded
        if not all([crop_model, fertilizer_model, yield_model, scaler]):
            return render_template('index.html', error="One or more models are missing. Please check the 'models/' directory.", crop_types=crop_types, soil_types=soil_types)

        # Prepare features and scale them
        features = np.array([[N, P, K, temp, humidity, ph, rainfall, soil_num]])
        features_scaled = scaler.transform(features)

        # Predict top 3 crops
        crop_probs = crop_model.predict_proba(features_scaled)[0]
        top_3_crops = np.argsort(crop_probs)[-3:][::-1]  
        recommended_crops = [crop_types[crop] if crop < len(crop_types) else "Unknown Crop" for crop in top_3_crops]

        # Get crop images
        crop_images = {}
        for crop in recommended_crops:
            image_path = os.path.join(IMAGE_FOLDER, f"{crop.lower()}.jpg")
            crop_images[crop] = image_path if os.path.exists(image_path) else "static/crop_images/default.jpg"

        # Predict fertilizer
        fertilizer_pred = fertilizer_model.predict(features_scaled)[0]

        # Predict yield
        yield_pred = yield_model.predict(features_scaled)[0]

        return render_template('index.html', 
                               crops=recommended_crops,
                               crop_images=crop_images,
                               fertilizer=fertilizer_pred,
                               yield_prediction=round(yield_pred, 2),
                               crop_types=crop_types,
                               soil_types=soil_types)

    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {str(e)}", crop_types=crop_types, soil_types=soil_types)

if __name__ == "__main__":
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode)
