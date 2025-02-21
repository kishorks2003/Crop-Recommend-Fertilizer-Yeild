import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Load dataset
df = pd.read_csv('datasets/agriculture_dataset.csv')

# Encoding Soil Type
soil_mapping = {
    "Alkaline": 0, "Laterite": 1, "Saline": 2, "Arid": 3, 
    "Black": 4, "Peaty": 5, "Alluvial": 6, "Red": 7, "Forest": 8
                }
df['Soil_Type'] = df['Soil_Type'].map(soil_mapping)


# Features and targets
X = df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall', 'Soil_Type']]
y_crop = df['Crop_Type']
y_fertilizer = df['Recommended Fertilizer']
y_yield = df['Yield (kg/ha)']

# Split all target variables at once
X_train, X_test, y_crop_train, y_crop_test, y_fertilizer_train, y_fertilizer_test, y_yield_train, y_yield_test = train_test_split(
    X, y_crop, y_fertilizer, y_yield, test_size=0.2, random_state=42)


# Scaling features (mean 0, variance 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_train_scaled, y_crop_train)

fertilizer_model = RandomForestClassifier(n_estimators=100, random_state=42)
fertilizer_model.fit(X_train_scaled, y_fertilizer_train)

yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
yield_model.fit(X_train_scaled, y_yield_train)

# Save models
pickle.dump(crop_model, open('models/crop_model.pkl', 'wb'))
pickle.dump(fertilizer_model, open('models/fertilizer_model.pkl', 'wb'))
pickle.dump(yield_model, open('models/yield_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

print("Models saved successfully!")
