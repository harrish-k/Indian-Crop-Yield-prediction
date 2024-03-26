from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import pickle

# Load the dataset
df = pd.read_csv('Crop_Production_Statistics.csv')



# Get unique values for dropdown menus
unique_districts = sorted(df['District'].unique().tolist())
unique_crops = sorted(df['Crop'].astype(str).unique().tolist())
unique_seasons = sorted(df['Season'].unique().tolist())

# Encode categorical columns
label_encoders = {}
for column in ['District', 'Crop', 'Season']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Replace NaN and infinity values with the median
median_production = df['Production'].median()
df['Production'].fillna(median_production, inplace=True)
df['Production'].replace([np.inf, -np.inf], median_production, inplace=True)

# Scale the target variable
scaler = StandardScaler()
df['Production'] = scaler.fit_transform(df['Production'].values.reshape(-1, 1))

# # Split data into training and test sets
# X = df[['District', 'Crop', 'Crop_Year', 'Season', 'Area']]
# y = df['Production']
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# model = RandomForestRegressor()
# model.fit(X, y)

# # Save the model to a file
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model, f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', unique_districts=unique_districts, unique_crops=unique_crops, unique_seasons=unique_seasons)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        district = request.form['district']
        crop = request.form['crop']
        crop_year = int(request.form['crop_year'])
        season = request.form['season']
        area = float(request.form['area'])

        # Encode categorical variables
        district_encoded = label_encoders['District'].transform([district])[0]
        crop_encoded = label_encoders['Crop'].transform([crop])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]

        # Make prediction
        prediction_data = [[district_encoded, crop_encoded, crop_year, season_encoded, area]]

        # Load the trained model from the pickle file
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Check if the loaded model is a tuple
        if isinstance(model, tuple):
            model = model[0]  # Extract the model object from the tuple

        # Make prediction using the loaded model
        prediction = model.predict(prediction_data)[0]

        # Inverse transform the prediction
        prediction = round(scaler.inverse_transform(np.array([[prediction]]))[0][0], 2)

        return render_template('index.html', district=district, season=season, crop_name=crop, year=crop_year, my_area=area, prediction=prediction, unique_districts=unique_districts, unique_crops=unique_crops, unique_seasons=unique_seasons)

if __name__ == '__main__':
    app.run(debug=True)
