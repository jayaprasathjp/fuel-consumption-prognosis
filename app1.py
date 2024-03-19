from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Loading models
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))


# Creating Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def Predict():
    if request.method=='POST':
        Year = float(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        avg_temp = float(request.form['avg_temp'])
        Area = string(request.form['Area'])
        Item = string(request.form['Item'])

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])

        transformed_features = preprocessor.transform(features)
        predicted_value = dtr.predict(transformed_features).reshape(1, -1)
        return predicted_value[0]

        return render_template('index.html', predicted_value=predicted_value)


# Python main
if __name__ == '__main__':
    app.debug=True
