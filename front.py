import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the saved model
loaded_model = keras.models.load_model('E:\Mini--Project\ModePrediction\mode.h5')

# Define class mapping
class_mapping = {0: 'Drive Assist', 1: 'EDO', 2: 'ICE on Regen'}

@app.route('/', methods=['GET', 'POST'])
def predict():
    predicted_class = None

    if request.method == 'POST':
        # Get input data from the form
        single_row_data = {
            'EngineOff': float(request.form['EngineOff']),
            'Outside rH (%)': float(request.form['Outside_rH']),
            'Outside Temp (C)': float(request.form['Outside_Temp']),
            'REC_GRADE': float(request.form['REC_GRADE']),
            'Electric Motor Power (kW)': float(request.form['Electric_Motor_Power']),
            'Generator Power (kW)': float(request.form['Generator_Power']),
            'Battery Power (kW)': float(request.form['Battery_Power']),
            'Total System Power (kW)': float(request.form['Total_System_Power']),
            'Battery Energy (Wh/s)': float(request.form['Battery_Energy']),
            'SPEED_MPH': float(request.form['SPEED_MPH']),
            'SCN_RPM': float(request.form['SCN_RPM']),
            'SCN_SOC': float(request.form['SCN_SOC']),
            'Total Torque Out (Nm)': float(request.form['Total_Torque_Out']),
            'RPM out (drive shaft)': float(request.form['RPM_out']),
            'Fuel Consumption (mL/sec)': float(request.form['Fuel_Consumption'])
        }

        # Create a DataFrame from the input data
        single_row_df = pd.DataFrame([single_row_data])

        # Make predictions for the single row
        predictions = loaded_model.predict(single_row_df)

        # Get the predicted class
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_mapping[predicted_class_index]

    # If it's a GET request or after prediction, render the HTML form with the result
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Drive Mode</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f5f5f5;
                }
                .container {
                    background-color: #fff;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
                    padding: 20px;
                    margin: 20px auto;
                    max-width: 600px;
                }
                form {
                    display: grid;
                    grid-gap: 10px;
                }
                label {
                    font-weight: bold;
                }
                input[type="text"] {
                    width: 80%;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }
                input[type="submit"] {
                    background-color: #007BFF;
                    color: #fff;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Drive Mode</h1>
                <form method="POST">
                    <!-- Add input fields for each feature in single_row_data -->
                    <label for="EngineOff">EngineOff:</label>
                    <input type="text" name="EngineOff">
                    <label for="Outside_rH">Outside rH (%):</label>
                    <input type="text" name="Outside_rH">
                    <label for="Outside_Temp">Outside Temp (C):</label>
                    <input type="text" name="Outside_Temp">
                    <label for="REC_GRADE">REC_GRADE:</label>
                    <input type="text" name="REC_GRADE">
                    <label for="Electric_Motor_Power">Electric Motor Power (kW):</label>
                    <input type="text" name="Electric_Motor_Power">
                    <label for="Generator_Power">Generator Power (kW):</label>
                    <input type="text" name="Generator_Power">
                    <label for="Battery_Power">Battery Power (kW):</label>
                    <input type="text" name="Battery_Power">
                    <label for="Total_System_Power">Total System Power (kW):</label>
                    <input type="text" name="Total_System_Power">
                    <label for="Battery_Energy">Battery Energy (Wh/s):</label>
                    <input type="text" name="Battery_Energy">
                    <label for="SPEED_MPH">SPEED_MPH:</label>
                    <input type="text" name="SPEED_MPH">
                    <label for="SCN_RPM">SCN_RPM:</label>
                    <input type="text" name="SCN_RPM">
                    <label for="SCN_SOC">SCN_SOC:</label>
                    <input type="text" name="SCN_SOC">
                    <label for="Total_Torque_Out">Total Torque Out (Nm):</label>
                    <input type="text" name="Total_Torque_Out">
                    <label for="RPM_out">RPM out (drive shaft):</label>
                    <input type="text" name="RPM_out">
                    <label for="Fuel_Consumption">Fuel Consumption (mL/sec):</label>
                    <input type="text" name="Fuel_Consumption">
                    <input type="submit" value="Predict">
                </form>
                {% if predicted_class %}
                <h2>Drive Mode</h2>
                <p>{{ predicted_class }}</p>
                {% endif %}
            </div>
        </body>
        </html>
    ''', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
