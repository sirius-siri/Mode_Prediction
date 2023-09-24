import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Load the saved model
loaded_model = keras.models.load_model('mode.h5')

# Define class mapping
class_mapping = {0: 'Drive Assist', 1: 'EDO', 2: 'ICE on Regen'}

# Define a dictionary with the values for a single row
single_row_data = {
    'EngineOff': 1.0,
    'Outside rH (%)': 31.97,
    'Outside Temp (C)': 5.411,
    'REC_GRADE': -3.3,
    'Electric Motor Power (kW)': 4.6358,
    'Generator Power (kW)': 0.0,
    'Battery Power (kW)': 4.6358,
    'Total System Power (kW)': 4.6358,
    'Battery Energy (Wh/s)': 1.287722222,
    'SPEED_MPH': 33.1,
    'SCN_RPM': 0.0,
    'SCN_SOC': 63.5,
    'Total Torque Out (Nm)': 11.25,
    'RPM out (drive shaft)': 1588.923,
    'Fuel Consumption (mL/sec)': 0.07949361
}

# Create a DataFrame from the single row data
single_row_df = pd.DataFrame([single_row_data])

# Make predictions for the single row
predictions = loaded_model.predict(single_row_df)

# Get the predicted class
predicted_class_index = np.argmax(predictions)
predicted_class = class_mapping[predicted_class_index]

# Print the predicted class
print(f"Predicted Class: {predicted_class}")
