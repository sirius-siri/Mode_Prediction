import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Load your dataset
data = pd.read_csv('mpdataset.CSV')

# Define selected features and class mapping
selected_features = [
    'EngineOff', 'Outside rH (%)', 'Outside Temp (C)', 'REC_GRADE',
    'Electric Motor Power (kW)', 'Generator Power (kW)',
    'Battery Power (kW)', 'Total System Power (kW)', 'Battery Energy (Wh/s)',
    'SPEED_MPH', 'SCN_RPM', 'SCN_SOC', 'Total Torque Out (Nm)',
    'RPM out (drive shaft)', 'Fuel Consumption (mL/sec)'
]
class_mapping = {'Drive Assist': 0, 'EDO': 1, 'ICE on Regen': 2}

# Data cleaning and preprocessing
data_subset_cleaned = data[selected_features + ['IHF_States']].dropna()
data_subset_cleaned = data_subset_cleaned[data_subset_cleaned['IHF_States'] != 'Full Regen']
data_subset_cleaned = data_subset_cleaned[data_subset_cleaned['IHF_States'] != 'Recirc']
data_subset_cleaned = data_subset_cleaned[data_subset_cleaned['IHF_States'] != 'Special']

X = data_subset_cleaned[selected_features]
y = data_subset_cleaned['IHF_States']

imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Convert class labels to integer labels
y_encoded = y.map(class_mapping)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_imputed, y_encoded, test_size=0.3, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Create a deep neural network model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # 3 classes for 'Drive Assist', 'EDO', and 'ICE on Regen'
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_validation, y_validation))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Set Loss:", test_loss)
print("Test Set Accuracy: {:.2%}".format(test_accuracy))
model.save('mode.h5')

# Classification Report for Test Set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
class_names = list(class_mapping.keys())

classification_rep_test = classification_report(y_test, y_pred_classes, target_names=class_names)
print("Test Set Classification Report:")
print(classification_rep_test)

# Manually test specific rows from the dataset
test_indices = [1, 10, 30, 45, 52]

# Select the rows from the original dataset
test_data = data.iloc[test_indices]

# Preprocess the selected data
test_features = test_data[selected_features]
test_features_imputed = pd.DataFrame(imputer.transform(test_features), columns=test_features.columns)

# Predict the output class for the selected rows using the trained model
predicted_probs = model.predict(test_features_imputed)
predicted_classes = [class_names[np.argmax(prob)] for prob in predicted_probs]

# Print the results
for i, index in enumerate(test_indices):
    actual_class = test_data['IHF_States'].iloc[i]
    predicted_class = predicted_classes[i]

    print(f"Row {index}:")
    print(f"Actual Class: {actual_class}")
    print(f"Predicted Class: {predicted_class}")

    if actual_class == predicted_class:
        print("Desired Output Achieved!")
    else:
        print("Desired Output NOT Achieved!")

    # Print the feature values for this row
    row_features = test_features.iloc[i]
    for feature, value in row_features.iteritems():
        print(f"{feature}: {value}")

    print("\n")

# Plot Accuracy and Loss as Line Plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
