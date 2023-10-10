# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset into a pandas DataFrame
data = pd.read_csv('Supplementary 2.csv')  # Replace 'your_dataset.csv' with the actual filename

# Preprocess the data (you may need to handle missing values and text data)
# For this example, we'll drop rows with missing values
data.dropna(inplace=True)

# Define the features (X) and the target variable (y)
# Assuming 'ASSOCIATION_CLASS' is a categorical column
X = pd.get_dummies(data[['YEAR', 'WEIGHT','GENE','REF_GENE','REF_SENTENCE','TITLE','DIS_CLASS','CONCLUSION']])

# Encode the target variable using label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['ASSOCIATION_CLASS'])  # Convert string labels to numerical labels

# Determine the number of unique classes for output layer
num_classes = len(label_encoder.classes_)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ...

# Define the neural network model using TensorFlow and Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation for classification
])
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions on the testing data
y_pred = np.argmax(model.predict(X_test), axis=1)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
