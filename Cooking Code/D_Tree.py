# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset into a pandas DataFrame
data = pd.read_csv('Supplementary 2.csv')  # Replace 'your_dataset.csv' with the actual filename

# Preprocess the data (you may need to handle missing values and text data)
# For this example, we'll drop rows with missing values
data.dropna(inplace=True)

# Define the features (X) and the target variable (y)
# Assuming 'ASSOCIATION_CLASS' is a categorical column
X = pd.get_dummies(data[['YEAR', 'WEIGHT','GENE','REF_GENE','REF_SENTENCE','TITLE','DIS_CLASS','CONCLUSION']])

# Define the target variable
y = data['ASSOCIATION_CLASS']  # Replace 'CONCLUSION' with the actual target column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()

# Train the Decision Tree model on the training data
decision_tree_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = decision_tree_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
