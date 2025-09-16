import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- Step 1: Load and Prepare the Data ---

print("Step 1: Loading and preparing data...")

# Load the dataset from the CSV file
df = pd.read_csv('crop_recommendation_dataset.csv')

# Separate the features (input variables) from the target (output variable)
# X gets all the columns except for 'label'
X = df.drop('label', axis=1)
# y gets only the 'label' column
y = df['label']

# Since the crop names are text, we need to convert them to numbers for the model.
# The LabelEncoder assigns a unique number to each crop name (e.g., Apple=0, Banana=1).
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into a training set and a testing set.
# The model will learn from the training set and we'll check its performance on the testing set.
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Data preparation complete.")
print(f"Training set has {X_train.shape[0]} samples.")
print(f"Testing set has {X_test.shape[0]} samples.")
print("-" * 30)


# --- Step 2: Train the Random Forest Model ---

print("Step 2: Training the Random Forest model...")

# Initialize the Random Forest Classifier.
# n_estimators is the number of decision trees in the forest. 100 is a robust choice.
# random_state ensures that we get the same results every time we run the script.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using our training data
model.fit(X_train, y_train)

print("Model training complete.")
print("-" * 30)


# --- Step 3: Evaluate the Model's Performance ---

print("Step 3: Evaluating model performance...")

# Use the trained model to make predictions on the test data (which it has never seen)
y_pred_encoded = model.predict(X_test)

# Compare the model's predictions with the actual correct labels to calculate accuracy
accuracy = accuracy_score(y_test, y_pred_encoded)

# The original labels are numbers; we can convert them back to text to see some examples
y_pred_text = label_encoder.inverse_transform(y_pred_encoded)

print(f"Model Accuracy on the test set: {accuracy * 100:.2f}%")
print("-" * 30)


# --- Step 4: Save the Model and Encoder to Files ---

print("Step 4: Saving the trained model and label encoder...")

# Save the trained model object to a file called 'crop_recommendation_model.pkl'
with open('crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the label encoder object to a file called 'label_encoder.pkl'
# This is important so the Flask app can convert the numeric predictions back to text labels if needed.
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Model and label encoder have been saved successfully.")
print("You can now restart your Flask application to use the new model.")