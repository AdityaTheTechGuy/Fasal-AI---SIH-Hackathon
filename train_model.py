# train_model_tuned.py

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- 1. Load and Prepare Data ---
print("Step 1: Loading and preparing data...")
df = pd.read_csv('crop_recommendation_dataset.csv')

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data preparation complete.")
print("-" * 30)


# --- 2. Hyperparameter Tuning with RandomizedSearchCV ---
print("Step 2: Setting up Hyperparameter Tuning...")

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the base model
rf = RandomForestClassifier(random_state=42)

# Initialize RandomizedSearchCV
# n_iter=25 means it will try 25 different combinations of parameters.
# cv=5 means it will use 5-fold cross-validation for each combination.
# n_jobs=-1 uses all available CPU cores to speed up the process.
rf_random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=25,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the random search model (this will take longer than the simple fit)
print("Starting the search... this may take a few minutes.")
rf_random_search.fit(X_train, y_train)

# Get the best model found by the search
best_model = rf_random_search.best_estimator_
print("\nHyperparameter search complete.")
print("Best Parameters Found:", rf_random_search.best_params_)
print("-" * 30)


# --- 3. Evaluate the Best Model ---
print("Step 3: Evaluating the best model...")
y_pred = best_model.predict(X_test)
new_accuracy = accuracy_score(y_test, y_pred)

print(f"Tuned Model Accuracy: {new_accuracy * 100:.2f}%")
print("-" * 30)


# --- 4. Save the Tuned Model ---
# Note: We are no longer using a LabelEncoder in this workflow,
# so we only need to save the model itself.
print("Step 4: Saving the tuned model...")
with open('crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("✅ Tuned model has been saved successfully.")