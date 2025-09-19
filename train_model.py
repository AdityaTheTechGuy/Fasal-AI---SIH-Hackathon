
# enhanced_train_model.py - Training with Large Enhanced Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import time
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- ENHANCED TRAINING CONFIGURATION ---
# Dataset selection - will try enhanced dataset first, fall back to original
ENHANCED_DATASET = 'enhanced_crop_dataset.csv'
ORIGINAL_DATASET = 'crop_recommendation_dataset.csv'
MODEL_OUTPUT = 'enhanced_crop_model.pkl'
ENABLE_HYPERPARAMETER_TUNING = True
USE_CROSS_VALIDATION = True
COMPARE_ALGORITHMS = True
TEST_SENSITIVITY = True
# ---------------------------------------

def load_and_prepare_data():
    """Load dataset with fallback options"""
    print("Step 1: Loading and preparing data...")

    # Try to load enhanced dataset first
    if os.path.exists(ENHANCED_DATASET):
        print(f"📊 Loading enhanced dataset: {ENHANCED_DATASET}")
        df = pd.read_csv(ENHANCED_DATASET)
        dataset_type = "enhanced"
    elif os.path.exists(ORIGINAL_DATASET):
        print(f"📊 Loading original dataset: {ORIGINAL_DATASET}")
        df = pd.read_csv(ORIGINAL_DATASET)
        dataset_type = "original"
    else:
        raise FileNotFoundError("No dataset found. Please generate enhanced_crop_dataset.csv first.")

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Standardize crop names
    print("🔧 Standardizing crop names...")
    original_crops = len(df['label'].unique())
    df['label'] = df['label'].str.strip().str.title()
    standardized_crops = len(df['label'].unique())

    if original_crops != standardized_crops:
        print(f"   Fixed case sensitivity: {original_crops} -> {standardized_crops} unique crops")

    # Display crop distribution
    print("\n📈 Crop distribution:")
    crop_counts = df['label'].value_counts().sort_index()
    for crop, count in crop_counts.items():
        print(f"   {crop}: {count:,} samples")

    # Prepare features and target
    X = df.drop('label', axis=1)
    y = df['label']

    print(f"\n✅ Data preparation complete")
    print(f"   Features: {list(X.columns)}")
    print(f"   Target classes: {len(y.unique())}")
    print(f"   Dataset type: {dataset_type}")

    return X, y, dataset_type

def compare_algorithms(X_train, X_test, y_train, y_test):
    """Compare different algorithms for sensitivity"""
    print("\n" + "="*60)
    print("Step 2: Comparing Algorithm Performance")
    print("="*60)

    algorithms = {
        'Random Forest (Small)': RandomForestClassifier(
            n_estimators=50, max_depth=15, random_state=42, 
            class_weight='balanced', n_jobs=-1
        ),
        'Random Forest (Medium)': RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, 
            class_weight='balanced', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, 
            random_state=42
        )
    }

    results = {}

    for name, model in algorithms.items():
        print(f"\n🔬 Testing {name}...")
        start_time = time.time()

        # Train model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation score
        if USE_CROSS_VALIDATION and len(X_train) < 100000:  # Skip CV for very large datasets
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            cv_mean, cv_std = None, None

        training_time = time.time() - start_time

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'training_time': training_time
        }

        print(f"   Accuracy: {accuracy:.4f}")
        if cv_mean:
            print(f"   CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
        print(f"   Training Time: {training_time:.2f}s")

    # Find best model
    best_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_name]['model']

    print(f"\n🏆 Best Algorithm: {best_name}")
    print(f"   Best Accuracy: {results[best_name]['accuracy']:.4f}")

    return best_model, best_name, results

def hyperparameter_tuning(model, X_train, y_train, model_name):
    """Perform hyperparameter tuning on the best model"""
    print(f"\n" + "="*60)
    print(f"Step 3: Hyperparameter Tuning for {model_name}")
    print("="*60)

    if not ENABLE_HYPERPARAMETER_TUNING:
        print("⏭️  Hyperparameter tuning disabled, using default parameters")
        return model

    # Define parameter grids based on model type
    if 'Random Forest' in model_name:
        param_grid = {
            'n_estimators': [30, 50, 75, 100],
            'max_depth': [10, 15, 20, 25],
            'min_samples_split': [2, 5, 8],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.8]
        }
        n_iter = 25
    elif 'Gradient Boosting' in model_name:
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0]
        }
        n_iter = 20
    else:
        print("⚠️  Unknown model type, skipping hyperparameter tuning")
        return model

    print(f"🔍 Searching {n_iter} parameter combinations...")

    # Perform randomized search
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring='accuracy'
    )

    search.fit(X_train, y_train)

    print(f"\n✅ Hyperparameter tuning complete!")
    print(f"   Best parameters: {search.best_params_}")
    print(f"   Best CV score: {search.best_score_:.4f}")
    print(f"   Improvement: {search.best_score_ - search.estimator.score(X_train, y_train):.4f}")

    return search.best_estimator_

def test_model_sensitivity(model, X_test, y_test):
    """Test the model's sensitivity to parameter changes"""
    print(f"\n" + "="*60)
    print("Step 4: Testing Model Sensitivity")
    print("="*60)

    if not TEST_SENSITIVITY:
        print("⏭️  Sensitivity testing disabled")
        return {}

    # Create test cases with different parameter variations
    base_case = np.array([[80, 60, 40, 25, 70, 6.5, 200]])
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    variations = [5, 5, 10, 2, 10, 0.3, 20]  # Reasonable variations

    print("🧪 Testing sensitivity to parameter changes...")

    # Get base prediction
    base_pred = model.predict(base_case)[0]
    base_proba = model.predict_proba(base_case)[0]

    sensitivity_results = {
        'base_prediction': base_pred,
        'base_confidence': max(base_proba),
        'parameter_sensitivity': {}
    }

    print(f"   Base prediction: {base_pred}")
    print(f"   Base confidence: {max(base_proba):.3f}")

    for i, (param, variation) in enumerate(zip(feature_names, variations)):
        # Test both positive and negative changes
        for direction, multiplier in [('increase', 1), ('decrease', -1)]:
            modified_case = base_case.copy()
            modified_case[0][i] += variation * multiplier

            # Ensure values stay within reasonable bounds
            if param == 'humidity':
                modified_case[0][i] = max(10, min(100, modified_case[0][i]))
            elif param == 'ph':
                modified_case[0][i] = max(3, min(10, modified_case[0][i]))
            elif param == 'temperature':
                modified_case[0][i] = max(8, min(45, modified_case[0][i]))

            new_pred = model.predict(modified_case)[0]
            new_proba = model.predict_proba(modified_case)[0]

            pred_changed = new_pred != base_pred
            confidence_change = max(new_proba) - max(base_proba)

            key = f"{param}_{direction}"
            sensitivity_results['parameter_sensitivity'][key] = {
                'prediction_changed': pred_changed,
                'new_prediction': new_pred,
                'confidence_change': confidence_change,
                'variation': variation * multiplier
            }

            if pred_changed:
                print(f"   ✓ {param} {direction} by {abs(variation * multiplier)}: {base_pred} -> {new_pred}")
            elif abs(confidence_change) > 0.05:
                print(f"   ~ {param} {direction} by {abs(variation * multiplier)}: confidence Δ{confidence_change:+.3f}")

    # Calculate overall sensitivity score
    prediction_changes = sum(1 for result in sensitivity_results['parameter_sensitivity'].values() 
                           if result['prediction_changed'])
    total_tests = len(sensitivity_results['parameter_sensitivity'])
    sensitivity_score = prediction_changes / total_tests

    print(f"\n📊 Sensitivity Analysis Results:")
    print(f"   Prediction changes: {prediction_changes}/{total_tests}")
    print(f"   Sensitivity score: {sensitivity_score:.3f}")

    if sensitivity_score > 0.3:
        print("   ✅ Model shows good sensitivity to parameter changes")
    elif sensitivity_score > 0.1:
        print("   ⚠️  Model shows moderate sensitivity")
    else:
        print("   ❌ Model shows low sensitivity - consider reducing ensemble size")

    return sensitivity_results

def save_enhanced_model(model, model_name, X, y, sensitivity_results, algorithm_results):
    """Save the model with comprehensive metadata"""
    print(f"\n" + "="*60)
    print("Step 5: Saving Enhanced Model")
    print("="*60)

    # Calculate feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X.columns, model.feature_importances_))
    else:
        feature_importance = {}

    # Prepare model metadata
    model_metadata = {
        'model': model,
        'model_name': model_name,
        'feature_names': list(X.columns),
        'feature_importance': feature_importance,
        'crop_labels': sorted(y.unique()),
        'training_samples': len(X),
        'n_features': len(X.columns),
        'n_crops': len(y.unique()),
        'algorithm_comparison': algorithm_results,
        'sensitivity_analysis': sensitivity_results,
        'model_params': model.get_params(),
        'creation_date': pd.Timestamp.now().isoformat()
    }

    # Save the model
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model_metadata, f)

    print(f"✅ Enhanced model saved as: {MODEL_OUTPUT}")
    print(f"   Model type: {model_name}")
    print(f"   Training samples: {len(X):,}")
    print(f"   Features: {len(X.columns)}")
    print(f"   Crops: {len(y.unique())}")

    if feature_importance:
        print("\n🔍 Top feature importance:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:5]:
            print(f"   {feature}: {importance:.4f}")

    return model_metadata

def main():
    """Main training pipeline"""
    print("🚀 Enhanced FasalAI Model Training Pipeline")
    print("="*60)

    start_time = time.time()

    try:
        # Step 1: Load and prepare data
        X, y, dataset_type = load_and_prepare_data()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\n📊 Data split:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Testing samples: {len(X_test):,}")

        # Step 2: Compare algorithms (if enabled)
        if COMPARE_ALGORITHMS:
            best_model, best_name, algorithm_results = compare_algorithms(X_train, X_test, y_train, y_test)
        else:
            # Use default Random Forest
            best_model = RandomForestClassifier(
                n_estimators=75, max_depth=20, random_state=42, 
                class_weight='balanced', n_jobs=-1
            )
            best_name = "Random Forest (Default)"
            algorithm_results = {}
            best_model.fit(X_train, y_train)

        # Step 3: Hyperparameter tuning
        tuned_model = hyperparameter_tuning(best_model, X_train, y_train, best_name)

        # Final evaluation
        final_predictions = tuned_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, final_predictions)

        print(f"\n🎯 Final Model Performance:")
        print(f"   Test Accuracy: {final_accuracy:.4f}")

        # Step 4: Test sensitivity
        sensitivity_results = test_model_sensitivity(tuned_model, X_test, y_test)

        # Step 5: Save model with metadata
        model_metadata = save_enhanced_model(
            tuned_model, best_name, X, y, sensitivity_results, algorithm_results
        )

        total_time = time.time() - start_time
        print(f"\n🏁 Training pipeline completed in {total_time:.2f} seconds")
        print(f"\n🎉 Your enhanced FasalAI model is ready!")
        print(f"   Model file: {MODEL_OUTPUT}")
        print(f"   Test accuracy: {final_accuracy:.4f}")

        if sensitivity_results.get('parameter_sensitivity'):
            sensitivity_score = sum(1 for r in sensitivity_results['parameter_sensitivity'].values() 
                                  if r['prediction_changed']) / len(sensitivity_results['parameter_sensitivity'])
            print(f"   Sensitivity score: {sensitivity_score:.3f}")

        return model_metadata

    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model_metadata = main()
