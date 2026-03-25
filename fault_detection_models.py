"""
Railway Track Fault Detection - ML Models for Fault Classification & Prediction
Phase 2: Build, train, and evaluate fault detection models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("RAILWAY TRACK FAULT DETECTION - MACHINE LEARNING MODELS")
print("=" * 80)

# =============================================================================
# STEP 1: LOAD PROCESSED DATA
# =============================================================================

print("\n📂 LOADING PROCESSED DATA")
plc_df = pd.read_csv('processed_data/plc_processed.csv')
print(f"✓ Loaded PLC data: {plc_df.shape}")

# =============================================================================
# STEP 2: PREPARE DATA FOR MODELING
# =============================================================================

def prepare_modeling_data(df):
    """Prepare data for ML modeling"""
    print("\n🔧 PREPARING DATA FOR MODELING")
    
    # Identify target variables
    target_cols = ['Failure_Type', 'Maintenance_Action']
    available_targets = [col for col in target_cols if col in df.columns]
    
    print(f"Available targets: {available_targets}")
    
    if not available_targets:
        print("No suitable target columns found!")
        return None, None, None
    
    # Use first available target
    target_col = available_targets[0]
    print(f"Using target: {target_col}")
    
    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target-related columns if they're numeric
    cols_to_remove = ['RUL_Predicted_days', 'Edge_Anomaly_Score', 
                     'Predicted_Failure_Prob', 'Cloud_Health_Index']
    numeric_features = [col for col in numeric_features 
                       if col not in cols_to_remove and col != target_col]
    
    print(f"Selected {len(numeric_features)} numeric features")
    
    # Handle target encoding
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        le = LabelEncoder()
        y = le.fit_transform(df[target_col].astype(str))
        print(f"Encoded {len(le.classes_)} classes: {le.classes_[:10]}...")
    else:
        y = df[target_col].values
        le = None
    
    # Prepare X
    X = df[numeric_features].fillna(0).values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 2 else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    print(f"Target distribution - Train: {np.bincount(y_train)}")
    print(f"Target distribution - Test: {np.bincount(y_test)}")
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            numeric_features, target_col, le, scaler)

X_train, X_test, y_train, y_test, feature_names, target_name, le, scaler = prepare_modeling_data(plc_df)

if X_train is None:
    print("❌ Could not prepare modeling data. Exiting.")
    exit()

# =============================================================================
# STEP 3: BASELINE MODELS - FAULT CLASSIFICATION
# =============================================================================

print("\n" + "=" * 70)
print("BASELINE MODELS - FAULT CLASSIFICATION")
print("=" * 70)

models = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    # 'GradientBoosting': GradientBoostingClassifier(random_state=42),
    # 'SVM': SVC(probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Test predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    accuracy = (y_pred == y_test).mean()
    print(f"Test Accuracy: {accuracy:.4f}")
    
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': accuracy,
        'predictions': y_pred
    }

# =============================================================================
# STEP 4: BEST MODEL TUNING
# =============================================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING - BEST BASELINE MODEL")
print("=" * 70)

# Find best baseline model
best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
print(f"Best baseline model: {best_model_name} (CV: {results[best_model_name]['cv_mean']:.4f})")

# Tune Random Forest (usually performs well for tabular data)
if best_model_name == 'RandomForest' or 'RandomForest' in results:
    print("\n🔧 Tuning Random Forest...")
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best RF params: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Test tuned model
    y_pred_rf = best_rf.predict(X_test)
    rf_accuracy = (y_pred_rf == y_test).mean()
    print(f"Tuned RF Test Accuracy: {rf_accuracy:.4f}")
    
    tuned_model = best_rf
else:
    tuned_model = results[best_model_name]['model']

# =============================================================================
# STEP 5: MODEL EVALUATION & VISUALIZATION
# =============================================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

# Detailed evaluation of best model
best_preds = tuned_model.predict(X_test)
best_proba = tuned_model.predict_proba(X_test)[:, 1] if hasattr(tuned_model, "predict_proba") else None

print("\n📊 Classification Report:")
print(classification_report(y_test, best_preds, target_names=le.classes_ if le else None))

# Confusion Matrix
plt.figure(figsize=(10, 8))

cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_[:10] if le else range(cm.shape[1]),
            yticklabels=le.classes_[:10] if le else range(cm.shape[0]))
plt.title('Confusion Matrix - Best Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('analysis_output/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_output/confusion_matrix.png")
plt.close()

# Feature Importance (if available)
if hasattr(tuned_model, 'feature_importances_'):
    importances = tuned_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('analysis_output/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: analysis_output/feature_importance.png")
    plt.close()
    
    print("\n🔍 Top 10 Important Features:")
    for i, idx in enumerate(indices[:10]):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# =============================================================================
# STEP 6: ANOMALY DETECTION MODEL
# =============================================================================

print("\n" + "=" * 70)
print("ANOMALY DETECTION MODEL")
print("=" * 70)

# Train Isolation Forest for anomaly detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_train)

# Predict anomalies
anomaly_labels = iso_forest.predict(X_test)
anomaly_scores = iso_forest.score_samples(X_test)

# Convert to binary (1=anomaly, 0=normal)
anomalies = (anomaly_labels == -1).astype(int)

print(f"Anomaly Detection Results:")
print(f"  Detected {anomalies.sum()} anomalies out of {len(y_test)} test samples")
print(f"  Anomaly rate: {anomalies.mean():.1%}")

# Plot anomaly scores
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(anomaly_scores, bins=50, alpha=0.7)
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.scatter(range(len(anomaly_scores)), anomaly_scores, c=anomalies, cmap='coolwarm', alpha=0.6)
plt.title('Anomaly Scores (Test Set)')
plt.xlabel('Sample Index')
plt.ylabel('Anomaly Score')
plt.axhline(y=np.percentile(anomaly_scores, 90), color='red', linestyle='--', label='90th Percentile')

plt.tight_layout()
plt.savefig('analysis_output/anomaly_detection.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_output/anomaly_detection.png")
plt.close()

# =============================================================================
# STEP 7: SAVE MODELS
# =============================================================================

print("\n" + "=" * 70)
print("SAVING TRAINED MODELS")
print("=" * 70)

os.makedirs('trained_models', exist_ok=True)

# Save best classification model
joblib.dump(tuned_model, 'trained_models/fault_classifier.pkl')
joblib.dump(scaler, 'trained_models/feature_scaler.pkl')
joblib.dump(le, 'trained_models/label_encoder.pkl')

# Save anomaly detector
joblib.dump(iso_forest, 'trained_models/anomaly_detector.pkl')

print("✓ Saved models:")
print("  - trained_models/fault_classifier.pkl")
print("  - trained_models/anomaly_detector.pkl")
print("  - trained_models/feature_scaler.pkl")
print("  - trained_models/label_encoder.pkl")

# =============================================================================
# STEP 8: MODEL PERFORMANCE SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)

print(f"\n🏆 BEST CLASSIFICATION MODEL:")
print(f"  Model: {best_model_name} (Tuned)")
print(f"  Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
print(f"  Cross-Validation: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std']:.4f})")

print(f"\n🔍 ANOMALY DETECTION:")
print(f"  Model: Isolation Forest")
print(f"  Anomaly Detection Rate: {anomalies.mean():.1%}")
print(f"  Anomalies Found: {anomalies.sum()}/{len(y_test)}")

print(f"\n📁 NEW FILES CREATED:")
print("  Visualizations:")
print("  • analysis_output/confusion_matrix.png")
print("  • analysis_output/feature_importance.png")
print("  • analysis_output/anomaly_detection.png")
print("  Models:")
print("  • trained_models/*.pkl")

print("\n🚀 NEXT STEPS:")
print("1. Review visualizations for model performance")
print("2. Test models with new data")
print("3. Deploy real-time monitoring system")
print("4. Build predictive maintenance dashboard")

print("\n" + "=" * 80)
print("✅ FAULT DETECTION MODELS READY FOR DEPLOYMENT!")
print("=" * 80)
