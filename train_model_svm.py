import json
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
with open('initial_dataset.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Drop unnecessary or non-numeric fields
df = df.drop(columns=[
    'handle', 'display_name', 'description', 'created_at'  # remove text fields
])

# Drop rows with missing values (optional but safe for training)
df = df.dropna()

# Encode labels: convert 'bot'/'human' to 1/0
df['label'] = df['label'].map({'bot': 1, 'human': 0})

# Separate features and labels
X = df.drop(columns=['label'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with StandardScaler and SVM
# SVM requires feature scaling and we'll use RBF kernel
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

# Train on full dataset
pipeline.fit(X, y)

# Evaluate the model on test set
y_pred = pipeline.predict(X_test)
print("=== Classification Report (SVM) ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix (SVM) ===")
print(confusion_matrix(y_test, y_pred))

# Save the model pipeline
joblib.dump(pipeline, 'bot_detector_model_svm.pkl')
print("✅ SVM model saved as bot_detector_model_svm.pkl")

# Feature importance visualization using permutation importance
# (SVM doesn't have built-in feature importance like Random Forest)
from sklearn.inspection import permutation_importance

result = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importances = result.importances_mean
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Permutation Importance")
plt.title("SVM: Which features matter most?")
plt.tight_layout()
plt.savefig("feature_importance_svm.png")
print("✅ Plot saved as feature_importance_svm.png")

# Cross validation
# Optional: shuffle once before CV
X, y = shuffle(X, y, random_state=6)

# Create new pipeline for CV
cv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

scores = cross_val_score(cv_pipeline, X, y, cv=5)  # 5-fold CV

print("Cross-validation accuracy scores:", scores)
print(f"Mean accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

######################################################
# Train final model on all data
print("\n=== Training Final SVM Model on Full Dataset ===")

# Load your labeled data
with open('initial_dataset.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Drop unused columns
df = df.drop(columns=['handle', 'display_name', 'description', 'created_at'])

# Drop rows with missing values
df = df.dropna()

# Map labels to numbers
df['label'] = df['label'].map({'bot': 1, 'human': 0})

# Features and labels
X = df.drop(columns=['label'])
y = df['label']

# Train final model on all data with pipeline
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

final_pipeline.fit(X, y)

# Save the trained model
joblib.dump(final_pipeline, 'bot_detector_model_svm_final.pkl')
print("✅ Final SVM model trained on full dataset and saved.")
