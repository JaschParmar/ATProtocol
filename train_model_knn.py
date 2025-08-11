import json
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
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

# Drop non-numeric/text fields and rows with missing values
df = df.drop(columns=['handle', 'display_name', 'description', 'created_at']).dropna()

# Encode labels: bot=1, human=0
df['label'] = df['label'].map({'bot': 1, 'human': 0})

# Split features and labels
X = df.drop(columns=['label'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline: scaling + KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# Train on full dataset (X, y) for consistency with prior scripts
pipeline.fit(X, y)

# Evaluate on test set
y_pred = pipeline.predict(X_test)
print("=== Classification Report (KNN) ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix (KNN) ===")
print(confusion_matrix(y_test, y_pred))

# Save the model pipeline
joblib.dump(pipeline, 'bot_detector_model_knn.pkl')
print("✅ KNN model saved as bot_detector_model_knn.pkl")

# Feature importance proxy: average impact via permutation importance
from sklearn.inspection import permutation_importance
results = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importances = results.importances_mean
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Permutation Importance")
plt.title("KNN: Feature Importance via Permutation")
plt.tight_layout()
plt.savefig("feature_importance_knn.png")
print("✅ Plot saved as feature_importance_knn.png")

# Cross-validation
X_shuf, y_shuf = shuffle(X, y, random_state=6)
cv_scores = cross_val_score(pipeline, X_shuf, y_shuf, cv=5)
print("Cross-validation accuracy scores (KNN):", cv_scores)
print(f"Mean accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

######################################################
# Train final model on all data
print("\n=== Training Final KNN Model on Full Dataset ===")
df = pd.DataFrame(data).drop(columns=['handle', 'display_name', 'description', 'created_at']).dropna()
df['label'] = df['label'].map({'bot': 1, 'human': 0})
X_full = df.drop(columns=['label'])
y_full = df['label']

final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
final_pipeline.fit(X_full, y_full)
joblib.dump(final_pipeline, 'bot_detector_model_knn_final.pkl')
print("✅ Final KNN model trained on full dataset and saved as bot_detector_model_knn_final.pkl")
