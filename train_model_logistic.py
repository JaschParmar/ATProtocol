import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

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

# Create pipeline with StandardScaler and LogisticRegression
# Logistic Regression benefits from feature scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Train on full dataset
pipeline.fit(X, y)

# Evaluate the model on test set
y_pred = pipeline.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Save the model pipeline
joblib.dump(pipeline, 'bot_detector_model_logistic.pkl')
print("✅ Logistic Regression model saved as bot_detector_model_logistic.pkl")

# Feature importance visualization (using coefficients)
# Get feature coefficients from the trained model
coefficients = pipeline.named_steps['classifier'].coef_[0]
feature_names = X.columns

# Get absolute values for importance ranking
abs_coefficients = np.abs(coefficients)

plt.figure(figsize=(10, 6))
plt.barh(feature_names, abs_coefficients)
plt.xlabel("Feature Importance (|Coefficient|)")
plt.title("Logistic Regression: Which features matter most?")
plt.tight_layout()
plt.savefig("feature_importance_logistic.png")
print("✅ Plot saved as feature_importance_logistic.png")

# Cross validation
# Optional: shuffle once before CV
X, y = shuffle(X, y, random_state=6)

# Create new pipeline for CV
cv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

scores = cross_val_score(cv_pipeline, X, y, cv=5)  # 5-fold CV

print("Cross-validation accuracy scores:", scores)
print(f"Mean accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

######################################################
# Train final model on all data
print("\n=== Training Final Model on Full Dataset ===")

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
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

final_pipeline.fit(X, y)

# Save the trained model
joblib.dump(final_pipeline, 'bot_detector_model_logistic_final.pkl')
print("✅ Final Logistic Regression model trained on full dataset and saved.")
