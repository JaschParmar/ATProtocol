import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # to save your model

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

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)  # Train on full dataset

# Evaluate the model
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(model, 'bot_detector_model.pkl')
print("✅ Model saved as bot_detector_model.pkl")

# Visualisation
import matplotlib.pyplot as plt

importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Which features matter most?")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("✅ Plot saved as feature_importance.png")

# cross validation
from sklearn.model_selection import cross_val_score
import numpy as np

# Optional: shuffle once before CV
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=6)

model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X, y, cv=5)  # 5-fold CV

print("Cross-validation accuracy scores:", scores)
print(f"Mean accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

######################################################
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
import joblib

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

# Train final model on all data
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'bot_detector_model_final.pkl')
print("✅ Final model trained on full dataset and saved.")