import json
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

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

# Create pipeline with StandardScaler and MLP
# MLP requires feature scaling and we'll use a simple 2-layer network
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(50, 25),  # Two hidden layers: 50 neurons, then 25
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=1000,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    ))
])

# Train on full dataset
pipeline.fit(X, y)

# Evaluate the model on test set
y_pred = pipeline.predict(X_test)
print("=== Classification Report (MLP) ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix (MLP) ===")
print(confusion_matrix(y_test, y_pred))

# Save the model pipeline
joblib.dump(pipeline, 'bot_detector_model_mlp.pkl')
print("✅ MLP model saved as bot_detector_model_mlp.pkl")

# Feature importance visualization using permutation importance
# (Neural networks don't have built-in feature importance)
from sklearn.inspection import permutation_importance

result = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importances = result.importances_mean
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Permutation Importance")
plt.title("MLP: Which features matter most?")
plt.tight_layout()
plt.savefig("feature_importance_mlp.png")
print("✅ Plot saved as feature_importance_mlp.png")

# Cross validation
# Optional: shuffle once before CV
X, y = shuffle(X, y, random_state=6)

# Create new pipeline for CV
cv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=1000,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    ))
])

scores = cross_val_score(cv_pipeline, X, y, cv=5)  # 5-fold CV

print("Cross-validation accuracy scores:", scores)
print(f"Mean accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

# Display training info
mlp_model = pipeline.named_steps['mlp']
print(f"\n=== MLP Training Info ===")
print(f"Number of iterations: {mlp_model.n_iter_}")
print(f"Number of layers: {mlp_model.n_layers_}")
print(f"Number of outputs: {mlp_model.n_outputs_}")
print(f"Final loss: {mlp_model.loss_:.4f}")

######################################################
# Train final model on all data
print("\n=== Training Final MLP Model on Full Dataset ===")

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
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=1000,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    ))
])

final_pipeline.fit(X, y)

# Save the trained model
joblib.dump(final_pipeline, 'bot_detector_model_mlp_final.pkl')
print("✅ Final MLP model trained on full dataset and saved.")

# Final model info
final_mlp = final_pipeline.named_steps['mlp']
print(f"Final model iterations: {final_mlp.n_iter_}")
print(f"Final model loss: {final_mlp.loss_:.4f}")
