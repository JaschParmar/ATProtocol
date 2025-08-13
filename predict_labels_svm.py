import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# === Load unlabeled dataset ===
with open('unlabeled_4k_dataset.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Drop unused fields
df_features = df.drop(columns=[
    'handle', 'display_name', 'description', 'created_at'
], errors='ignore')

# === CRITICAL FIX: Handle missing values ===
print(f"Dataset shape before cleaning: {df_features.shape}")
print(f"Missing values per column:")
print(df_features.isnull().sum())

# Fill missing values with appropriate defaults
df_features = df_features.fillna({
    'followers_count': 0,
    'follows_count': 0,
    'followers_following_ratio': 0,
    'posts_count': 0,
    'account_age': 0,
    'description_length': 0,
    'is_verified': False,
    'average_post_interval': df_features['average_post_interval'].median(),
    'repeated_post_ratio': 0,
    'contains_links_ratio': 0,
    'hashtags_per_post': 0,
    'burst_posting': False,
    'avg_likes_received': 0,
    'avg_replies_received': 0,
    'replies_to_others': 0,
    'reposts_of_others': 0
})

# Convert boolean columns to int (if any)
bool_columns = df_features.select_dtypes(include=['bool']).columns
df_features[bool_columns] = df_features[bool_columns].astype(int)

print(f"Dataset shape after cleaning: {df_features.shape}")
print(f"Missing values after cleaning: {df_features.isnull().sum().sum()}")

# === Load model pipeline ===
model_pipeline = joblib.load('bot_detector_model_svm_final.pkl')

# === Predict ===
predictions = model_pipeline.predict(df_features)
predicted_labels = ['bot' if label == 1 else 'human' for label in predictions]

# Add predicted labels and print each one
for i, entry in enumerate(data):
    entry['predicted_label_svm'] = predicted_labels[i]
    print(f"🔍 {entry.get('handle', '[unknown]')} → {predicted_labels[i]}")

# === Save labeled data ===
with open('labeled_4k_dataset_svm.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✅ All accounts labeled with SVM and saved to labeled_4k_dataset_svm.json")

# === Count and show bot/human split ===
label_counts = Counter(predicted_labels)
print("\n=== Label Distribution (SVM) ===")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# === Plot feature importance (permutation importance) ===
from sklearn.inspection import permutation_importance

result = permutation_importance(model_pipeline, df_features, predictions, n_repeats=10, random_state=42, n_jobs=-1)
importances = result.importances_mean
feature_names = df_features.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Permutation Importance")
plt.title("SVM: Feature Importance for Bot Detection")
plt.tight_layout()
plt.savefig("feature_importance_svm_on_4k.png")
print("✅ Feature importance plot saved as feature_importance_svm_on_4k.png")

# === Show feature importance ranking ===
print("\n=== Feature Importance Ranking (SVM) ===")
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))
