import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load unlabeled dataset
with open('unlabeled_4k_dataset.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Drop unused fields
df_features = df.drop(columns=['handle', 'display_name', 'description', 'created_at'], errors='ignore')

# Handle missing values (same strategy as logistic)
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
bool_cols = df_features.select_dtypes(include=['bool']).columns
df_features[bool_cols] = df_features[bool_cols].astype(int)

# Load trained KNN pipeline
model_pipeline = joblib.load('bot_detector_model_knn_final.pkl')

# Predict
preds = model_pipeline.predict(df_features)
predicted_labels = ['bot' if p == 1 else 'human' for p in preds]

# Attach predictions
for i, entry in enumerate(data):
    entry['predicted_label_knn'] = predicted_labels[i]
    print(f"🔍 {entry.get('handle', '[unknown]')} → {predicted_labels[i]}")

# Save results
with open('labeled_4k_dataset_knn.json', 'w') as f:
    json.dump(data, f, indent=2)
print("✅ Accounts labeled with KNN saved to labeled_4k_dataset_knn.json")

# Distribution
counts = Counter(predicted_labels)
print("\n=== Label Distribution (KNN) ===")
for lbl, cnt in counts.items():
    print(f"{lbl}: {cnt}")

# Permutation importance plot
from sklearn.inspection import permutation_importance
results = permutation_importance(model_pipeline, df_features, preds, n_repeats=10, random_state=42, n_jobs=-1)
importances = results.importances_mean
feature_names = df_features.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Permutation Importance")
plt.title("KNN: Feature Importance on 4K Dataset")
plt.tight_layout()
plt.savefig("feature_importance_knn_on_4k.png")
print("✅ Plot saved as feature_importance_knn_on_4k.png")
