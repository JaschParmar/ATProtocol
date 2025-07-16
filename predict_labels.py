import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from collections import Counter

# === Load unlabeled dataset ===
with open('unlabeled_4k_dataset.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Drop unused fields
df_features = df.drop(columns=[
    'handle', 'display_name', 'description', 'created_at'
], errors='ignore')

# === Load model ===
model = joblib.load('bot_detector_model_final.pkl')

# === Predict ===
predictions = model.predict(df_features)
predicted_labels = ['bot' if label == 1 else 'human' for label in predictions]

# Add predicted labels and print each one
for i, entry in enumerate(data):
    entry['predicted_label'] = predicted_labels[i]
    print(f"🔍 {entry.get('handle', '[unknown]')} → {predicted_labels[i]}")

# === Save labeled data ===
with open('labeled_4k_dataset.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✅ All accounts labeled and saved to labeled_4k_dataset.json")

# === Count and show bot/human split ===
label_counts = Counter(predicted_labels)
print("\n=== Label Distribution ===")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# === Plot feature importance ===
importances = model.feature_importances_
feature_names = df_features.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Feature Importance for Bot Detection")
plt.tight_layout()
plt.savefig("feature_importance_on_4k.png")
#plt.show()
print("✅ Feature importance plot saved as feature_importance_on_4k.png")