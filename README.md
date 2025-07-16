# Bluesky Bot Detector

This repository contains a complete end-to-end pipeline for detecting bot accounts on the Bluesky social network, inspired by academic research on Twitter bot detection.

## 📌 Overview

The aim of this project is to adapt Twitter-based bot detection strategies to the Bluesky platform using its API and a manually curated training set. It includes:

- Data collection from Bluesky via the ATProtocol
- Manual labeling of an initial training set
- Feature engineering from user metadata and post behavior
- Model training using Random Forest with evaluation
- Prediction on a larger, unlabeled dataset of ~4,200 accounts

---

## 🧪 Dataset Creation

### 🎯 Initial Labeled Dataset (108 accounts)

- **Human accounts**: 50 accounts sampled randomly from the follower list of [Jay](https://bsky.app/profile/jay.bsky.team), the founder of Bluesky.
- **Bot accounts**: 58 accounts identified manually by monitoring suspicious behavior using [firesky.tv](https://firesky.tv) (a live Bluesky firehose).

Each account was labeled manually (`bot` or `human`) and saved in `initial_dataset.json`.

### 📊 Features Extracted

Features were grouped into:

#### 🧍 User-Based Features
- `followers_count`, `follows_count`, `followers_following_ratio`
- `posts_count`, `account_age`, `is_verified`
- `description_length`

#### 📝 Post-Based Features
Extracted by analyzing up to 100 recent posts per user (API limit):
- `average_post_interval`
- `repeated_post_ratio`
- `contains_links_ratio`
- `hashtags_per_post`
- `burst_posting` (rapid-fire posting pattern)
- `avg_likes_received`
- `avg_replies_received`
- `replies_to_others`
- `reposts_of_others`

**Note:** The Bluesky API does *not* support many features available in Twitter such as:
- Retweet network structure
- Mentions timeline
- Total likes given by the account
- Geo-location
- Client app used

These limitations influenced feature selection.

---

## 🤖 Model Training

The labeled dataset was used to train a **Random Forest Classifier**. Key results:

### 🧪 Evaluation (80/20 train-test split):
=== Classification Report ===

                precision    recall  f1-score   support

      human         1.00      1.00      1.00        12
        bot         1.00      1.00      1.00         7

    accuracy                            1.00        19
    macro avg       1.00      1.00      1.00        19
    weighted avg    1.00      1.00      1.00        19

=== Confusion Matrix ===
  
  [[12  0]
  
   [ 0  7]]

### 🔁 5-Fold Cross Validation:
Cross-validation accuracy scores:
[0.8947, 0.8333, 0.9444, 0.7222, 0.9444]

Mean accuracy: 0.87
Standard deviation: ±0.08

### 🧠 Feature Importance (Top 3)
1. `avg_likes_received`
2. `avg_replies_received`
3. `average_post_interval`

These behavioral markers proved more significant than raw follower counts.

---

## 🌍 Scaling Up to 4K+ Accounts

To create a larger, unlabeled dataset:
- 200,000 followers of the **official Bluesky account** were scraped.
- A random sample of 4,321 was taken.
- Accounts with **fewer than 5 posts** were filtered out.
- 72 were deleted/inaccessible during the process.
- Final count: **4,249 active accounts**

The trained model was then used to predict labels for each account.

### 🔚 Results:

Out of 4,249 accounts:

	•	Human: 712 ✅
	•	Bot:   3,537 🤖

 While this ratio appears extreme, manual inspection of random samples confirmed the model's predictions aligned well with observable behavior.

---

## 📁 Files in This Repo

### 🧠 Code Files

| File                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `train_model.py`              | Trains a Random Forest model using the labeled dataset (`initial_dataset.json`) |
| `predict_labels.py`           | Applies the trained model to predict bot/human labels for the 4K dataset    |
| `initial_dataset.py`          | Script to build the initial labeled dataset (50 humans + manually found bots) |
| `generate_50_human.py`        | Collects 50 human accounts by sampling users followed by Jay                |
| `generate_4k_accounts.py`     | Samples 4,000+ random accounts from the official Bluesky followers list     |
| `4k_dataset.py`               | Computes features for the 4K+ accounts and saves the full unlabeled dataset |
| `trial.py`                    | Script for early-stage feature testing and debugging                        |

### 📊 Data Files

| File                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `initial_dataset.json`        | 108 manually labeled accounts (bots + humans)                              |
| `50_human.json`               | Raw list of 50 human handles used to build initial dataset                  |
| `4k_accounts.json`            | List of ~4321 sampled random Bluesky accounts                              |
| `unlabeled_4k_dataset.json`   | Feature-computed dataset for 4k+ accounts (no labels yet)                  |
| `labeled_4k_dataset.json`     | Final 4k+ dataset with model-predicted bot/human labels                    |

### 🤖 Model Files

| File                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `bot_detector_model.pkl`      | Model trained with 80/20 split (evaluation model)                          |
| `bot_detector_model_final.pkl`| Final model trained on full labeled data (used for real predictions)       |

### 📈 Visuals

| File                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `feature_importance.png`      | Feature importance chart from training phase                               |
| `feature_importance_on_4k.png`| Feature importance chart after labeling the 4k dataset                     |























