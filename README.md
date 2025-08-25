# Bluesky Bot Detector

This repository contains a complete end-to-end pipeline for detecting bot accounts on the Bluesky social network, inspired by academic research on Twitter bot detection. The project implements **ensemble learning** using multiple machine learning models with majority voting for robust bot detection.

## 📌 Overview

The aim of this project is to adapt Twitter-based bot detection strategies to the Bluesky platform using its API and a manually curated training set. It includes:

- Data collection from Bluesky via the ATProtocol
- Manual labeling of an initial training set
- Feature engineering from user metadata and post behavior
- **Multi-model ensemble training** (Random Forest, Logistic Regression, KNN, SVM, Neural Network)
- **Majority voting classification** for improved accuracy
- Prediction on a larger, unlabeled dataset of ~4,200 accounts

***

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

***

## 🤖 Ensemble Model Training

The labeled dataset was used to train **five different machine learning models** to create a robust ensemble classifier:

### 🔬 Individual Model Performance

| Model | Cross-Validation Accuracy | Test Accuracy | Precision (Bot) | Recall (Bot) |
|-------|--------------------------|---------------|-----------------|--------------|
| **Random Forest** | 0.87 ± 0.08 | 1.00 | 1.00 | 1.00 |
| **Logistic Regression** | 0.88 ± 0.08 | 0.89 | 0.86 | 0.86 |
| **K-Nearest Neighbors** | 0.87 ± 0.07 | 0.79 | 0.80 | 0.57 |
| **Support Vector Machine** | 0.84 ± 0.09 | 0.84 | 0.83 | 0.71 |
| **Neural Network (MLP)** | 0.65 ± 0.14 | 0.47 | 0.36 | 0.57 |

### 🗳️ Majority Voting Ensemble

The final classification uses **majority voting** across all five models:
- If ≥3 models predict "bot" → classified as **bot**
- If ≥3 models predict "human" → classified as **human**

**Ensemble Agreement Patterns:**
- **Unanimous decisions**: 27.5% of cases (25.2% unanimous bot, 2.3% unanimous human)
- **Strong agreement (4-1)**: 57.2% of cases
- **Close decisions (3-2)**: 15.3% of cases

### 🧠 Key Feature Insights
Across all models, the most important features for bot detection were:
1. `avg_likes_received` - Behavioral engagement patterns
2. `avg_replies_received` - Social interaction metrics  
3. `average_post_interval` - Temporal posting behavior

These behavioral markers proved more significant than simple follower counts.

***

## 🌍 Scaling Up to 4K+ Accounts

To create a larger, unlabeled dataset:
- 200,000 followers of the **official Bluesky account** were scraped.
- A random sample of 4,321 was taken.
- Accounts with **fewer than 5 posts** were filtered out.
- 72 were deleted/inaccessible during the process.
- Final count: **4,249 active accounts**

### 🎯 Final Ensemble Results

**Majority Voting Classification:**
- **Human accounts**: 631 (14.9%) ✅
- **Bot accounts**: 3,618 (85.1%) 🤖

**Model Agreement Analysis:**
- High confidence decisions (4+ models agree): 82.0%
- Unanimous bot detection: 1,072 accounts (25.2%)
- Unanimous human detection: 98 accounts (2.3%)

The high bot detection rate reflects the specific sampling methodology from official Bluesky followers, where automated and promotional accounts are common.

***

## 📁 Repository Structure

### 🧠 Training Scripts

| File | Description |
|------|-------------|
| `train_model.py` | Random Forest model training and evaluation |
| `train_model_logistic.py` | Logistic Regression with feature scaling |
| `train_model_knn.py` | K-Nearest Neighbors with permutation importance |
| `train_model_svm.py` | Support Vector Machine with RBF kernel |
| `train_model_mlp.py` | Multi-layer Perceptron neural network |

### 🔮 Prediction Scripts

| File | Description |
|------|-------------|
| `predict_labels.py` | Random Forest predictions on 4K dataset |
| `predict_labels_logistic.py` | Logistic Regression predictions |
| `predict_labels_knn.py` | KNN predictions |
| `predict_labels_svm.py` | SVM predictions |
| `predict_labels_mlp.py` | Neural network predictions |
| `majority_voting.py` | **Ensemble majority voting system** |

### 📊 Dataset Files

| File | Description |
|------|-------------|
| `initial_dataset.json` | 108 manually labeled accounts (training set) |
| `unlabeled_4k_dataset.json` | 4,249 accounts with extracted features |
| `labeled_4k_dataset_majority_voting.json` | **Final ensemble predictions** |
| `labeled_4k_dataset_[model].json` | Individual model predictions |

### 🤖 Trained Models

| File | Description |
|------|-------------|
| `bot_detector_model_[model]_final.pkl` | Production-ready trained models |
| `bot_detector_model_[model].pkl` | Evaluation models (80/20 split) |

### 📈 Analysis & Visualization

| File | Description |
|------|-------------|
| `feature_importance_[model].png` | Feature importance charts for each model |
| `majority_voting_analysis.png` | Comprehensive ensemble analysis visualization |

***

## 🚀 Usage

1. **Train all models:**
   ```bash
   python train_model.py
   python train_model_logistic.py
   python train_model_knn.py
   python train_model_svm.py
   python train_model_mlp.py
   ```

2. **Generate predictions:**
   ```bash
   python predict_labels.py
   python predict_labels_logistic.py
   python predict_labels_knn.py
   python predict_labels_svm.py
   python predict_labels_mlp.py
   ```

3. **Create ensemble predictions:**
   ```bash
   python majority_voting.py
   ```

The final ensemble results will be saved in `labeled_4k_dataset_majority_voting.json`.

***
