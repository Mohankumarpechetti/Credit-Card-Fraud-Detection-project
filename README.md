# 💳 Credit Card Fraud Detection using Machine Learning

## 📘 Project Overview
Credit card fraud is a significant challenge in the financial industry.  
This project aims to detect fraudulent transactions using various **Machine Learning classification models** and improve accuracy through **data preprocessing and model tuning**.

---

## 🧠 Objective
To build a robust machine learning model that can **accurately identify fraudulent credit card transactions** while minimizing false positives.

---

## 📂 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions  
- **Fraud cases:** 492 (highly imbalanced)
- **Features:** 30 (28 anonymized numerical features + `Time`, `Amount`, `Class`)

---

## 🧹 Data Preprocessing
- Handled class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**
- Applied **StandardScaler** for feature scaling
- Split data into **Train (80%)** and **Test (20%)**

---

## ⚙️ Machine Learning Models Used
| Model | Description |
|--------|-------------|
| Logistic Regression | Baseline linear model |
| Decision Tree | Simple non-linear classifier |
| Random Forest | Ensemble of decision trees |
| Support Vector Machine (SVM) | Finds optimal separating hyperplane |
| XGBoost | Gradient boosting algorithm for classification |

---

## 📊 Evaluation Metrics
| Metric | Meaning |
|---------|----------|
| Accuracy | Overall correctness of predictions |
| Precision | % of predicted frauds that are actual frauds |
| Recall | % of actual frauds correctly identified |
| F1-Score | Balance between precision and recall |
| ROC-AUC | Area under the ROC curve (model discrimination ability) |

---

## 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|-----------|------------|--------|-----------|----------|
| Logistic Regression | 0.975 | 0.92 | 0.87 | 0.89 | 0.98 |
| Decision Tree | 0.998 | 0.96 | 0.94 | 0.95 | 0.99 |
| Random Forest | **0.9996** | **0.98** | **0.97** | **0.98** | **0.999** |
| SVM | 0.998 | 0.93 | 0.90 | 0.91 | 0.99 |
| XGBoost | 0.999 | 0.95 | 0.94 | 0.94 | 0.998 |

✅ **Best Model:** Random Forest Classifier  
✅ **Test Accuracy:** 99.96%

---

## 📈 Visualizations
- Correlation heatmap of features  
- Class distribution before & after SMOTE  
- Confusion Matrix  
- ROC Curve  
- Feature Importance chart

---

## 🧰 Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn, XGBoost  
- **Tools:** Jupyter Notebook / Google Colab

---

## 📬 Results & Insights
- The dataset was extremely **imbalanced**, requiring oversampling.
- **Random Forest** delivered the best recall and precision balance.
- Proper feature scaling and tuning improved performance significantly.

---
