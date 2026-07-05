# 📉 Customer Churn Prediction

Customer churn prediction using Logistic Regression with cross-validation, class balancing, and StandardScaler feature normalization.

## ✨ Features
* **Multi-Model Cross Validation:** Compares Logistic Regression, Random Forest, and SVC using K-Fold cross validation to identify the best performing algorithm.
* **Class Imbalance Handling:** Uses balanced class weights to address the 80/20 churn distribution, ensuring the model learns from minority class patterns rather than predicting the majority class only.
* **StandardScaler Normalization:** Scales all features to a standard distribution before model training to ensure fair distance-based classification.
* **Confusion Matrix Visualization:** Evaluates misclassifications visually using ConfusionMatrixDisplay to assess true and false predictions for both classes.
* **Classification Report:** Evaluates model performance using precision, recall, and F1-score — prioritising churn recall over raw accuracy for real-world business relevance.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Data Analytics:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn (LogisticRegression, RandomForestClassifier, SVC, StandardScaler, cross_val_score)
* **Dataset:** customer_churn.csv

## 🚀 Setup
1. Clone the repo.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Place customer_churn.csv and new_customers_1.csv in the project folder.
4. Run Main.ipynb cell by cell.
5. Find predictions in Predictions_test.csv!

## 📊 Results
* **Best Model:** Logistic Regression with balanced class weights
* **Train/ Test Accuracy:** train and test accuracy nearly identical — no overfitting
* **Churn Recall:** 81%
* **Key Observation:** Raw accuracy is misleading on imbalanced datasets. With 83% of customers not churning, a naive model achieves 82% by predicting no churn for everyone. The balanced model correctly identifies 81% of actual churners — the metric that matters for the business.
