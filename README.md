# 📉 Customer Churn Prediction

Predicting which telecom customers are likely to leave. Churn matters because retaining an existing customer costs significantly less than acquiring a new one — so catching churners early has direct business value. The main challenge here was an 80/20 class imbalance that makes raw accuracy a useless metric.

## ✨ What I Did and Why
* **Class Imbalance Handling:** Used `class_weight='balanced'` on Logistic Regression instead of SMOTE or undersampling — a simpler, effective first approach for a dataset where the imbalance isn't extreme. This penalizes misclassifying the minority class (churners) more heavily during training.
* **Multi-Model Cross Validation:** Compared Logistic Regression, Random Forest, and SVC. Logistic Regression won — the dataset is likely close to linearly separable, and simpler models generalize better on tabular data with enough samples.
* **StandardScaler Normalization:** Scaled before training. Essential for Logistic Regression and SVC, which are sensitive to feature magnitude.
* **Recall-Focused Evaluation:** Prioritized churn recall over accuracy. With 83% of customers not churning, predicting "no churn" for everyone gives 83% accuracy while being completely useless for the business.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Data Analytics:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn (LogisticRegression, RandomForestClassifier, SVC, StandardScaler, cross_val_score)
* **Dataset:** `customer_churn.csv` — telecom customer data with features like tenure, monthly charges, contract type, and services used

## 🚀 Setup
1. Clone the repo.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Place `customer_churn.csv` and `new_customers_1.csv` in the project folder.
4. Run `Main.ipynb` cell by cell.
5. Find predictions in `Predictions_test.csv`.

## 📊 Results
* **Best Model:** Logistic Regression with `class_weight='balanced'`
* **Accuracy:** ~84% · **Churn Recall:** 81%
* **Key Observation:** A naive model predicting "no churn" for everyone scores 83% accuracy — barely below ours. What the naive model gets wrong: it catches 0% of churners. Our model catches 81% of them. That's the difference that matters.

## ⚠️ Limitations
* `class_weight='balanced'` increases churn recall but also increases false positives — the business cost of wrongly flagging a loyal customer needs to be weighed against the cost of missing a churner.
* Feature importance wasn't analyzed — doesn't answer *why* customers are churning, only *who* is likely to.
