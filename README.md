# 🏦 Loan Approval Prediction System

This project develops a **machine learning-based loan approval prediction system** using demographic, financial, and credit history data.  
It covers the entire workflow from data preprocessing, model training with XGBoost, hyperparameter tuning, and deployment via a Streamlit web app.

---

## ⚙️ Project Workflow

1. **Data Analysis (loan_approval_analysis.ipynb)**  
   - Exploratory data analysis (EDA).  
   - Feature engineering and preprocessing.  
   - Initial model experiments.  

2. **Model Training (train_model.py)**  
   - Data preprocessing pipeline with `OrdinalEncoder`, `OneHotEncoder`, and `RobustScaler`.  
   - Model training using **XGBoost Classifier**.  
   - Hyperparameter tuning with `GridSearchCV`.  
   - Saving trained model and preprocessing assets (`.pkl` files).  

3. **Deployment (app.py)**  
   - Streamlit web app for user-friendly predictions.  
   - Input form for applicant details.  
   - Real-time loan approval prediction (✅ Approved / ❌ Rejected).  

---

## 📊 Dataset

The dataset contains features such as:  
- **Demographics**: age, gender, education, home ownership.  
- **Financial**: income, loan amount, interest rate, percent income.  
- **Credit history**: credit score, previous loan defaults.  
- **Target**: Loan status (approved/rejected).  

---

## 📈 Results and Conclusion

- The **XGBoost model** achieved strong performance after hyperparameter tuning.  
- Key features influencing approval included **credit score, income, loan percent income, and credit history length**.  
- The deployed Streamlit app provides an easy-to-use interface for real-time predictions.
