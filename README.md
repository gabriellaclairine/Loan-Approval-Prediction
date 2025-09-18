# Loan Approval Prediction Web App 🏦

A user-friendly web application built with Streamlit to predict loan approval decisions. This app leverages a pre-trained XGBoost model to provide instant predictions based on applicant data, helping to streamline the credit risk assessment process.


[Image of a loan application form]


## ✨ Features

- **Interactive UI**: An intuitive web interface for users to input applicant information easily.
- **Instant Predictions**: Get real-time loan approval predictions ("Approved" or "Rejected").
- **Powered by ML**: Utilizes an XGBoost classifier model trained on a credit dataset for accurate decision-making.
- **Data Preprocessing**: Includes a complete preprocessing pipeline for scaling and encoding user input to match the model's requirements.

---

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Handling**: Pandas, NumPy
- **Core Language**: Python

---

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/gabriellaclairine/Loan-Approval-Prediction.git
    cd Loan-Approval-Prediction
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You need to create a `requirements.txt` file containing all necessary packages like streamlit, pandas, scikit-learn, xgboost, etc.)*

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

The application will now be running on your local server!
