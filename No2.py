# Import library yang dibutuhkan
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')  # Supaya warning tidak ditampilkan

# Kelas untuk menangani data -> mengelola data (load, bersihkan, encode, scale)
class LoanDataHandler:
    def __init__(self, file_path):
        self.file_path = file_path  # Path ke file dataset
        self.df = None  # DataFrame mentah
        self.X = None  # Fitur
        self.y = None  # Target (label)
        self.numerical_cols = [  # Kolom numerik yang akan di-scale
            'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score'
        ]
        self.ordinal_encoder = None
        self.onehot_encoder = None
        self.scaler = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)  # Membaca file CSV

    def preprocess(self):
        df = self.df.copy()  # Salin data
        df['person_income'] = df.groupby('loan_status')['person_income'].transform(lambda x: x.fillna(x.median()))  # Isi nilai kosong
        df['person_gender'] = df['person_gender'].str.lower().str.replace(' ', '')  # Normalisasi gender
        df['person_gender'] = df['person_gender'].replace({'male': 'Male', 'female': 'Female'})  # Konsistensi penulisan gender
        df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0}).astype(int)  # Encode boolean

        education_order = [['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']]  # Urutan pendidikan
        self.ordinal_encoder = OrdinalEncoder(categories=education_order)
        df['person_education'] = self.ordinal_encoder.fit_transform(df[['person_education']]).astype(int)  # Encode ordinal

        onehot_cols = ['person_gender', 'person_home_ownership', 'loan_intent']  # Kolom one-hot
        self.onehot_encoder = OneHotEncoder(sparse=False, drop='first')  # One-hot encoder
        onehot_array = self.onehot_encoder.fit_transform(df[onehot_cols])  # Transformasi
        onehot_df = pd.DataFrame(onehot_array, columns=self.onehot_encoder.get_feature_names_out(onehot_cols), index=df.index)  # Buat DataFrame baru
        df = pd.concat([df.drop(columns=onehot_cols), onehot_df], axis=1)  # Gabungkan dengan data asli

        self.X = df.drop('loan_status', axis=1)  # Pisahkan fitur
        self.y = df['loan_status']  # Simpan target

        self.scaler = RobustScaler()  # Gunakan RobustScaler untuk mengatasi outlier
        self.X[self.numerical_cols] = self.scaler.fit_transform(self.X[self.numerical_cols])  # Scaling kolom numerik

    def save_preprocessors(self):
        # Simpan semua encoder dan scaler ke file
        with open('ordinal_encoder.pkl', 'wb') as f:
            pickle.dump(self.ordinal_encoder, f)
        with open('onehot_encoder.pkl', 'wb') as f:
            pickle.dump(self.onehot_encoder, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('feature_list.pkl', 'wb') as f:
            pickle.dump(self.X.columns.tolist(), f)  # Simpan nama kolom fitur


# Kelas untuk menangani model -> mengelola model machine learning (train, evaluasi, tuning, save)
class LoanModelHandler:
    def __init__(self, X, y):
        self.X = X  # Fitur
        self.y = y  # Target
        self.model = XGBClassifier(n_estimators=100, random_state=42)  # Model awal
        self.best_model = None  # Model setelah tuning

    def split_data(self):
        return train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=0)  # Split data training dan test

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)  # Latih model

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)  # Prediksi
        print(classification_report(y_test, y_pred, target_names=['0', '1']))  # Tampilkan laporan klasifikasi

    def tune_model(self, X_train, y_train):
        # Parameter tuning untuk XGBoost
        params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1]
        }
        grid_search = GridSearchCV(self.model, params, cv=5, n_jobs=-1)  # Grid search
        grid_search.fit(X_train, y_train)  # Latih grid search
        print("Best Params:", grid_search.best_params_)  # Tampilkan parameter terbaik
        self.best_model = grid_search.best_estimator_  # Simpan model terbaik

    def evaluate_best_model(self, X_test, y_test):
        y_pred = self.best_model.predict(X_test)  # Prediksi dengan model terbaik
        print("\nEvaluation After Tuning:")  # Tampilkan laporan
        print(classification_report(y_test, y_pred, target_names=['0', '1']))

    def save_model(self, filename='xgb_final_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.best_model, f)  # Simpan model terbaik ke file


# === Pipeline utama ===
if __name__ == '__main__':
    data_handler = LoanDataHandler("Dataset_A_loan.csv")  # Buat instance pengelola data
    data_handler.load_data()  # Load data
    data_handler.preprocess()  # Preprocessing
    data_handler.save_preprocessors()  # Simpan encoder dan scaler

    model_handler = LoanModelHandler(data_handler.X, data_handler.y)  # Buat instance pengelola model
    X_train, X_test, y_train, y_test = model_handler.split_data()  # Split data

    print("\nBefore Tuning")
    model_handler.train_model(X_train, y_train)  # Latih model
    model_handler.evaluate_model(X_test, y_test)  # Evaluasi sebelum tuning

    print("\nAfter Tuning")
    model_handler.tune_model(X_train, y_train)  # Tuning hyperparameter
    model_handler.evaluate_best_model(X_test, y_test)  # Evaluasi setelah tuning
    model_handler.save_model()  # Simpan model final