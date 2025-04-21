# Import library yang dibutuhkan
import pickle
import pandas as pd
import streamlit as st
import warnings

# Nonaktifkan warning
warnings.filterwarnings('ignore')

# --- Fungsi untuk Load Model dan Encoder ---
def load_assets():
    # Load model XGBoost yang sudah dilatih
    with open('xgb_final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Load ordinal encoder
    with open('ordinal_encoder.pkl', 'rb') as f:
        ord_enc = pickle.load(f)
    # Load one-hot encoder
    with open('onehot_encoder.pkl', 'rb') as f:
        onehot_enc = pickle.load(f)
    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    # Load daftar nama fitur (untuk memastikan urutan dan struktur input sesuai dengan saat training)
    with open('feature_list.pkl', 'rb') as f:
        feature_list = pickle.load(f)
    return model, ord_enc, onehot_enc, scaler, feature_list

# --- Fungsi untuk Preprocessing Data Input ---
def preprocess_input(data_df, ord_enc, onehot_encoder, scaler, feature_list):
    # Normalisasi teks untuk gender
    data_df['person_gender'] = data_df['person_gender'].str.lower().str.replace(' ', '')
    data_df['person_gender'] = data_df['person_gender'].replace({'male': 'Male', 'female': 'Female'})
    # Encode kolom boolean
    data_df['previous_loan_defaults_on_file'] = data_df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0}).astype(int)
    # Ordinal encoding untuk pendidikan
    data_df['person_education'] = ord_enc.transform(data_df[['person_education']])

    # One-hot encoding untuk kolom kategori
    onehot_cols = ['person_gender', 'person_home_ownership', 'loan_intent']
    encoded = onehot_encoder.transform(data_df[onehot_cols])
    encoded_df = pd.DataFrame(encoded, columns=onehot_encoder.get_feature_names_out(onehot_cols), index=data_df.index)

    # Gabungkan hasil one-hot ke dataframe
    data_df = data_df.drop(columns=onehot_cols)
    data_df = pd.concat([data_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Scaling data numerik
    num_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
    data_df[num_cols] = scaler.transform(data_df[num_cols])

    # Pastikan semua kolom yang dibutuhkan ada, jika tidak tambahkan dengan nilai 0
    for col in feature_list:
        if col not in data_df.columns:
            data_df[col] = 0

    # Urutkan kolom sesuai dengan urutan saat training
    return data_df[feature_list]

# --- Fungsi untuk Prediksi ---
def predict(model, processed_data):
    prediction = model.predict(processed_data)
    return 'Approved' if prediction[0] == 1 else 'Rejected'

# --- Fungsi Utama untuk UI Streamlit ---
def main():
    # Konfigurasi tampilan halaman
    st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

    # Judul dan deskripsi
    st.title("🏦 Loan Approval Prediction")
    st.markdown("Masukkan data peminjam untuk memprediksi apakah pinjaman akan disetujui atau tidak.")

    # Load semua asset model dan encoder
    model, ord_enc, onehot_enc, scaler, feature_list = load_assets()

    # Form input pengguna
    with st.form("loan_form"):
        person_age = st.slider("Usia", min_value=0, max_value=150, value=25)
        person_gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        person_education = st.selectbox("Pendidikan", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
        person_income = st.number_input("Pendapatan Tahunan", min_value=0.0, value=75000.0)
        person_emp_exp = st.slider("Tahun Pengalaman Kerja", min_value=0, max_value=125, value=5)
        person_home_ownership = st.selectbox("Status Tempat Tinggal", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        loan_amnt = st.number_input("Jumlah Pinjaman", min_value=50.0, value=25000.0)
        loan_intent = st.selectbox("Tujuan Pinjaman", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        loan_int_rate = st.slider("Suku Bunga (%)", min_value=1.0, max_value=25.0, value=15.0)
        cb_person_cred_hist_length = st.slider("Lama Riwayat Kredit (tahun)", 1.0, 40.0, 3.5)
        credit_score = st.slider("Skor Kredit", min_value=300, max_value=850, value=650)
        previous_loan_defaults_on_file = st.selectbox("Pernah Gagal Bayar?", ["Yes", "No"])

        submitted = st.form_submit_button("Prediksi")  # Tombol submit form

        if submitted:
            # Hitung persentase pinjaman terhadap pendapatan
            loan_percent_income = loan_amnt / person_income if person_income != 0 else 0

            # Masukkan data pengguna ke DataFrame
            user_input = pd.DataFrame([{
                'person_age': person_age,
                'person_gender': person_gender,
                'person_education': person_education,
                'person_income': person_income,
                'person_emp_exp': person_emp_exp,
                'person_home_ownership': person_home_ownership,
                'loan_amnt': loan_amnt,
                'loan_intent': loan_intent,
                'loan_int_rate': loan_int_rate,
                'loan_percent_income': loan_percent_income,
                'cb_person_cred_hist_length': cb_person_cred_hist_length,
                'credit_score': credit_score,
                'previous_loan_defaults_on_file': previous_loan_defaults_on_file
            }])

            # Lakukan preprocessing
            processed_input = preprocess_input(user_input, ord_enc, onehot_enc, scaler, feature_list)
            # Prediksi hasil
            result = predict(model, processed_input)

            # Tampilkan hasil prediksi
            st.subheader("Hasil Prediksi")
            if result == "Approved":
                st.success("✅ Disetujui")
                return 'Approved'
            else:
                st.error("❌ Ditolak")
                return 'Rejected'

# --- Jalankan Aplikasi ---
if __name__ == "__main__":
    main()
