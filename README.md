# Aplikasi Web Prediksi Persetujuan Pinjaman 🏦

Sebuah aplikasi web yang mudah digunakan, dibangun dengan Streamlit untuk memprediksi keputusan persetujuan pinjaman. Aplikasi ini memanfaatkan model XGBoost yang telah dilatih untuk memberikan prediksi instan berdasarkan data pemohon, membantu mempercepat proses penilaian risiko kredit.



## ✨ Fitur Utama

- **Antarmuka Interaktif**: Tampilan web yang intuitif untuk memudahkan pengguna memasukkan informasi pemohon.
- **Prediksi Instan**: Dapatkan hasil prediksi persetujuan pinjaman ("Approved" atau "Rejected") secara real-time.
- **Didukung Machine Learning**: Menggunakan model klasifikasi XGBoost yang dilatih pada dataset kredit untuk pengambilan keputusan yang akurat.
- **Preprocessing Data**: Termasuk pipeline preprocessing lengkap untuk penskalaan dan encoding input pengguna agar sesuai dengan format yang dibutuhkan model.

---

## 🛠️ Teknologi yang Digunakan

- **Framework**: Streamlit
- **Machine Learning**: Scikit-learn, XGBoost
- **Manajemen Data**: Pandas, NumPy
- **Bahasa**: Python

---

## 🚀 Cara Menjalankan Secara Lokal

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/username-anda/Loan-Approval-Prediction.git](https://github.com/gabriellaclairine/Loan-Approval-Prediction.git)
    cd Loan-Approval-Prediction
    ```

2.  **Buat dan aktifkan virtual environment (disarankan):**
    ```bash
    # Untuk Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Untuk macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install semua library yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Catatan: Anda perlu membuat file `requirements.txt` yang berisi semua paket yang diperlukan seperti streamlit, pandas, scikit-learn, xgboost, dll.)*

4.  **Jalankan aplikasi Streamlit:**
    ```bash
    streamlit run app.py
    ```

---

## 📁 Struktur Repositori
