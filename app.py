import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

@st.cache_resource
def train_model():
    df = pd.read_excel("Car_sales.xlsx")

    df = df.dropna(subset=["Price_in_thousands"])

    fitur_model = [
        "Engine_size",
        "Horsepower",
        "Wheelbase",
        "Width",
        "Length",
        "Curb_weight",
        "Fuel_capacity",
        "Fuel_efficiency",
        "Power_perf_factor"
    ]

    for col in fitur_model:
        df[col] = df[col].fillna(df[col].median())

    X = df[fitur_model]
    y = df["Price_in_thousands"]

    model = LinearRegression()
    model.fit(X, y)

    return model

model = train_model()
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #ffffff 0%, #f7f0ff 45%, #eadcff 100%);
}

[data-testid="stHeader"] {
    background: rgba(255,255,255,0);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

.hero {
    padding: 34px;
    border-radius: 32px;
    background: rgba(255, 255, 255, 0.70);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(155, 89, 255, 0.22);
    box-shadow: 0 20px 60px rgba(111, 66, 193, 0.16);
    margin-bottom: 28px;
}

.badge {
    display: inline-block;
    padding: 8px 16px;
    background: linear-gradient(135deg, #7b2cff, #b36bff);
    color: white;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 14px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    color: #4b168f;
    margin-bottom: 8px;
}

.hero-desc {
    font-size: 16px;
    color: #6d5a83;
    max-width: 850px;
}

/* Card untuk kolom utama */
[data-testid="column"] {
    background: rgba(255, 255, 255, 0.68);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(155, 89, 255, 0.22);
    border-radius: 30px;
    padding: 28px 30px;
    box-shadow: 0 20px 55px rgba(111, 66, 193, 0.13);
}

/* Mengurangi efek aneh di mobile */
[data-testid="stHorizontalBlock"] {
    gap: 2rem;
}

h2, h3 {
    color: #4b168f !important;
    font-weight: 800 !important;
}

p, label {
    color: #6d5a83 !important;
}

.stNumberInput label {
    color: #4b168f !important;
    font-weight: 600 !important;
}

div.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 18px;
    border: none;
    background: linear-gradient(135deg, #7b2cff, #b36bff);
    color: white;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 14px 30px rgba(123, 44, 255, 0.28);
    transition: 0.25s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    color: white;
    box-shadow: 0 18px 38px rgba(123, 44, 255, 0.36);
}

.result-box {
    text-align: center;
    padding: 36px 24px;
    border-radius: 28px;
    background: linear-gradient(135deg, #7b2cff, #b36bff);
    color: white;
    box-shadow: 0 22px 48px rgba(123, 44, 255, 0.32);
    margin-top: 18px;
    margin-bottom: 24px;
}

.result-label {
    font-size: 15px;
    opacity: 0.9;
    margin-bottom: 10px;
}

.result-price {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 6px;
}

.result-note {
    font-size: 13px;
    opacity: 0.88;
}

.info-box {
    padding: 20px;
    border-radius: 22px;
    background: rgba(247, 240, 255, 0.9);
    border: 1px solid rgba(155, 89, 255, 0.20);
    margin-top: 20px;
}

.info-title {
    color: #4b168f;
    font-weight: 800;
    font-size: 18px;
    margin-bottom: 8px;
}

.info-text {
    color: #6d5a83;
    font-size: 14px;
}

.footer {
    text-align: center;
    margin-top: 28px;
    color: #6d5a83;
    font-size: 14px;
}

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <div class="badge">Data Science Project • Linear Regression</div>
    <div class="hero-title">🚗 Car Price Predictor</div>
    <div class="hero-desc">
        Sistem prediksi harga mobil berbasis Machine Learning untuk membantu perusahaan manufaktur otomotif
        menentukan estimasi harga berdasarkan spesifikasi kendaraan.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# MAIN LAYOUT
# =========================
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.subheader("Input Spesifikasi Mobil")
    st.write("Masukkan spesifikasi kendaraan untuk mendapatkan estimasi harga mobil.")

    engine_size = st.number_input("Engine Size", min_value=0.0, value=2.0, step=0.1)
    horsepower = st.number_input("Horsepower", min_value=0, value=150, step=1)
    wheelbase = st.number_input("Wheelbase", min_value=0.0, value=105.0, step=0.1)
    width = st.number_input("Width", min_value=0.0, value=70.0, step=0.1)
    length = st.number_input("Length", min_value=0.0, value=180.0, step=0.1)
    curb_weight = st.number_input("Curb Weight", min_value=0.0, value=3.0, step=0.1)
    fuel_capacity = st.number_input("Fuel Capacity", min_value=0.0, value=15.0, step=0.1)
    fuel_efficiency = st.number_input("Fuel Efficiency", min_value=0, value=28, step=1)
    power_perf_factor = st.number_input("Power Performance Factor", min_value=0.0, value=80.0, step=0.1)

    st.write("")
    tombol = st.button("Prediksi Harga Mobil")

with col2:
    st.subheader("Hasil Prediksi")
    st.write("Estimasi harga akan muncul setelah tombol prediksi ditekan.")

    if tombol:
        input_data = pd.DataFrame({
            "Engine_size": [engine_size],
            "Horsepower": [horsepower],
            "Wheelbase": [wheelbase],
            "Width": [width],
            "Length": [length],
            "Curb_weight": [curb_weight],
            "Fuel_capacity": [fuel_capacity],
            "Fuel_efficiency": [fuel_efficiency],
            "Power_perf_factor": [power_perf_factor]
        })

        prediksi = model.predict(input_data)[0]

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Perkiraan Harga Mobil</div>
            <div class="result-price">${prediksi:.3f}K</div>
            <div class="result-note">Harga dalam satuan ribuan dollar</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Detail Input")
        st.dataframe(input_data, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            <div class="info-title">Insight</div>
            <div class="info-text">
                Hasil prediksi dihitung berdasarkan model Linear Regression yang telah dilatih
                menggunakan dataset penjualan mobil.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-box">
            <div class="result-label">Estimasi Harga</div>
            <div class="result-price">$ --</div>
            <div class="result-note">Isi data spesifikasi terlebih dahulu</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <div class="info-title">Cara Menggunakan</div>
            <div class="info-text">
                Isi spesifikasi mobil di bagian kiri, lalu klik tombol Prediksi Harga Mobil.
                Sistem akan menampilkan estimasi harga berdasarkan input tersebut.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    Dibuat oleh <b>Piki Alpian</b> • Final Project Data Science • Prediksi Harga Mobil
</div>
""", unsafe_allow_html=True)
