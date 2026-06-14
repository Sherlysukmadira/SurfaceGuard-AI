import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="SurfaceGuard AI",
    page_icon="🔍",
    layout="wide"
)



# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "runs/classify/train-2/weights/best.pt"

model = YOLO(MODEL_PATH)

# ==================================================
# HISTORY FILE
# ==================================================

HISTORY_FILE = "history.csv"

if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(
        columns=[
            "Datetime",
            "Prediction",
            "Confidence"
        ]
    ).to_csv(HISTORY_FILE, index=False)

# ==================================================
# DEFECT COLORS
# ==================================================

defect_colors = {
    "crack": "#FF0000",      # merah
    "hole": "#800000",       # maroon
    "rust": "#FFA500",       # orange
    "scratch": "#FFD700",    # kuning
    "normal": "#00AA00"      # hijau
}

recommendations = {
    "crack": "Segera lakukan inspeksi lanjutan karena retakan dapat mengurangi kekuatan material.",
    "hole": "Periksa proses manufaktur karena lubang dapat menyebabkan kegagalan produk.",
    "rust": "Lakukan pembersihan dan perlindungan anti-korosi.",
    "scratch": "Evaluasi proses penanganan material untuk mengurangi goresan.",
    "normal": "Permukaan logam berada dalam kondisi baik dan tidak memerlukan tindakan khusus."
}
st.markdown("""
<style>

/* Background utama */
.stApp {
    background: #0F172A;
}

/* Header Streamlit */
[data-testid="stHeader"] {
    background: #0F172A;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1E3A8A,
        #172554
    );
}

/* Semua teks */
h1, h2, h3, h4, h5, h6,
p, span, div, label {
    color: white !important;
}

/* Radio button */
.stRadio label {
    color: white !important;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background-color: #1E293B;
    border-radius: 12px;
    padding: 15px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #1E293B;
}

/* Metric */
[data-testid="metric-container"] {
    background-color: #1E293B;
    border-radius: 12px;
    padding: 10px;
}

/* Tombol */
.stButton button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
}

/* Hilangkan garis putih atas */
header {
    background: transparent !important;
}

/* Main container */
.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Upload box */
[data-testid="stFileUploader"] {
    background-color: #1E293B !important;
    border: 2px dashed #3B82F6 !important;
    border-radius: 15px !important;
    padding: 15px !important;
}

/* Area drag & drop */
[data-testid="stFileUploaderDropzone"] {
    background-color: #0F172A !important;
}

/* Tulisan upload */
[data-testid="stFileUploaderDropzone"] * {
    color: white !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #1E293B;
    border-radius: 15px;
    padding: 10px;
}

/* Area drag & drop */
[data-testid="stFileUploaderDropzone"] {
    background-color: #0F172A !important;
    border: 2px dashed #3B82F6 !important;
    border-radius: 12px !important;
}

/* Tombol Browse Files */
[data-testid="stFileUploaderDropzone"] button {
    background-color: #2563EB !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
}

/* Hover tombol */
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #1D4ED8 !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🔍 SurfaceGuard AI")

page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "🔍 Prediction",
        "📊 Dashboard"
    ]
)

# ==================================================
# HOME
# ==================================================

if page == "🏠 Home":

    st.markdown("""
    <div style="
    background: linear-gradient(135deg,#0f172a,#1e3a8a);
    padding:30px;
    border-radius:15px;
    text-align:center;
    ">
    <h1 style="color:white;">
    🔍 SurfaceGuard AI
    </h1>

    <h3 style="color:#cbd5e1;">
    AI-Based Metal Surface Defect Classification System
    </h3>

    <p style="color:white;">
    Automatic Detection of Crack, Hole, Rust, Scratch, and Normal Surface Conditions
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader(
        "AI-Based Metal Surface Defect Classification System"
    )

    # st.markdown("---")

    st.write("""
    SurfaceGuard AI merupakan sistem berbasis Artificial Intelligence
    yang digunakan untuk mengidentifikasi cacat pada permukaan logam
    secara otomatis menggunakan model YOLOv8 Classification.
    """)

    st.markdown("### 🎯 Kegunaan")

    st.markdown("""
    - Membantu proses quality control industri
    - Mengurangi human error
    - Mempercepat inspeksi produk
    - Mendukung otomatisasi manufaktur
    """)

    st.markdown("### 🔍 Jenis Defect")
   

    # Menggunakan HTML table custom agar menyatu dengan tema dark mode
    st.markdown("""
    <div style="overflow-x:auto;">
        <table style="width:100%; border-collapse: collapse; background-color: #1E293B; border-radius: 10px; overflow: hidden; color: white;">
            <thead>
                <tr style="background-color: #1E3A8A; text-align: left;">
                    <th style="padding: 12px 15px; font-weight: bold; border-bottom: 2px solid #3B82F6;">Defect</th>
                    <th style="padding: 12px 15px; font-weight: bold; border-bottom: 2px solid #3B82F6;">Description</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 12px 15px; font-weight: bold; color: #FFD700;">Crack</td>
                    <td style="padding: 12px 15px; color: #E2E8F0;">Retakan pada permukaan logam</td>
                </tr>
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 12px 15px; font-weight: bold; color: #FFD700;">Hole</td>
                    <td style="padding: 12px 15px; color: #E2E8F0;">Lubang pada material</td>
                </tr>
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 12px 15px; font-weight: bold; color: #FFD700;">Rust</td>
                    <td style="padding: 12px 15px; color: #E2E8F0;">Korosi/Karat</td>
                </tr>
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 12px 15px; font-weight: bold; color: #FFD700;">Scratch</td>
                    <td style="padding: 12px 15px; color: #E2E8F0;">Goresan permukaan</td>
                </tr>
                <tr>
                    <td style="padding: 12px 15px; font-weight: bold; color: #00AA00;">Normal</td>
                    <td style="padding: 12px 15px; color: #E2E8F0;">Tidak ditemukan cacat</td>
                </tr>
            </tbody>
        </table>
    </div>
    <br>
    """, unsafe_allow_html=True)

    defect_df = pd.DataFrame({
        "Defect": [
            "Crack",
            "Hole",
            "Rust",
            "Scratch",
            "Normal"
        ],
        "Description": [
            "Retakan pada permukaan logam",
            "Lubang pada material",
            "Korosi/Karat",
            "Goresan permukaan",
            "Tidak ditemukan cacat"
        ]
    })

    st.dataframe(
        defect_df,
        use_container_width=True
    )

    st.markdown("### 📖 Cara Penggunaan")

    st.markdown("""
    1. Masuk ke menu Prediction
    2. Upload gambar atau gunakan kamera
    3. Klik Predict
    4. Lihat hasil klasifikasi
    """)

    st.markdown("---")
    st.caption("Developed by Sherly Sukmadira Putri")

# ==================================================
# PREDICTION
# ==================================================

elif page == "🔍 Prediction":

    st.title("🔍 Surface Defect Prediction")

    input_type = st.radio(
        "Choose Input Method",
        [
            "Upload Image",
            "Use Camera"
        ]
    )

    uploaded_file = None

    if input_type == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"]
        )

    else:
        uploaded_file = st.camera_input(
            "Take a Picture"
        )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        temp_path = "temp.jpg"
        image.save(temp_path)

        with st.spinner("Predicting..."):

            results = model.predict(
                source=temp_path,
                imgsz=256
            )

        probs = results[0].probs
        names = results[0].names

        top1 = probs.top1
        confidence = probs.top1conf.item()

        predicted_class = names[top1]

        descriptions = {
            "crack":
            "Retakan pada permukaan logam yang dapat menurunkan kekuatan material.",

            "hole":
            "Lubang pada permukaan logam yang dapat memengaruhi kualitas produk.",

            "rust":
            "Korosi atau karat akibat oksidasi pada logam.",

            "scratch":
            "Goresan pada permukaan material.",

            "normal":
            "Tidak ditemukan cacat pada permukaan logam."
        }

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("Prediction Result")

            color = defect_colors.get(
                predicted_class.lower(),
                "#FFFFFF"
            )

            st.markdown(
                f"""
                <h2 style='color:{color};'>
                Defect: {predicted_class}
                </h2>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <p style='color:{color};
                        font-size:18px;
                        font-weight:bold;'>
                Confidence Score: {confidence*100:.2f}%
                </p>
                """,
                unsafe_allow_html=True
            )

            if predicted_class.lower() in descriptions:
                st.markdown(
                    f"""
                    <div style="
                        border-left:5px solid {color};
                        padding:10px;
                        background-color:{color}20;
                        border-radius:10px;">
                        <b>Description</b><br>
                        {descriptions[predicted_class.lower()]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("<br>", unsafe_allow_html=True)

            # st.markdown(
            #     f"""
            #     <div style="
            #         border-left:5px solid {color};
            #         padding:10px;
            #         background-color:#fafafa;
            #         border-radius:10px;">
            #         <b>Recommendation</b><br>
            #         {recommendations[predicted_class.lower()]}
            #     </div>
            #     """,
            #     unsafe_allow_html=True
            # )

        # ==========================================
        # SAVE HISTORY
        # ==========================================

        history = pd.read_csv(HISTORY_FILE)

        new_row = pd.DataFrame({
            "Datetime": [
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ],
            "Prediction": [
                predicted_class
            ],
            "Confidence": [
                round(confidence * 100, 2)
            ]
        })

        history = pd.concat(
            [history, new_row],
            ignore_index=True
        )

        history.to_csv(
            HISTORY_FILE,
            index=False
        )

# ==================================================
# DASHBOARD
# ==================================================

elif page == "📊 Dashboard":

    st.title("📊 Prediction Dashboard")

    history = pd.read_csv(HISTORY_FILE)

    if len(history) == 0:

        st.warning(
            "Belum ada data prediksi."
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Predictions",
                len(history)
            )

        with col2:

            avg_conf = history[
                "Confidence"
            ].mean()

            st.metric(
                "Average Confidence",
                f"{avg_conf:.2f}%"
            )

        with col3:

            st.metric(
            "Model Classes",
            "5 Defects"
        )

        st.markdown("---")

        st.subheader(
            "Defect Distribution"
        )

        fig = px.pie(
            history,
            names="Prediction",
            title="Prediction Distribution",
            color="Prediction",
            color_discrete_map={
                "crack": "#FF0000",
                "hole": "#800000",
                "rust": "#FFA500",
                "scratch": "#FFD700",
                "normal": "#00AA00",

                "Crack": "#FF0000",
                "Hole": "#800000",
                "Rust": "#FFA500",
                "Scratch": "#FFD700",
                "Normal": "#00AA00"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader(
            "Prediction History"
        )

        st.dataframe(
            history,
            use_container_width=True
        )

    st.markdown("---")
    st.caption("Developed by Sherly Sukmadira Putri")