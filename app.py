import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Fabric Inspection",
    page_icon="🧵",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}

[data-testid="stFileUploader"] {
    background-color: #1e293b;
    border: 2px dashed #3b82f6;
    border-radius: 15px;
    padding: 20px;
}

.good-status {
    background: #14532d;
    color: #dcfce7;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.ng-status {
    background: #7f1d1d;
    color: #fee2e2;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.result-box {
    background: #1e293b;
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

model = YOLO(MODEL_PATH)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class='main-title'>
🧵 AI Fabric Quality Inspection System
</div>

<div class='sub-title'>
Automated Fabric Defect Detection using YOLOv8
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Fabric Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:

        image.save(tmp.name, "JPEG")

        with st.spinner("🔍 Inspecting Fabric..."):
            results = model(tmp.name)

    annotated_image = results[0].plot()

    # --------------------------------------------------
    # IMAGE DISPLAY
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Fabric")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🎯 Detection Result")
        st.image(annotated_image, use_container_width=True)

    # --------------------------------------------------
    # RESULT SECTION
    # --------------------------------------------------

    boxes = results[0].boxes

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)

    st.subheader("📋 Inspection Report")

    if len(boxes) > 0:

        st.markdown(
            "<div class='ng-status'>❌ FABRIC STATUS : NG (REJECTED)</div>",
            unsafe_allow_html=True
        )

        st.metric("Total Defects Found", len(boxes))

        st.write("### Detected Defects")

        for box in boxes:

            defect_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0]) * 100

            st.write(
                f"🔴 **{defect_name}** | Confidence: **{confidence:.1f}%**"
            )

    else:

        st.markdown(
            "<div class='good-status'>✅ FABRIC STATUS : GOOD (ACCEPTED)</div>",
            unsafe_allow_html=True
        )

        st.success("No defects detected in the fabric.")

        st.balloons()

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Developed using YOLOv8 and Streamlit for Automated Fabric Defect Detection"
)
