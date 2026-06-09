```python
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import pandas as pd

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="AI Fabric Inspection",
    page_icon="🧵",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #0f172a, #1e293b);
}

.title {
    text-align: center;
    color: white;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}

.status-ng {
    background-color: #dc2626;
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.status-ok {
    background-color: #16a34a;
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.footer {
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
    st.error("❌ best.pt model file not found.")
    st.stop()

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -----------------------------------
# HEADER
# -----------------------------------

st.markdown(
    '<div class="title">🧵 AI Fabric Quality Inspection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Automated Fabric Defect Detection using YOLOv8</div>',
    unsafe_allow_html=True
)

# -----------------------------------
# FILE UPLOADER
# -----------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Fabric Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# PREDICTION
# -----------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)

    result = results[0]
    annotated_image = result.plot()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🎯 Detection Result")
        st.image(annotated_image, use_container_width=True)

    st.markdown("---")

    boxes = result.boxes

    # -----------------------------------
    # DEFECT FOUND
    # -----------------------------------

    if len(boxes) > 0:

        st.markdown(
            '<div class="status-ng">❌ FABRIC STATUS : NG (DEFECT DETECTED)</div>',
            unsafe_allow_html=True
        )

        defect_data = []

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            defect_name = model.names[class_id]

            # Rename model class names
            if "not_valid" in defect_name.lower():
                defect_name = "Hole"

            defect_data.append({
                "Defect Type": defect_name,
                "Confidence (%)": round(confidence * 100, 2)
            })

        df = pd.DataFrame(defect_data)

        st.markdown("### 📋 Defect Report")

        colA, colB = st.columns(2)

        with colA:
            st.metric(
                "Total Defects Found",
                len(defect_data)
            )

        with colB:
            st.metric(
                "Average Confidence",
                f"{df['Confidence (%)'].mean():.1f}%"
            )

        st.dataframe(
            df,
            use_container_width=True
        )

    # -----------------------------------
    # NO DEFECT FOUND
    # -----------------------------------

    else:

        st.markdown(
            '<div class="status-ok">✅ FABRIC STATUS : GOOD (NO DEFECT FOUND)</div>',
            unsafe_allow_html=True
        )

        st.balloons()

        st.metric(
            "Detected Defects",
            "0"
        )

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.markdown(
    '<div class="footer">Developed using YOLOv8 and Streamlit for Fabric Quality Inspection</div>',
    unsafe_allow_html=True
)
```
