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

""", unsafe_allow_html=True)

# -----------------------------------

# LOAD MODEL

# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
st.error("❌ best.pt model file not found.")
st.stop()

model = YOLO(MODEL_PATH)

# -----------------------------------

# HEADER

# -----------------------------------

st.markdown(
"🧵 AI Fabric Quality Inspection System",
unsafe_allow_html=True
)

st.markdown(
"Automated Fabric Defect Detection using YOLOv8",
unsafe_allow_html=True
)

# -----------------------------------

# FILE UPLOADER

# -----------------------------------

uploaded_file = st.file_uploader(
"📤 Upload Fabric Image",
type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

```
image = Image.open(uploaded_file).convert("RGB")

with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    image.save(tmp.name)
    results = model(tmp.name)

annotated_image = results[0].plot()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Original Image")
    st.image(image, use_container_width=True)

with col2:
    st.subheader("🎯 Detection Result")
    st.image(annotated_image, use_container_width=True)

boxes = results[0].boxes

st.markdown("---")

# -----------------------------------
# DEFECT FOUND
# -----------------------------------
if len(boxes) > 0:

    st.markdown(
        "<div class='status-ng'>❌ FABRIC STATUS : NG (DEFECT DETECTED)</div>",
        unsafe_allow_html=True
    )

    defect_data = []

    for box in boxes:

        defect_name = model.names[int(box.cls[0])]

        # Rename class label
        if defect_name == "Not_valid_Defect":
            defect_name = "Hole"

        confidence = float(box.conf[0])

        defect_data.append({
            "Defect Type": defect_name,
            "Confidence (%)": round(confidence * 100, 2)
        })

    df = pd.DataFrame(defect_data)

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Total Defects Found", len(defect_data))

    with c2:
        st.metric(
            "Average Confidence",
            f"{df['Confidence (%)'].mean():.1f}%"
        )

    st.subheader("📋 Defect Report")
    st.dataframe(df, use_container_width=True)

# -----------------------------------
# NO DEFECT FOUND
# -----------------------------------
else:

    st.markdown(
        "<div class='status-ok'>✅ FABRIC STATUS : OK (NO DEFECT FOUND)</div>",
        unsafe_allow_html=True
    )

    st.balloons()

    st.metric(
        "Detected Defects",
        "0"
    )
```

# -----------------------------------

# FOOTER

# -----------------------------------

st.markdown("---")

st.markdown(
"""

Developed by Team | YOLOv8 Fabric Defect Detection

""",
unsafe_allow_html=True
)
