import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

model = YOLO(MODEL_PATH)

st.title("🧵 Fabric Defect Detection")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp:

        image.save(tmp.name, "JPEG")

        results = model(tmp.name)

    annotated = results[0].plot()

    st.image(
        annotated,
        caption="Detection Result",
        use_container_width=True
    )

    if len(results[0].boxes) > 0:

        st.error(
            f"❌ {len(results[0].boxes)} defect(s) detected"
        )

        for box in results[0].boxes:

            defect = model.names[
                int(box.cls[0])
            ]

            conf = float(box.conf[0])

            st.write(
                f"{defect} - {conf:.2%}"
            )

    else:

        st.success(
            "✅ No defects detected"
        )
