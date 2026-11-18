import streamlit as st

from PIL import Image

from src.predict import SteelDefectPredictor


st.set_page_config(
    page_title="Steel Defect Classifier",
    page_icon="🔍",
    layout="centered"
)


st.title("Steel Surface Defect Classifier")

st.write(
    "Upload a steel surface image "
    "to predict its defect category."
)


@st.cache_resource
def load_predictor():

    return SteelDefectPredictor(
        "models/resnet18_finetuned.pth"
    )


predictor = load_predictor()


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded steel image",
        width="stretch"
    )

    if st.button("Predict defect"):

        result = predictor.predict(
            uploaded_file
        )

        st.success(
            f"Prediction: {result['predicted_class']}"
        )

        st.metric(
            "Model confidence",
            f"{result['confidence']:.2%}"
        )

        st.subheader("Class probabilities")

        st.bar_chart(
            result["probabilities"]
        )