import pathlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = pathlib.Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "model.pkl"

# Feature columns taken from the notebook's data.info()
FEATURE_COLUMNS = [
    "Ca_393.3",
    "Ca_396.8",
    "Ca_422.6",
    "Mg_279.5",
    "Mg_280.3",
    "Mg_285.2",
    "K_766.5",
    "K_769.9",
    "Na_589.0",
    "Na_589.6",
    "Fe_248.3",
    "Fe_373.5",
    "Fe_404.5",
    "C_247.8",
    "H_656.3",
    "O_777.0",
    "Peak_Intensity",
    "Max_Intensity",
    "Mean_Intensity",
    "Std_Intensity",
    "Mg_over_Ca",
    "K_over_Na",
    "Fe_over_C",
    "AUC_200_300",
    "AUC_300_500",
    "AUC_500_900",
    "Centroid",
    "FWHM",
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "Model file 'model.pkl' not found. "
            "Run `python train_model.py` locally (with the CSV in this folder) "
            "to create it, then redeploy."
        )
        return None
    model = joblib.load(MODEL_PATH)
    return model


def predict_single(model, features_dict):
    df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)
    preds = model.predict(df)
    return preds[0]


def main():
    st.title("Rose Species Classification (LIBS)")
    st.write(
        "This app uses **your trained Support Vector Machine (SVM) model** based on the "
        "`roses_LIBS_50000.csv` dataset. You only provide the LIBS feature values here; "
        "no training or CSV upload happens from the GUI."
    )

    model = load_model()
    if model is None:
        st.stop()

    st.subheader("Enter LIBS Features")
    cols = st.columns(3)
    inputs = {}

    # For simplicity, use generic ranges. You can refine these based on your data's statistics.
    for idx, feature in enumerate(FEATURE_COLUMNS):
        col = cols[idx % 3]
        # Using number_input; defaults can be 0.0, user can paste real values.
        inputs[feature] = col.number_input(feature, value=0.0, format="%.4f")

    if st.button("Predict Class", type="primary"):
        try:
            pred_class = predict_single(model, inputs)
            st.success(f"Predicted Class: **{pred_class}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()


