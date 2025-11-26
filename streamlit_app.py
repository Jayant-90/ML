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
        "This app uses a Support Vector Machine (SVM) trained on LIBS spectral features "
        "to classify rose samples into species (Class_1 ... Class_5)."
    )

    model = load_model()
    if model is None:
        st.stop()

    tab_manual, tab_batch = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

    with tab_manual:
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

    with tab_batch:
        st.subheader("Upload CSV for Batch Prediction")
        st.write(
            "Upload a CSV file with the same feature columns as the training data "
            "(all columns except `Class`)."
        )
        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
                if missing_cols:
                    st.error(
                        "The uploaded file is missing these required columns: "
                        + ", ".join(missing_cols)
                    )
                else:
                    preds = model.predict(df[FEATURE_COLUMNS])
                    result_df = df.copy()
                    result_df["Predicted_Class"] = preds

                    st.write("Preview of predictions:")
                    st.dataframe(result_df.head())

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Predictions as CSV",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error processing file: {e}")


if __name__ == "__main__":
    main()


