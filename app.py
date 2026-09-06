from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Disease Predictor", page_icon="🩺")

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #f5fbff 0%, #eef7f7 100%);
        }
        .stApp {
            max-width: 1100px;
            margin: 0 auto;
        }
        .result-card {
            background: linear-gradient(135deg, #e8f8f5 0%, #eff9ff 100%);
            border: 1px solid #d7ebe6;
            border-radius: 18px;
            padding: 1.5rem;
            box-shadow: 0 8px 24px rgba(22, 118, 134, 0.08);
        }
        .small-muted {
            color: #4a6570;
            font-size: 0.92rem;
        }
        .prediction-title {
            font-size: 2.2rem;
            font-weight: 700;
        }
        [data-testid="stSidebar"] {
            background: #f4fbff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Training.csv"
MODEL_PATH = BASE_DIR / "disease_model.pkl"


@st.cache_data
def load_training_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


try:
    df = load_training_data()
    model = load_model()
except (FileNotFoundError, OSError, pickle.PickleError, ValueError) as exc:
    st.error(f"The app could not load its model data. Details: {exc}")
    st.stop()

symptoms = [column for column in df.columns if column != "prognosis"]

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🩺 Disease Prediction System")
st.caption("For academic/demo use only. Not a medical diagnosis.")

with st.sidebar:
    st.header("How to use")
    st.write("1. Select the symptoms you are experiencing.")
    st.write("2. Click predict to see the likely disease match.")
    st.write("3. Try 3–6 symptoms for better accuracy.")
    st.markdown("---")
    st.write("This tool is intended for demos and educational experiments only.")

selected_symptoms = st.multiselect(
    "Symptoms",
    options=symptoms,
    help="Select one or more symptoms from the list.",
    key="selected_symptoms",
)

if selected_symptoms:
    st.markdown(
        "<div class='small-muted'>Selected symptoms: {}</div>".format(", ".join(selected_symptoms)),
        unsafe_allow_html=True,
    )
else:
    st.info("No symptoms selected yet. Pick one or more to begin.")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Predict disease", type="primary"):
        if len(selected_symptoms) == 0:
            st.warning("Please select at least one symptom.")
        else:
            feature_vector = [0] * len(symptoms)
            for symptom in selected_symptoms:
                feature_vector[symptoms.index(symptom)] = 1

            input_array = np.array(feature_vector, dtype=int).reshape(1, -1)
            prediction = model.predict(input_array)[0]

            result_data = {
                "symptoms": ", ".join(selected_symptoms),
                "prediction": prediction,
            }

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_array)[0]
                classes = getattr(model, "classes_", [])
                top_matches = sorted(
                    zip(classes, probabilities),
                    key=lambda item: item[1],
                    reverse=True,
                )
                result_data["confidence"] = float(np.max(probabilities))
                result_data["top_matches"] = top_matches
            else:
                result_data["confidence"] = None
                result_data["top_matches"] = []

            st.session_state.history.insert(0, result_data)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="prediction-title">Predicted disease</div>', unsafe_allow_html=True)
            st.success(f"{prediction}")

            if result_data["confidence"] is not None:
                confidence = result_data["confidence"]
                st.metric("Confidence", f"{confidence * 100:.1f}%")
                st.subheader("Top possible matches")
                for disease_name, probability in result_data["top_matches"][:3]:
                    st.write(f"{disease_name}: {probability * 100:.1f}%")
                    st.progress(probability)

                matched_df = pd.DataFrame(
                    result_data["top_matches"],
                    columns=["Disease", "Probability"],
                )
                st.bar_chart(matched_df.set_index("Disease")["Probability"] * 100)
            else:
                st.info("This model does not provide probability scores for the prediction.")

            st.caption(f"Selected symptoms ({len(selected_symptoms)}): {', '.join(selected_symptoms)}")
            st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if st.button("Clear selection"):
        st.session_state["selected_symptoms"] = []

st.markdown("---")
st.caption("Tip: selecting 3–6 symptoms usually gives more useful results.")

if st.session_state.history:
    st.subheader("Prediction history")
    history_df = pd.DataFrame(st.session_state.history)
    history_df["confidence"] = history_df["confidence"].map(
        lambda value: f"{value * 100:.1f}%" if value is not None else "N/A"
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.metric("Total predictions", len(history_df))
    with summary_col2:
        latest_prediction = history_df.iloc[0]["prediction"]
        st.metric("Latest result", latest_prediction)
    with summary_col3:
        top_prediction = history_df["prediction"].mode().iloc[0] if not history_df.empty else "N/A"
        st.metric("Most common", top_prediction)

    st.dataframe(history_df[["symptoms", "prediction", "confidence"]], use_container_width=True)

    export_df = history_df[["symptoms", "prediction", "confidence"]].copy()
    export_df.columns = ["Symptoms", "Predicted Disease", "Confidence"]
    csv_data = export_df.to_csv(index=False).encode("utf-8")

    col_export, col_clear, col_reset = st.columns([1, 1, 1])
    with col_export:
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name="disease_prediction_history.csv",
            mime="text/csv",
        )
    with col_clear:
        if st.button("Clear history"):
            st.session_state.history = []
    with col_reset:
        if st.button("Reset session"):
            st.session_state.history = []
            st.session_state["selected_symptoms"] = []
