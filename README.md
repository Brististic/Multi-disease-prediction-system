# Multi Disease Prediction System

A Streamlit web application that predicts likely diseases based on user-selected symptoms. It uses a trained machine learning model and presents results with confidence scores and prediction history.

Live demo: https://multi-disease-prediction-system-b4hsqdcav2evctqck5zexa.streamlit.app/

## Features
- Symptom-based disease prediction
- Confidence score display
- Top possible disease matches
- Prediction history tracking
- CSV export for results
- Clean, responsive Streamlit interface

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn

## Project Structure
- `app.py` – Streamlit app
- `Training.csv` – training dataset
- `Testing.csv` – test dataset
- `disease_model.pkl` – trained prediction model
- `requirements.txt` – project dependencies

## How to Run Locally
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes
This application is intended for academic and demo use only. It is not a substitute for professional medical diagnosis.

## Disclaimer
The predictions are generated from a machine learning model and should not be treated as medical advice.
