import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------- File locations ----------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "heart_model.joblib"
INFO_PATH = BASE_DIR / "models" / "model_info.json"

# ---------- Page setup ----------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")


# ---------- Load the saved model and info (cached) ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_info():
    with open(INFO_PATH) as f:
        return json.load(f)


model = load_model()
info = load_info()
cats = info["categories"]
ranges = info["numeric_ranges"]

LABELS = {
    "age": "Age",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "thalch": "Maximum heart rate",
    "oldpeak": "ST depression (oldpeak)",
}


def pick(label, column, default, help_text):
    """Dropdown built from the categories the model learned."""
    options = cats[column]
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index, help=help_text)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a machine learning model trained on the UCI Heart "
        "Disease dataset (about 920 patients from four hospitals)."
    )
    st.write(f"**Model type:** {info['model_type']}")
    st.write(f"**scikit-learn:** {info['sklearn_version']}")
    st.warning("Learning project only. NOT a medical tool.")

# ---------- Main page ----------
st.title("❤️ Heart Disease Prediction")

with st.expander("How to use this app", expanded=True):
    st.markdown(
        """
1. Fill in every field. Hover over the **?** icons for explanations.
2. Press **Predict**.
3. Read the estimated probability. Results close to 50% mean the model is **unsure**.

Choose **Unknown** in a dropdown if the value was not recorded.
The model was trained with such gaps and can handle them.
        """
    )

with st.form("patient_form"):
    left, right = st.columns(2)

    with left:
        st.subheader("Basic information")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=55,
                              help="Age in whole years.")
        sex = pick("Sex", "sex", "Male", "Biological sex recorded in the dataset.")
        cp = pick("Chest pain type", "cp", "asymptomatic",
                  "'asymptomatic' means no chest pain symptoms. "
                  "Surprisingly, it is linked with higher disease rates in this dataset.")
        trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=1,
                                   max_value=300, value=130,
                                   help="Blood pressure at rest. Typical values: 90 to 180.")
        chol = st.number_input("Cholesterol (mg/dl)", min_value=1, max_value=700, value=230,
                               help="Serum cholesterol. Typical values: 150 to 350.")
        fbs = pick("Fasting blood sugar above 120 mg/dl?", "fbs", "False",
                   "True if the fasting blood sugar was above 120 mg/dl.")

    with right:
        st.subheader("Heart and exercise tests")
        restecg = pick("Resting ECG result", "restecg", "normal",
                       "Result of an electrocardiogram taken at rest.")
        thalch = st.number_input("Maximum heart rate achieved", min_value=30,
                                 max_value=250, value=140,
                                 help="Highest heart rate reached during the exercise test.")
        exang = pick("Exercise-induced chest pain?", "exang", "False",
                     "Did exercise cause chest pain (angina)?")
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=-3.0, max_value=10.0,
                                  value=1.0, step=0.1,
                                  help="How much the ECG's ST segment dropped during exercise "
                                       "compared with rest. 0 means no drop.")
        slope = pick("Slope of peak exercise ST segment", "slope", "flat",
                     "Direction of the ST segment at peak exercise.")

    submitted = st.form_submit_button("Predict", type="primary")

# ---------- Validation and prediction ----------
if submitted:
    numeric_values = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalch": thalch,
        "oldpeak": oldpeak,
    }

    errors = []
    warnings = []

    # Error: combinations that look implausible
    if thalch > (220 - age) + 40:
        errors.append(
            f"A maximum heart rate of {thalch} is implausibly high for age {age}. "
            "Please double-check the value."
        )

    # Warning: values outside what the model learned from
    for col, value in numeric_values.items():
        low, high = ranges[col]
        if value < low or value > high:
            warnings.append(
                f"{LABELS[col]} = {value:g} is outside the range the model was trained on "
                f"({low:g} to {high:g}). The estimate may be less reliable."
            )

    if errors:
        for message in errors:
            st.error(message)
        st.stop()

    for message in warnings:
        st.warning(message)

    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
    }])[info["feature_columns"]]

    prob = float(model.predict_proba(patient)[0, 1])

    with st.container(border=True):
        st.subheader("Result")
        st.metric("Estimated probability of heart disease", f"{prob:.1%}")
        st.progress(prob)

        if prob >= 0.65:
            st.error("The model estimates a HIGHER likelihood of heart disease.")
        elif prob <= 0.35:
            st.success("The model estimates a LOWER likelihood of heart disease.")
        else:
            st.info("The model is UNSURE: the estimate is close to 50%.")

        st.caption("An estimate from a learning project, not a diagnosis. "
                   "Please consult a doctor for any real health concern.")

    with st.expander("Show the values the model received"):
        st.dataframe(patient.astype(str).T.rename(columns={0: "Value"}))