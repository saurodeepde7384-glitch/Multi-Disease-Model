import streamlit as st
import pandas as pd
import joblib
import time

st.set_page_config(
    page_title="Multi Disease Prediction System",
    layout="wide"
)


@st.cache_resource
def load_heart():
    return joblib.load("Models/model_heart_disease.pkl")


@st.cache_resource
def load_diabetes():
    return joblib.load("Models/model_diabetes.pkl")


@st.cache_resource
def load_breast():
    return joblib.load("Models/model_breast_cancer.pkl")


def risk_label(prob):
    if prob < 0.34:
        return "Low Risk"
    elif prob < 0.67:
        return "Moderate Risk"
    else:
        return "High Risk"


def show_risk_badge(prob):
    label = risk_label(prob)
    if label == "Low Risk":
        st.success(label)
    elif label == "Moderate Risk":
        st.warning(label)
    else:
        st.error(label)


def run_prediction_animation():
    """Animated multi-step status shown while a prediction is being made."""
    with st.status("Running prediction...", expanded=True) as status:
        st.write("Validating patient data...")
        time.sleep(0.35)
        st.write("Running model inference...")
        time.sleep(0.35)
        st.write("Preparing report...")
        time.sleep(0.25)
        status.update(label="Analysis complete", state="complete", expanded=False)


def show_result(pred, prob, positive_label, negative_label, summary, animate=False):
    st.divider()
    st.subheader("Prediction Result")

    result_col, prob_col = st.columns([1.3, 1])

    with result_col:
        if pred == 1:
            st.error(positive_label)
        else:
            st.success(negative_label)
            if animate:
                st.balloons()

    with prob_col:
        if prob is not None:
            placeholder = st.empty()

            if animate:
                steps = 25
                for i in range(steps + 1):
                    current = prob * i / steps
                    with placeholder.container():
                        st.metric("Model Probability", f"{current * 100:.2f}%")
                        st.progress(current)
                    time.sleep(0.015)
            else:
                with placeholder.container():
                    st.metric("Model Probability", f"{prob * 100:.2f}%")
                    st.progress(prob)

            show_risk_badge(prob)

    with st.expander("View Patient Input Summary"):
        st.dataframe(summary, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## MDPS")
    st.caption("Multi Disease Prediction System")
    st.divider()

    disease = st.radio(
        "Select a prediction model",
        ["Heart Disease", "Diabetes", "Breast Cancer"],
        key="disease_choice"
    )

    st.divider()
    st.caption(
        "This tool provides an AI-generated risk estimate and is not a "
        "substitute for professional medical diagnosis. Always consult "
        "a qualified healthcare provider."
    )

st.title("Multi Disease Prediction System")
st.caption("AI-Powered Health Risk Prediction")
st.divider()

st.header(disease)
st.caption(f"{disease} risk assessment based on patient clinical data.")
st.write("")


# ---------------------------------------------------------------------------
# Heart Disease
# ---------------------------------------------------------------------------
if disease == "Heart Disease":

    with st.spinner("Loading Heart Disease Model..."):
        model = load_heart()

    tab_predict, tab_info = st.tabs(["Predict", "Model Info"])

    with tab_info:
        c1, c2 = st.columns(2)
        c1.metric("Algorithm", "Decision Tree")
        c2.metric("Test Accuracy", "97.73%")
        st.caption(
            "Trained on clinical vitals: age, sex, resting heart rate, "
            "systolic/diastolic blood pressure, blood glucose, CK-MB and "
            "Troponin levels."
        )

    with tab_predict:

        with st.container(border=True):
            st.subheader("Patient Information")
            st.caption("Enter the patient's clinical information below.")

            with st.form("heart_form"):

                col1, col2 = st.columns(2)

                with col1:
                    age = st.number_input("Age (years)", 1, 120, 30, key="heart_age")
                    gender = st.selectbox("Biological Sex", ["Male", "Female"], key="heart_gender")
                    heart_rate = st.number_input("Resting Heart Rate (bpm)", 30, 220, 72, key="heart_hr")
                    systolic = st.number_input("Systolic Blood Pressure (mmHg)", 50, 250, 120, key="heart_sys")

                with col2:
                    diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", 30, 150, 80, key="heart_dia")
                    blood_sugar = st.number_input("Blood Glucose Level (mg/dL)", 50, 500, 100, key="heart_bs")
                    ck_mb = st.number_input("CK-MB Level (ng/mL)", 0.0, 500.0, 2.0, key="heart_ckmb")
                    troponin = st.number_input("Troponin Level (ng/mL)", 0.0, 20.0, 0.01,step=0.001, format="%.3f", key="heart_trop")

                submitted = st.form_submit_button("Predict Heart Disease", use_container_width=True)

        if submitted:

            gender_value = 1 if gender == "Male" else 0

            # Column names must match Heart_Attack_Data.csv exactly
            # (the notebook only strips whitespace, case is unchanged).
            # Gender is numeric (0/1) in the source data, so the pipeline
            # treats it as a numerical feature, not a one-hot category.
            data = pd.DataFrame({
                "Age": [age],
                "Gender": [gender_value],
                "Heart_rate": [heart_rate],
                "Systolic_blood_pressure": [systolic],
                "Diastolic_blood_pressure": [diastolic],
                "Blood_sugar": [blood_sugar],
                "CK-MB": [ck_mb],
                "Troponin": [troponin]
            })

            run_prediction_animation()

            prediction = model.predict(data)[0]
            probability = model.predict_proba(data)[0][1] if hasattr(model, "predict_proba") else None

            st.toast("Prediction complete!")

            summary = pd.DataFrame({
                "Parameter": [
                    "Age", "Biological Sex", "Resting Heart Rate",
                    "Systolic Blood Pressure", "Diastolic Blood Pressure",
                    "Blood Glucose Level", "CK-MB Level", "Troponin Level"
                ],
                "Value": [
                    f"{age} years", gender, f"{heart_rate} bpm",
                    f"{systolic} mmHg", f"{diastolic} mmHg",
                    f"{blood_sugar} mg/dL", f"{ck_mb} ng/mL", f"{troponin} ng/mL"
                ]
            })

            show_result(prediction, probability, "Heart Disease Detected", "No Heart Disease Detected", summary, animate=True)
            st.session_state["heart_result"] = (prediction, probability, summary)

        elif "heart_result" in st.session_state:
            pred, prob, summary = st.session_state["heart_result"]
            show_result(pred, prob, "Heart Disease Detected", "No Heart Disease Detected", summary, animate=False)


# ---------------------------------------------------------------------------
# Diabetes
# ---------------------------------------------------------------------------
elif disease == "Diabetes":

    with st.spinner("Loading Diabetes Model..."):
        model = load_diabetes()

    tab_predict, tab_info = st.tabs(["Predict", "Model Info"])

    with tab_info:
        c1, c2 = st.columns(2)
        c1.metric("Algorithm", "Gradient Boosting")
        c2.metric("Test Accuracy", "92.00%")
        st.caption(
            "Trained on age, sex, BMI, blood pressure, glucose, HbA1c, "
            "insulin, physical activity, family history and smoking status."
        )

    with tab_predict:

        with st.container(border=True):
            st.subheader("Patient Information")
            st.caption("Enter the patient's clinical information below.")

            with st.form("diabetes_form"):

                col1, col2 = st.columns(2)

                with col1:
                    age = st.number_input("Age (years)", 18, 80, 40, key="diab_age")
                    gender = st.selectbox("Biological Sex", ["Male", "Female"], key="diab_gender")
                    bmi = st.number_input("Body Mass Index (kg/m²)", 18.0, 39.9, 25.0, key="diab_bmi")
                    bp = st.number_input("Blood Pressure (mmHg)", 90, 180, 120, key="diab_bp")
                    glucose = st.number_input("Blood Glucose Level (mg/dL)", 70, 220, 100, key="diab_glucose")

                with col2:
                    hba1c = st.number_input("HbA1c Level (%)", 4.5, 11.5, 5.5, key="diab_hba1c")
                    insulin = st.number_input("Insulin Level (µIU/mL)", 20, 300, 100, key="diab_insulin")
                    activity = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"], key="diab_activity")
                    family = st.selectbox("Family History of Diabetes", ["No", "Yes"], key="diab_family")
                    smoking = st.selectbox("Smoking Status", ["No", "Yes"], key="diab_smoking")

                submitted = st.form_submit_button("Predict Diabetes", use_container_width=True)

        if submitted:

            # Gender, PhysicalActivity, FamilyHistory and Smoking were one-hot
            # encoded from these exact raw strings during training, so they are
            # passed through as-is rather than converted to integers.
            data = pd.DataFrame({
                "Age": [age],
                "Gender": [gender],
                "BMI": [bmi],
                "BloodPressure": [bp],
                "Glucose": [glucose],
                "HbA1c": [hba1c],
                "Insulin": [insulin],
                "PhysicalActivity": [activity],
                "FamilyHistory": [family],
                "Smoking": [smoking]
            })

            run_prediction_animation()

            prediction = model.predict(data)[0]
            probability = model.predict_proba(data)[0][1] if hasattr(model, "predict_proba") else None

            st.toast("Prediction complete!")

            summary = pd.DataFrame({
                "Parameter": [
                    "Age", "Biological Sex", "Body Mass Index", "Blood Pressure",
                    "Blood Glucose Level", "HbA1c Level", "Insulin Level",
                    "Physical Activity", "Family History of Diabetes", "Smoking Status"
                ],
                "Value": [
                    f"{age} years", gender, f"{bmi} kg/m²", f"{bp} mmHg",
                    f"{glucose} mg/dL", f"{hba1c}%", f"{insulin} µIU/mL",
                    activity, family, smoking
                ]
            })

            show_result(prediction, probability, "Diabetes Detected", "No Diabetes Detected", summary, animate=True)
            st.session_state["diabetes_result"] = (prediction, probability, summary)

        elif "diabetes_result" in st.session_state:
            pred, prob, summary = st.session_state["diabetes_result"]
            show_result(pred, prob, "Diabetes Detected", "No Diabetes Detected", summary, animate=False)


# ---------------------------------------------------------------------------
# Breast Cancer
# ---------------------------------------------------------------------------
elif disease == "Breast Cancer":

    with st.spinner("Loading Breast Cancer Model..."):
        model = load_breast()

    tab_predict, tab_info = st.tabs(["Predict", "Model Info"])

    with tab_info:
        c1, c2, c3 = st.columns(3)
        c1.metric("Algorithm", "Decision Tree")
        c2.metric("Test Accuracy", "99.60%")
        c3.metric("Population", "Female")
        st.caption(
            "Trained on age, family history, BMI, smoking, alcohol use, "
            "physical activity, breastfeeding history, hormone therapy, "
            "menopausal status, tumor size and palpable lump presence."
        )

    with tab_predict:

        with st.container(border=True):
            st.subheader("Patient Information")
            st.caption("Enter the patient's clinical information below.")

            with st.form("breast_form"):

                gender = "Female"

                col1, col2 = st.columns(2)

                with col1:
                    age = st.number_input("Age (years)", 18, 90, 40, key="bc_age")
                    family = st.selectbox("Family History of Breast Cancer", ["No", "Yes"], key="bc_family")
                    bmi = st.number_input("Body Mass Index (kg/m²)", 15.0, 45.0, 24.0, key="bc_bmi")
                    smoking = st.selectbox("Smoking Status", ["No", "Yes"], key="bc_smoking")
                    alcohol = st.selectbox("Alcohol Consumption", ["Low", "Moderate", "High"], key="bc_alcohol")
                    activity = st.slider("Physical Activity (hours/week)", 0.0, 20.0, 5.0, 0.5, key="bc_activity")

                with col2:
                    breastfeeding = st.selectbox("History of Breastfeeding", ["No", "Yes"], key="bc_breastfeeding")
                    hormone = st.selectbox("Hormone Replacement Therapy", ["No", "Yes"], key="bc_hormone")
                    menopause = st.selectbox("Menopausal Status", ["No", "Yes"], key="bc_menopause")
                    tumor = st.slider("Tumor Size (mm)", 0.0, 100.0, 10.0, 0.5, key="bc_tumor")
                    lump = st.selectbox("Palpable Breast Lump", ["No", "Yes"], key="bc_lump")

                submitted = st.form_submit_button("Predict Breast Cancer", use_container_width=True)

        if submitted:

            # Gender, Family_History, Smoking, Alcohol, Breastfeeding,
            # Hormone_Therapy, Menopause and Lump_Present were one-hot encoded
            # from these exact raw strings during training (Alcohol has three
            # categories: Low / Moderate / High), so raw text is used as-is.
            data = pd.DataFrame({
                "Age": [age],
                "Gender": [gender],
                "Family_History": [family],
                "BMI": [bmi],
                "Smoking": [smoking],
                "Alcohol": [alcohol],
                "Physical_Activity_hrs": [activity],
                "Breastfeeding": [breastfeeding],
                "Hormone_Therapy": [hormone],
                "Menopause": [menopause],
                "Tumor_Size_mm": [tumor],
                "Lump_Present": [lump]
            })

            run_prediction_animation()

            prediction = model.predict(data)[0]
            probability = model.predict_proba(data)[0][1] if hasattr(model, "predict_proba") else None

            st.toast("Prediction complete!")

            summary = pd.DataFrame({
                "Parameter": [
                    "Age", "Biological Sex", "Family History of Breast Cancer",
                    "Body Mass Index", "Smoking Status", "Alcohol Consumption",
                    "Physical Activity", "History of Breastfeeding",
                    "Hormone Replacement Therapy", "Menopausal Status",
                    "Tumor Size", "Palpable Breast Lump"
                ],
                "Value": [
                    f"{age} years", "Female", family, f"{bmi} kg/m²", smoking, alcohol,
                    f"{activity} hours/week", breastfeeding, hormone, menopause,
                    f"{tumor} mm", lump
                ]
            })

            show_result(prediction, probability, "Breast Cancer Detected", "No Breast Cancer Detected", summary, animate=True)
            st.session_state["breast_result"] = (prediction, probability, summary)

        elif "breast_result" in st.session_state:
            pred, prob, summary = st.session_state["breast_result"]
            show_result(pred, prob, "Breast Cancer Detected", "No Breast Cancer Detected", summary, animate=False)