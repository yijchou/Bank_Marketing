import streamlit as st
import pandas as pd
import logging
from joblib import load


# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_age_input(age_input):
    try:
        age = int(age_input)
        if age < 0:
            st.warning("Age cannot be negative. Please enter a valid age.")
            age = 27  # Greatest number of people among all ages (mode)
    except ValueError:
        if age_input:
            st.warning("Please enter a valid number for age.")
        age = 27

    return age


def process_job(job_input):
    job_map = {
        "Administrator": "admin.",
        "Blue Collar": "blue-collar",
        "Entrepreneur": "entrepreneur",
        "Housemaid": "housemaid",
        "Management": "management",
        "Retired": "retired",
        "Self Employed": "self-employed",
        "Services": "services",
        "Student": "student",
        "Technician": "technician",
        "Unemployed": "unemployed",
        "Unknown": "unknown"
    }

    if job_input not in job_map:
        return "unknown"
    else:
        return job_map[job_input]


def process_marital_status(marital_status_input):
    marital_map = {
        "Divorced": "divorced",
        "Married": "married",
        "Single": "single",
        "Unknown": "unknown"
    }

    if marital_status_input not in marital_map:
        return "unknown"
    else:
        return marital_map[marital_status_input]


def process_education_status(education_status_input):
    education_map = {
        "4 Years of Basic Education (Quatro Anos de Ensino Básico)": "basic.4y",
        "6 Years of Basic Education (Seis Anos de Ensino Básico)": "basic.6y",
        "9 Years of Basic Education (Nove Anos de Ensino Básico)": "basic.9y",
        "Secondary Education (Ensino Secundário)": "high.school",
        "Illiterate (Analfabeto)": "illiterate",
        "Professional Course (Curso Profissional)": "professional.course",
        "University Degree (Grau Universitário)": "university.degree",
        "Unknown": "unknown"
    }

    if education_status_input not in education_map:
        return "unknown"
    else:
        return education_map[education_status_input]


def process_default_status(default_status_input):
    default_map = {
        "Yes": "yes",
        "No": "no",
        "Unknown": "unknown"
    }

    if default_status_input not in default_map:
        return "unknown"
    else:
        return default_map[default_status_input]


def process_housing_status(housing_status_input):
    housing_map = {
        "Yes": "yes",
        "No": "no",
        "Unknown": "unknown"
    }

    if housing_status_input not in housing_map:
        return "unknown"
    else:
        return housing_map[housing_status_input]


def process_loan_status(loan_status_input):
    loan_map = {
        "Yes": "yes",
        "No": "no",
        "Unknown": "unknown"
    }

    if loan_status_input not in loan_map:
        return "unknown"
    else:
        return loan_map[loan_status_input]


def process_contact(contact_type_input):
    contact_type_map = {
        "Cellular": "cellular",
        "Telephone": "telephone",
    }

    if contact_type_input not in contact_type_map:
        return "unknown"
    else:
        return contact_type_map[contact_type_input]


def process_month(month_input):
    month_map = {
        "January": "jan",
        "February": "feb",
        "March": "mar",
        "April": "apr",
        "May": "may",
        "June": "jun",
        "July": "jul",
        "August": "aug",
        "September": "sep",
        "October": "oct",
        "November": "nov",
        "December": "dec"
    }

    if month_input not in month_map:
        return "unknown"
    else:
        return month_map[month_input]


def process_day_of_week(day_of_week_input):
    weekday_map = {
        "Monday": "mon",
        "Tuesday": "tue",
        "Wednesday": "wed",
        "Thursday": "thu",
        "Friday": "fri",
    }

    if day_of_week_input not in weekday_map:
        return "unknown"
    else:
        return weekday_map[day_of_week_input]


def process_duration_input(duration_input):
    try:
        duration = int(duration_input)
        if duration < 0:
            st.warning("Duration cannot be negative. Please enter a valid duration.")
            duration = 0
    except ValueError:
        if duration_input:
            st.warning("Please enter a valid number for duration.")
        duration = 27

    return duration


def process_number_of_contact(number_of_contact_input):
    try:
        number_of_contact = int(number_of_contact_input)
        if number_of_contact < 0:
            st.warning("Number of Contact cannot be negative. Please enter a valid number.")
            number_of_contact = 0
    except ValueError:
        if number_of_contact_input:
            st.warning("Please enter a valid number for Number of Contact.")
        number_of_contact = 0

    return number_of_contact


def process_previous_outcome(previous_outcome_input):
    outcome_map = {
        "Failure": "failure",
        "Non-Existent": "nonexistent",
        "Success": "success",
    }

    if previous_outcome_input not in outcome_map:
        return "unknown"
    else:
        return outcome_map[previous_outcome_input]


def predict_single(input_df):
    model_path = f"../models/{input_model}.joblib"
    model = load(model_path)

    prediction = model.predict(input_df)
    predicted_label = "Subscribe" if prediction[0] == 1 else "Not Subscribe"
    probability = model.predict_proba(input_df)[:, 1]

    input_df["predicted_label"] = predicted_label
    input_df["probability"] = probability


def predict_multiple(input_df):
    model_path = f"../models/{input_model}.joblib"
    model = load(model_path)

    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[:, 1]

    input_df["predicted_label"] = predictions
    input_df["predicted_label"] = input_df["predicted_label"].apply(lambda x: "Subscribe" if x == 1
                                                                        else "Not Subscribe")
    input_df["probability"] = probabilities


# Streamlit UI
st.title("Model Prediction 📈")

st.header("About This Feature")
st.write("""This feature provides users prediction functionality so that they can both manually choose options and 
upload their CSV data files based on their personal circumstances.""")

input_model = "Random Forest"

input_types = [
    "  ",
    "Manually Input",
    "Upload CSV Data Files"
]

input_type = st.selectbox("Please select an option: ", input_types)

if input_type == "Manually Input":
    st.header("Enter Your Information: ")

    # Age
    age_input = st.text_input("Age")
    age = process_age_input(age_input)

    # Job
    jobs = ["  ", "Administrator", "Blue Collar", "Entrepreneur", "Housemaid", "Management", "Retired", "Self Employed",
            "Services", "Student", "Technician", "Unemployed", "Unknown"]
    job_input = st.selectbox(
        "Job",
        jobs
    )
    job = process_job(job_input)

    # Marital
    marital_statuses = ["", "Divorced", "Married", "Single", "Unknown"]
    marital_status_input = st.selectbox("Marital Status", marital_statuses)
    marital_status = process_marital_status(marital_status_input)

    # Education
    education_statuses = ["  ", "4 Years of Basic Education (Quatro Anos de Ensino Básico)",
                          "6 Years of Basic Education (Seis Anos de Ensino Básico)",
                          "9 Years of Basic Education (Nove Anos de Ensino Básico)",
                          "Secondary Education (Ensino Secundário)", "Illiterate (Analfabeto)",
                          "Professional Course (Curso Profissional)", "University Degree (Grau Universitário)",
                          "Unknown"]
    education_status_input = st.selectbox("Education", education_statuses)
    education_status = process_education_status(education_status_input)

    # Default
    default_status_input = st.selectbox("Has Credit in Default?", ["  ", "Yes", "No", "Unknown"])
    default_status = process_default_status(default_status_input)

    # Housing
    housing_status_input = st.selectbox("Has Housing Loan?", ["  ", "Yes", "No", "Unknown"])
    housing_status = process_housing_status(housing_status_input)

    # Loan
    loan_status_input = st.selectbox("Has Personal Loan?", ["  ", "Yes", "No", "Unknown"])
    loan_status = process_loan_status(loan_status_input)

    # Contact Type
    contact_type_input = st.selectbox("Contact Communication Type", ["  ", "Cellular", "Telephone"])
    contact_type = process_contact(contact_type_input)

    # Month
    month_input = st.selectbox("Last Contact Month",
                               ["  ", "January", "February", "March", "April", "May", "June", "July", "August",
                                "September",
                                "October", "November", "December"])
    month = process_month(month_input)

    # Day of the week
    day_of_week_input = st.selectbox("Last Contact Day of the Week", ["  ", "Monday", "Tuesday", "Wednesday",
                                                                      "Thursday", "Friday"])
    day_of_week = process_day_of_week(day_of_week_input)

    # Contact duration
    duration_input = st.text_input("Last Contact Duration (seconds)")
    duration = process_duration_input(duration_input)

    # Number of contacts
    number_of_contact_input = st.text_input("Number of Contacts Performed During this Campaign")
    number_of_contact = process_number_of_contact(number_of_contact_input)

    # Previous Outcome
    previous_outcome_input = st.selectbox("Outcome of the Previous Marketing Campaign", ["  ", "Failure",
                                                                                         "Non-Existent", "Success"])
    previous_outcome = process_previous_outcome(previous_outcome_input)

    user_input_data = {
        "Age": age_input,
        "Job": job_input,
        "Marital Status": marital_status_input,
        "Education Level": education_status_input,
        "Default History": default_status_input,
        "Housing Loan": housing_status_input,
        "Personal Loan": loan_status_input,
        "Contact Type": contact_type_input,
        "Last Contact Month": month_input,
        "Last Contact Day": day_of_week_input,
        "Last Contact Duration (seconds)": duration_input,
        "Contacts During Campaign": number_of_contact_input,
        "Previous Campaign Outcome": previous_outcome_input
    }

    data = pd.DataFrame([user_input_data])

    inputs_filled = all([
        age_input != "",
        job_input != "  ",
        marital_status_input != "  ",
        education_status_input != "  ",
        default_status_input != "  ",
        housing_status_input != "  ",
        loan_status_input != "  ",
        contact_type_input != "  ",
        month_input != "  ",
        day_of_week_input != "  ",
        duration_input != "",
        number_of_contact_input != "",
        previous_outcome_input != "  "
    ])

    if inputs_filled:
        st.header("Input Data Preview:")
        st.table(data)

    input_data = {
        "age": [age],
        "job": [job],
        "marital": [marital_status],
        "education": [education_status],
        "default": [default_status],
        "housing": [housing_status],
        "loan": [loan_status],
        "contact": [contact_type],
        "month": [month],
        "day_of_week": [day_of_week],
        "duration": [duration],
        "campaign": [number_of_contact],
        "poutcome": [previous_outcome]
    }

    input_df = pd.DataFrame.from_dict(input_data)

    if st.button("Predict"):
        predict_single(input_df)

        data["Predicted Subscription Status"] = input_df["predicted_label"]
        data["Prediction Probability"] = input_df["probability"]

        st.success('Done!')
        st.balloons()
        st.header("Prediction Results with Input Data:")
        st.table(data)
elif input_type == "Upload CSV Data Files":
    st.header("Upload Your CSV Files: ")
    uploaded_file = st.file_uploader("Upload your input CSV file", type="csv")

    default_file_path = f"../Test.csv"
    st.warning("Desired CSV Format Is As Followed. Please Match It.")
    st.table(pd.read_csv(default_file_path))

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            expected_columns = {"Age", "Job", "Marital Status", "Education Level", "Default History", "Housing Loan",
                                "Personal Loan", "Contact Type", "Last Contact Month", "Last Contact Day",
                                "Last Call Duration (s)", "Contacts During Campaign", "Previous Campaign Outcome"}
            if not expected_columns.issubset(data.columns):
                missing_columns = expected_columns - set(data.columns)
                st.error(f"Missing columns in the CSV file: {', '.join(missing_columns)}.")
                st.error("Please upload a file with the correct structure.")
            else:
                st.header("Input Data Preview:")
                if len(data) > 100:
                    st.write("Data is too large to display. Showing the first 100 rows:")
                    st.dataframe(data.head(100))
                else:
                    st.table(data)

                input_data = {
                    "age": data["Age"].apply(process_age_input),
                    "job": data["Job"].map(lambda x: process_job(x)),
                    "marital": data["Marital Status"].map(lambda x: process_marital_status(x)),
                    "education": data["Education Level"].map(lambda x: process_education_status(x)),
                    "default": data["Default History"].map(lambda x: process_default_status(x)),
                    "housing": data["Housing Loan"].map(lambda x: process_housing_status(x)),
                    "loan": data["Personal Loan"].map(lambda x: process_loan_status(x)),
                    "contact": data["Contact Type"].map(lambda x: process_contact(x)),
                    "month": data["Last Contact Month"].map(lambda x: process_month(x)),
                    "day_of_week": data["Last Contact Day"].map(lambda x: process_day_of_week(x)),
                    "duration": data["Last Call Duration (s)"].apply(process_duration_input),
                    "campaign": data["Contacts During Campaign"].apply(process_number_of_contact),
                    "poutcome": data["Previous Campaign Outcome"].map(lambda x: process_previous_outcome(x)),
                }
                input_df = pd.DataFrame(input_data)

                if st.button("Predict"):
                    predict_multiple(input_df)

                    data["Predicted Subscription Status"] = input_df["predicted_label"]
                    data["Prediction Probability"] = input_df["probability"]

                    st.success('Done!')
                    st.balloons()
                    st.header("Prediction Results with Input Data:")
                    st.table(data)
        except Exception as e:
            st.error(f"An error occurred while processing the CSV file: {e}")
