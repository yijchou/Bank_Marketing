import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

sns.set(style="whitegrid")


@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        expected_columns = {'age', 'job', 'marital', 'education', 'default',
                            'housing', 'loan', 'contact', 'month', 'day_of_week',
                            'duration', 'campaign', 'poutcome', 'y'}

        if not expected_columns.issubset(df.columns):
            missing_columns = expected_columns - set(df.columns)
            st.error(f"Missing columns in the CSV file: {', '.join(missing_columns)}.")
            # st.error("Please upload a file with the correct structure.")

            return None

        if 'y' in df.columns:
            df["y"] = df["y"].eq("yes").mul(1)

        df["age"] = pd.to_numeric(df["age"], errors="coerce")

        return df
    except Exception as e:
        st.error(f"An error occurred while reading the CSV file: {e}")
        return None


@st.cache_data
def encode_features(df):
    le = LabelEncoder()
    features_to_encode = ["education", "job", "default", "housing", "loan", "marital", "contact", "month",
                          "day_of_week", "poutcome"]
    for feature in features_to_encode:
        df[feature] = le.fit_transform(df[feature])
    return df


def plot_response_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(x="y", data=df, palette="viridis")
    st.pyplot(fig)


def plot_categorical_analysis(df, feature_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    order = df[feature_name].value_counts().index
    sns.countplot(y=feature_name, hue="y", data=df, order=order)
    st.pyplot(fig)


def show_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(15, 12))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="RdYlGn", linewidths=0.2, annot_kws={"size": 12})
    st.pyplot(plt.gcf())


def show_homepage():
    st.title("Data Analysis Dashboard 📊")

    st.header("About This Feature")
    st.write("""
    This feature allows bank managers to upload CSV data files and perform various data analyses.""")

    st.header("Getting Start")
    st.write("""
    1. Use the sidebar to upload your CSV file that matches the predefined format.
    2. Select the type of analysis you want to perform.
    3. View the results on the dashboard.
    """)

    default_file_path = f"../new_train2.csv"
    st.warning("Desired CSV Format Is As Followed. Please Match It.")
    st.table(pd.read_csv(default_file_path).head(5))


def categorize_duration(df):
    df["duration_range"] = pd.qcut(df["duration"], 10, duplicates="drop")
    return df


def plot_duration_range_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    order = df["duration_range"].value_counts().index.sort_values()
    sns.countplot(x="duration_range", hue="y", data=df, order=order, palette="viridis")
    plt.xticks(rotation=45)
    st.pyplot(fig)


def main():
    st.sidebar.header("User Input Features")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    analysis_options = [
                           "Home", "Response distribution", "Correlation heatmap", "All feature distributions"
                       ] + ["Distribution by " + col for col in [
        "age", "job", "marital", "education", "default", "housing", "loan", "contact",
        "month", "day_of_week", "campaign", "poutcome"
    ]]

    selected_options = st.sidebar.multiselect(
        "Choose analyses to display",
        analysis_options,
        default=["Home"]
    )

    if "Home" in selected_options:
        show_homepage()

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is None:
            st.error("Failed to load data. Please check the format of your CSV file.")
            return

        df = categorize_duration(df)

        if "Response distribution" in selected_options:
            st.header("Response Distribution")
            plot_response_distribution(df)

        if "Correlation heatmap" in selected_options:
            df_encoded = encode_features(df)
            st.header("Feature Correlation Heatmap")
            show_correlation_heatmap(df_encoded)

        if "All feature distributions" in selected_options:
            for col in [
                "age", "job", "marital", "education", "default", "housing", "loan", "contact",
                "month", "day_of_week", "campaign", "poutcome"
            ]:
                plot_categorical_analysis(df, col)
            plot_duration_range_distribution(df)

        for option in selected_options:
            if option.startswith("Distribution by "):
                feature_name = option.split("Distribution by ")[1]
                st.header(f"{feature_name} distribution".title())
                plot_categorical_analysis(df, feature_name)

    else:
        if "Home" not in selected_options:
            st.info("Please upload a CSV file to begin the analysis.")


if __name__ == "__main__":
    main()
