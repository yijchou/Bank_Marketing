import streamlit as st

st.set_page_config(
    page_title="Bank Marketing Campaign App",
)

st.sidebar.success("Select a page above.")

st.title("Bank Marketing Campaign 💰")

st.header("About The App")

st.write("""
This app offers features designed to explore, analyze, and model banking data related to term deposit subscriptions.
""")

# st.header("About The Dataset")
#
# st.write(""" The **Bank Marketing Campaign Dataset** that originates from the **UCI Machine Learning Repository**
# is a dataset used for analyzing and modeling marketing strategies. This dataset is derived from direct marketing
# campaigns (phone calls) of a Portuguese banking institution. The marketing campaigns were based on phone calls,
# often with the aim of promoting a term deposit product. The dataset includes a wide range of inputs such as age,
# job, marital status, education level, default history, housing and personal loan status, and so on. """)

st.header("Available Features")
st.markdown("""
- 📊 **Data Analysis**: 
    - <span style="margin-left: 0.2em;">Explore and analyze the data</span>
- 🤔 **Performance Evaluation**: 
    - <span style="margin-left: 0.2em;">View performance metrics</span>
- 📈 **Prediction**: 
    - <span style="margin-left: 0.2em;">Predict whether or not someone would end up subscribing to the term deposit</span>
""", unsafe_allow_html=True)
