#To create a Streamlit app for predicting the cost for a tourist, you can follow these steps. Note that you'll need to have Streamlit installed in your Python environment. You can install it using `pip install streamlit`.

#1. Create a new Python file, e.g., `tourist_cost_app.py`.
#2. Copy and paste the following code into your file:

#```python
# Importing the required libraries
import streamlit as st
import numpy as np
import pandas as pd
from os.path import dirname,join,realpath

# Load the dataset
data = pd.read_csv("V:\\my project\\kenya data\\New folder\\Clean_Kenya_Tourism_datasets (2).csv")

# Cleaning the data
# ... (Copy the data cleaning code from your original code)

# Convert categorical features to numeric
for colname in data.select_dtypes("object"):
    data[colname], _ = data[colname].factorize()

# Feature engineering
data["total_people"] = data["total_female"] + data["total_male"]
data["total_nights"] = data["night_mainland"]
data.drop('ID', axis='columns', inplace=True)

# Split the data
features_cols = data.drop(columns=["total_cost"])
cols = features_cols.columns
target = data["total_cost"]
X_train, X_test, y_train, y_test = train_test_split(data[cols], target, test_size=0.20, random_state=2020)

# Model training
XGB_par = XGBRegressor(n_estimators=100, colsample_bynode=0.8, learning_rate=0.02, max_depth=7)
XGB_par.fit(X_train, y_train)

# Streamlit app
st.title("Tourist Cost Prediction App")

# Sidebar with input parameters
st.sidebar.header("Input Parameters")
country = st.sidebar.selectbox("Country", data["country"].unique())
age_group = st.sidebar.selectbox("Age Group", data["age_group"].unique())
travel_with = st.sidebar.selectbox("Travel With", data["travel_with"].unique())
total_female = st.sidebar.slider("Total Female", min_value=0, max_value=10, step=1, value=1)
total_male = st.sidebar.slider("Total Male", min_value=0, max_value=10, step=1, value=1)
purpose = st.sidebar.selectbox("Purpose", data["purpose"].unique())
main_activity = st.sidebar.selectbox("Main Activity", data["main_activity"].unique())
info_source = st.sidebar.selectbox("Info Source", data["info_source"].unique())
tour_arrangement = st.sidebar.selectbox("Tour Arrangement", data["tour_arrangement"].unique())
package_transport_int = st.sidebar.selectbox("Package Transport", data["package_transport_int"].unique())
night_mainland = st.sidebar.slider("Night Mainland", min_value=0, max_value=30, step=1, value=10)
payment_mode = st.sidebar.selectbox("Payment Mode", data["payment_mode"].unique())
first_trip_Kenya = st.sidebar.selectbox("First Trip to Kenya", data["first_trip_Kenya"].unique())
most_impressing = st.sidebar.selectbox("Most Impressing", data["most_impressing"].unique())

# User input dictionary
user_input = {
    "country": country,
    "age_group": age_group,
    "travel_with": travel_with,
    "total_female": total_female,
    "total_male": total_male,
    "purpose": purpose,
    "main_activity": main_activity,
    "info_source": info_source,
    "tour_arrangement": tour_arrangement,
    "package_transport_int": package_transport_int,
    "night_mainland": night_mainland,
    "payment_mode": payment_mode,
    "first_trip_Kenya": first_trip_Kenya,
    "most_impressing": most_impressing,
}

# Create a DataFrame from the user input
user_input_df = pd.DataFrame([user_input])

# Convert categorical features to numeric
for colname in user_input_df.select_dtypes("object"):
    user_input_df[colname], _ = user_input_df[colname].factorize()

# Feature engineering
user_input_df["total_people"] = user_input_df["total_female"] + user_input_df["total_male"]
user_input_df["total_nights"] = user_input_df["night_mainland"]
user_input_df.drop(columns=["total_female", "total_male", "night_mainland"], inplace=True)

# Predict the cost
prediction = XGB_par.predict(user_input_df)[0]

# Display the prediction
st.header("Predicted Total Cost:")
st.write(f"${prediction:,.2f}")

# Visualize predictions
st.sidebar.header("Model Evaluation")
show_plot = st.sidebar.checkbox("Show Predictions vs Residuals")
