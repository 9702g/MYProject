
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Header
st.header("Kenya Tourism Expenditure Prediction")
st.subheader("A simple machine learning app to predict how much money a tourist will spend when visiting Kenya.")

# Form
my_form = st.form(key="financial_form")

# Function to transform Yes and No options
@st.cache
def func(value):
    if value == 1:
        return "Yes"
    else:
        return "No"

# Selectbox for country
country = my_form.selectbox("Select country", [
    "SWIZERLAND", "UNITED KINGDOM", "CHINA", "SOUTH AFRICA", "UNITED STATES OF AMERICA",
    "NIGERIA", "INDIA", "BRAZIL", "CANADA", "MALT", "MOZAMBIQUE", "RWANDA", "AUSTRIA",
    "MYANMAR", "GERMANY", "KENYA", "ALGERIA", "IRELAND", "DENMARK", "SPAIN", "FRANCE",
    "ITALY", "EGYPT", "QATAR", "MALAWI", "JAPAN", "SWEDEN", "NETHERLANDS", "UAE", "UGANDA",
    "AUSTRALIA", "YEMEN", "NEW ZEALAND", "BELGIUM", "NORWAY", "ZIMBABWE", "ZAMBIA", "CONGO",
    "BURGARIA", "PAKISTAN", "GREECE", "MAURITIUS", "DRC", "OMAN", "PORTUGAL", "KOREA",
    "SWAZILAND", "TUNISIA", "KUWAIT", "DOMINICA", "ISRAEL", "FINLAND", "CZECH REPUBLIC",
    "UKRAIN", "ETHIOPIA", "BURUNDI", "SCOTLAND", "RUSSIA", "GHANA", "NIGER", "MALAYSIA",
    "COLOMBIA", "LUXEMBOURG", "NEPAL", "POLAND", "SINGAPORE", "LITHUANIA", "HUNGARY",
    "INDONESIA", "TURKEY", "TRINIDAD TOBACCO", "IRAQ", "SLOVENIA", "UNITED ARAB EMIRATES",
    "COMORO", "SRI LANKA", "IRAN", "MONTENEGRO", "ANGOLA", "LEBANON", "SLOVAKIA", "ROMANIA",
    "MEXICO", "LATVIA", "CROATIA", "CAPE VERDE", "SUDAN", "COSTARICA", "CHILE", "NAMIBIA",
    "TAIWAN", "SERBIA", "LESOTHO", "GEORGIA", "PHILIPINES", "IVORY COAST", "MADAGASCAR",
    "DJIBOUT", "CYPRUS", "ARGENTINA", "URUGUAY", "MORROCO", "THAILAND", "BERMUDA", "ESTONIA",
    "BOTSWANA", "VIETNAM", "GUINEA", "MACEDONIA", "HAITI", "LIBERIA", "SAUD ARABIA", "BOSNIA",
    "BULGARIA", "PERU", "BANGLADESH", "JAMAICA", "SOMALI"
])

# Selectbox for age group
age_group = my_form.selectbox("Select your age range", ["1-24", "25-44", "45-64", "65+"])

# Selectbox for travel with
travel_with = my_form.selectbox("Who do you plan to travel with?", ["Friends/Relatives", "Alone", "Spouse", "Children", "Spouse and Children"])

# Selectbox for purpose
purpose = my_form.selectbox("What is the purpose of visiting Kenya?", [
    "Leisure and Holidays", "Visiting Friends and Relatives", "Business", 
    "Meetings and Conference", "Volunteering", "Scientific and Academic", "Other"
])

# Number input for total number of people
total_number = my_form.number_input("How many people are you traveling with in Kenya?", min_value=1)

# Selectbox for main activity
main_activity = my_form.selectbox("What is the main activity you want to do in Kenya?", [
    "Wildlife tourism", "Cultural tourism", "Mountain climbing", "Beach tourism", 
    "Conference tourism", "Hunting tourism",

 "Bird watching", "Business", 
    "Diving and Sport Fishing"
])

# Selectbox for tour arrangement
tour_arrangement = my_form.selectbox("How do you arrange your tour?", ["Independent", "Package Tour"])

# Selectbox for package_transport_int
package_transport_int = my_form.selectbox("Does the package tour include International Transportation?", [0, 1], format_func=func)

# Selectbox for package_accomodation
package_accomodation = my_form.selectbox("Does the package tour include Accomodation service?", [0, 1], format_func=func)

# Selectbox for package_food
package_food = my_form.selectbox("Does the package tour include Food service?", [0, 1], format_func=func)

# Selectbox for package_transport_int
package_transport_int = my_form.selectbox("Does the package tour include Local Transportation when you are in Kenya?", [0, 1], format_func=func)

# Selectbox for package_sightseeing
package_sightseeing = my_form.selectbox("Does the package tour include Sight Seeing service?", [0, 1], format_func=func)

# Selectbox for package_guided_tour
package_guided_tour = my_form.selectbox("Does the package tour include Tour guiding service?", [0, 1], format_func=func)

# Selectbox for package_insurance
package_insurance = my_form.selectbox("Does the package tour include Insurance?", [0, 1], format_func=func)

# Selectbox for payment_mode
payment_mode = my_form.selectbox("What is your payment mode for tourism service?", ["Cash", "Credit Card", "Other", "Travellers Cheque"])

# Selectbox for first_trip_kenya
first_trip_kenya = my_form.selectbox("Is this your first trip to Kenya?", [0, 1], format_func=func)

# Number input for nights_stayed
nights_stayed = my_form.number_input("How many days do you plan to spend in Kenya Mainland?", min_value=0)

# Form submit button
submit = my_form.form_submit_button(label="Make Prediction")

# Load the model and one-hot-encoder and scaler
filename = 'tourism-data.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
    loaded_model = pickle.load(open(filename,'rb'))

# Result dictionary
result_dic = {
    1: " from Ksh 0 to Ksh 500,000",
    2: "from Ksh 500,001 to Ksh 1,000,000",
    3: "from Ksh 1,000,001 to Ksh 5,000,000",
    4: "from Ksh 5,000,001 to Ksh 10,000,000",
    5: "from Ksh 10,000,001 and above",
}

# Function to clean and transform the input
@st.cache
def preprocessing_data(data, one_hot_enc, scaler):
    # For other variables let's use one-hot-encoder
    multi_categorical_variables = [
        "country", "age_group", "tour_arrangement", "travel_with", "purpose",
        "main_activity", "payment_mode"
    ]
    multi_categorical_data = data[multi_categorical_variables]
    multi_categorical_data = one_hot_enc.transform(multi_categorical_data)
    data = data.drop(multi_categorical_variables, axis=1)
    data = data.to_numpy()
    final_data = np.concatenate([data, multi_categorical_data], axis=1)
    final_data = scaler.transform(final_data)
    return final_data

# If form submitted
if submit:
    # Collect inputs
    input_data = {
        "country": country, "age_group": age_group, "travel_with": travel_with,
        "total_number": total_number, "purpose": purpose, "main_activity": main_activity,
        "tour_arrangement": tour_arrangement, "package_transport_int": package_accomodation,
        "package_accomodation": package_accomodation, "package_food": package_food,
        "package_transport_int": package_transport_int, "package_sightseeing": package_sightseeing,
        "package_guided_tour": package_guided_tour, "package_insurance": package_insurance,
        "nights_stayed": nights_stayed, "payment_mode": payment_mode,
        "first_trip_kenya": first_trip_kenya
    }
    # Create a dataframe
    data = pd.DataFrame(input_data, index=[0])
    # Clean and transform input
    transformed_data = preprocessing_data(data=data, one_hot_enc=one_hot_encoder, scaler=scaler)
    # Perform prediction
    prediction = loaded_model.predict(transformed_data)
    output = int(prediction[0])
    # Display results of the Tourism prediction
    st.header("Results")
    st.write(" You are expected to spend: {}".format(result_dic[output]))
