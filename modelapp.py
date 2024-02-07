# import packages 
import streamlit as st
import pandas as pd
import numpy as np
from os.path import dirname, join, realpath


# add banner image
st.header("Kenya Tourism Expenditure Prediction")
st.subheader(
    """
A simple machine learning app to predict how much money a tourist will spend when visiting Kenya.
"""
)

# form to collect Tourist information
my_form = st.form(key="financial_form")


@st.cache
# function to trasform Yes and No options
def func(value):
    if value == 1:
        return "Yes"
    else:
        return "No"


country = my_form.selectbox(
    "select country",
    (
        "SWIZERLAND",
        "UNITED KINGDOM",
        "CHINA",
        "SOUTH AFRICA",
        "UNITED STATES OF AMERICA",
        "NIGERIA",
        "INDIA",
        "BRAZIL",
        "CANADA",
        "MALT",
        "MOZAMBIQUE",
        "RWANDA",
        "AUSTRIA",
        "MYANMAR",
        "GERMANY",
        "KENYA",
        "ALGERIA",
        "IRELAND",
        "DENMARK",
        "SPAIN",
        "FRANCE",
        "ITALY",
        "EGYPT",
        "QATAR",
        "MALAWI",
        "JAPAN",
        "SWEDEN",
        "NETHERLANDS",
        "UAE",
        "UGANDA",
        "AUSTRALIA",
        "YEMEN",
        "NEW ZEALAND",
        "BELGIUM",
        "NORWAY",
        "ZIMBABWE",
        "ZAMBIA",
        "CONGO",
        "BURGARIA",
        "PAKISTAN",
        "GREECE",
        "MAURITIUS",
        "DRC",
        "OMAN",
        "PORTUGAL",
        "KOREA",
        "SWAZILAND",
        "TUNISIA",
        "KUWAIT",
        "DOMINICA",
        "ISRAEL",
        "FINLAND",
        "CZECH REPUBLIC",
        "UKRAIN",
        "ETHIOPIA",
        "BURUNDI",
        "SCOTLAND",
        "RUSSIA",
        "GHANA",
        "NIGER",
        "MALAYSIA",
        "COLOMBIA",
        "LUXEMBOURG",
        "NEPAL",
        "POLAND",
        "SINGAPORE",
        "LITHUANIA",
        "HUNGARY",
        "INDONESIA",
        "TURKEY",
        "TRINIDAD TOBACCO",
        "IRAQ",
        "SLOVENIA",
        "UNITED ARAB EMIRATES",
        "COMORO",
        "SRI LANKA",
        "IRAN",
        "MONTENEGRO",
        "ANGOLA",
        "LEBANON",
        "SLOVAKIA",
        "ROMANIA",
        "MEXICO",
        "LATVIA",
        "CROATIA",
        "CAPE VERDE",
        "SUDAN",
        "COSTARICA",
        "CHILE",
        "NAMIBIA",
        "TAIWAN",
        "SERBIA",
        "LESOTHO",
        "GEORGIA",
        "PHILIPINES",
        "IVORY COAST",
        "MADAGASCAR",
        "DJIBOUT",
        "CYPRUS",
        "ARGENTINA",
        "URUGUAY",
        "MORROCO",
        "THAILAND",
        "BERMUDA",
        "ESTONIA",
        "BOTSWANA",
        "VIETNAM",
        "GUINEA",
        "MACEDONIA",
        "HAITI",
        "LIBERIA",
        "SAUD ARABIA",
        "BOSNIA",
        "BULGARIA",
        "PERU",
        "BANGLADESH",
        "JAMAICA",
        "SOMALI",
    ),
)

age_group = my_form.selectbox("Select you age range", ("1-24", "25-44", "45-64", "65+"))

travel_with = my_form.selectbox(
    "Who do you plan to travel with?",
    ("Friends/Relatives", "Alone", "Spouse", "Children", "Spouse and Children"),
)
purpose = my_form.selectbox(
    "What is the purpose of visiting Tanzania?",
    (
        "Leisure and Holidays",
        "Visiting Friends and Relatives",
        "Business",
        "Meetings and Conference",
        "Volunteering",
        "Scientific and Academic",
        "Other",
    ),
)

total_number = my_form.number_input(
    "How many people are you are coming with in Tanzania?", min_value=1
)

main_activity = my_form.selectbox(
    "What is the main activity you want to do when you are in Tanzania?",
    (
        "Wildlife tourism",
        "Cultural tourism",
        "Mountain climbing",
        "Beach tourism",
        "Conference tourism",
        "Hunting tourism",
        "Bird watching",
        "business",
        "Diving and Sport Fishing",
    ),
)

tour_arrangement = my_form.selectbox(
    "How do your arrange your tour?", ("Independent", "Package Tour")
)

package_transport_int = my_form.selectbox(
    "Does the package tour include International Transportation?",
    (0, 1),
    format_func=func,
)
package_accomodation = my_form.selectbox(
    "Does the package tour include Accomodation service?", (0, 1), format_func=func,
)
package_food = my_form.selectbox(
    "Does the package tour include Food service?", (0, 1), format_func=func
)
package_transport = my_form.selectbox(
    "Does the package tour include Local Transportation when you are in Tanzania?",
    (0, 1),
    format_func=func,
)
package_sightseeing = my_form.selectbox(
    "Does the package tour include Sight Seeing service?", (0, 1), format_func=func
)
package_guided_tour = my_form.selectbox(
    "Does the package tour include Tour guiding service?", (0, 1), format_func=func
)
package_insurance = my_form.selectbox(
    "Does the package tour include Insurance?", (0, 1), format_func=func
)
payment_mode = my_form.selectbox(
    "What is your payment mode for tourism service?",
    ("Cash", "Credit Card", "Other", "Travellers Cheque"),
)
first_trip_tz = my_form.selectbox(
    "Is this your first trip to kenya?", (0, 1), format_func=func
)

night_mainland = my_form.number_input(
    "How many days you plan to spend in Kenya Mainland", min_value=0,
)
night_zanzibar = my_form.number_input(
    "How many days you plan to spend in kenya", min_value=0
)

submit = my_form.form_submit_button(label="make prediction")


# load the model and one-hot-encoder and scaler

# Specify the path to the pickle file
pkl_file_path = 'histgradient-kenya-tourism-model (1).pkl'

# Open the pickle file in read binary mode
with open(pkl_file_path, 'rb') as file:
    # Load the object from the pickle file
    loaded_model = pickle.load(file)

# Now you can use the loaded_model object as needed


# result dictionary
result_dic = {
    1: " from ksh 0 to ksh 500,000",
    2: "from ksh 500,001 to ksh 1,000,000",
    3: "from ksh 1,000,001 to ksh 5,000,000",
    4: "from ksh 5,000,001 to ksh 10,000,000",
    5: "from ksh 10,000,001 and above",
}


@st.cache
# function to clean and tranform the input
def preprocessing_data(data, one_hot_enc, scaler):

    # For other variables let's use one-hot-encoder
    multi_categorical_variables = [
        "country",
        "age_group",
        "tour_arrangement",
        "travel_with",
        "purpose",
        "main_activity",
        "payment_mode",
    ]

    multi_categorical_data = data[multi_categorical_variables]

    multi_categorical_data = one_hot_enc.transform(multi_categorical_data)

    data = data.drop(multi_categorical_variables, axis=1)

    data = data.to_numpy()

    final_data = np.concatenate([data, multi_categorical_data], axis=1)

    final_data = scaler.transform(final_data)

    return final_data


if submit:

    # collect inputs
    input = {
        "country": country,
        "age_group": age_group,
        "travel_with": travel_with,
        "total_number": total_number,
        "purpose": purpose,
        "main_activity": main_activity,
        "tour_arrangement": tour_arrangement,
        "package_transport_int": package_accomodation,
        "package_accomodation": package_accomodation,
        "package_food": package_food,
        "package_transport_tz": package_transport_tz,
        "package_sightseeing": package_sightseeing,
        "package_guided_tour": package_guided_tour,
        "package_insurance": package_insurance,
        "night_mainland": night_mainland,
        "night_zanzibar": night_zanzibar,
        "payment_mode": payment_mode,
        "first_trip_tz": first_trip_tz,
    }

    # create a dataframe
    data = pd.DataFrame(input, index=[0])

    # clean and transform input
    transformed_data = preprocessing_data(
        data=data, one_hot_enc=one_hot_encoder, scaler=scaler
    )

    # perform prediction
    prediction = model.predict(transformed_data)
    output = int(prediction[0])

    # Display results of the Tourism prediction
    st.header("Results")
    st.write(" You are expected to spend:{} ".format(result_dic[output]))



