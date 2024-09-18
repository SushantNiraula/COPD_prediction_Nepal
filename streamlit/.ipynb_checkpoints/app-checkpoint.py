# Import the libraries
import pandas as pd
import pickle
import streamlit as st

# Load the trained model
# model_path = r'C:\Users\Admin\Desktop\Desmondonam\Omdena\Nepal_CBWP\COPD_Prediction\Best_Random_Forest_Model.pkl'

with open('../models/Best_Random_Forest_Model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit App
def main():
    st.title("COPD Prediction Dashboard")

    # User input
    st.sidebar.header("User Input")

    age = st.sidebar.slider("Age", 30, 80, 50)
    gender = st.sidebar.selectbox("Gender", [1, 0])
    bmi = st.sidebar.slider("BMI", 10, 40, 25)
    smoking_status = st.sidebar.selectbox("Smoking Status", ["Current", "Former", "Never"])
    biomass_fuel_exposure = st.sidebar.selectbox("Biomass Fuel Exposure", [1, 0])
    occupational_exposure = st.sidebar.selectbox("Occupational Exposure", [1, 0])
    family_history = st.sidebar.selectbox("Family History", [1, 0])
    air_pollution_level = st.sidebar.slider("Air Pollution Level", 0, 300, 50)
    respiratory_infections = st.sidebar.selectbox("Respiratory Infections in Childhood", [1, 0])
    location = st.sidebar.selectbox("Location", ["Kathmandu", "Pokhara", "Biratnagar", "Lalitpur", "Birgunj", 'Chitan', "Hetauda", "Dharan", "Butwal"])


    
    # Process the input data
    input_data = {
        
        'Smoking_Status': [smoking_status],
        'Biomass_Fuel_Exposure': [biomass_fuel_exposure],
        'Occupational_Exposure': [occupational_exposure],
        'Family_History_COPD': [family_history],
        'Location':[location],
        'Respiratory_Infections_Childhood': [respiratory_infections],
        'Age_Category': [age],
        'BMI_category': [bmi],
        'Air_Pollution_Level_category': [air_pollution_level],
        'Gender_encoded': [gender]       
    }

    # Convert the data to a dataframe
    input_df = pd.DataFrame(input_data)

    '''
    ['Smoking_Status',	'Biomass_Fuel_Exposure',	'Occupational_Exposure',	'Family_History_COPD',	'Location',	'Respiratory_Infections_Childhood',      'Age_Category',	'BMI_category',	'Air_Pollution_Level_category',	'Gender_encoded']
    '''

    #pipeLINE
    
with open('../models/pipe.pkl', 'rb') as f:
    pipe = pickle.load(f)

    input_test=pipe.transform(input_df)

    
   #  # Encoding
   #  input_df['Gender_'] = input_df["Gender_"].map({'Male': 1, 'Female': 0})
   #  input_df['Smoking_Status_encoded'] = input_df['Smoking_Status_encoded'].map({'Current': 1, 'Former': 0.5, 'Never': 0})
   #  input_df['Biomass_Fuel_Exposure'] = input_df["Biomass_Fuel_Exposure"].map({'Yes': 1, 'No': 0})
   #  input_df['Occupational_Exposure'] = input_df["Occupational_Exposure"].map({'Yes': 1, 'No': 0})
   #  input_df['Family_History_COPD'] = input_df["Family_History_COPD"].map({'Yes': 1, 'No': 0})
   #  input_df['Respiratory_Infections_Childhood'] = input_df["Respiratory_Infections_Childhood"].map({'Yes': 1, 'No': 0})
   #  # location_dummies = pd.get_dummies(input_df['Location'], prefix='Location')
   #  # input_df = pd.concat([input_df, location_dummies], axis = 1)

   # # input_df.drop('Location', axis=1, inplace = True)
    

    # Prediction
    prediction = model.predict(input_test)
    if prediction[0] == 1:
        st.write("Predictions: COPD Detected")
    else:
        st.write("Prediction: No COPD Detected")

if __name__ == "__main__":
    main()