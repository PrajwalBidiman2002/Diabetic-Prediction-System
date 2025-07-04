import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Load the dataset
diabete_df = pd.read_csv('Diabetes.csv')

# Split the data
X = diabete_df.drop('Outcome', axis=1)
y = diabete_df['Outcome']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the model
model = svm.SVC(kernel='linear')
model.fit(X, y)

# Streamlit Web App
st.title('Diabetes Prediction Web App')
st.write("Enter the patient's details to predict diabetes.")

# Input fields
pregnancies = st.number_input('Pregnancies', min_value=0)
glucose = st.number_input('Glucose', min_value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0)
skin_thickness = st.number_input('Skin Thickness', min_value=0)
insulin = st.number_input('Insulin', min_value=0)
bmi = st.number_input('BMI', min_value=0.0)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0)
age = st.number_input('Age', min_value=0)

# Prediction
if st.button('Predict'):
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age)
    input_array = np.asarray(input_data).reshape(1, -1)
    std_input = scaler.transform(input_array)
    
    prediction = model.predict(std_input)

    if prediction[0] == 1:
        st.error('The person is likely to have diabetes.')
    else:
        st.success('The person is not likely to have diabetes.')
        

