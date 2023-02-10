import pickle
import streamlit as st
import pandas as pd
import numpy as np

heart_disease_model = pickle.load(open('C:/Users/Lenovo/Downloads/heart_model.sav','rb'))


# page title
st.title('Heart Disease Prediction using ML')

# col1, col2, col3 = st.columns(3)

age = st.number_input('Age')

sex = st.radio('Sex', ('Male', 'Female'))

cp = st.number_input('Chest Pain types')
trestbps = st.number_input('Resting Blood Pressure')

chol = st.number_input('Serum Cholestoral in mg/dl')

fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')

restecg = st.number_input('Resting Electrocardiographic results')

thalach = st.number_input('Maximum Heart Rate achieved')

exang = st.number_input('Exercise Induced Angina')

oldpeak = st.number_input('ST depression induced by exercise')

slope = st.number_input('Slope of the peak exercise ST segment')

ca = st.number_input('Major vessels colored by flourosopy')

thal = st.selectbox('Thal', ('Normal', 'Fixed Defect', 'Reversable Defect'))

thal = 0 if thal == 'Normal' else 1 if thal == 'Fixed Defect' else 2

# code for Prediction
heart_diagnosis = ''

# creating a button for Prediction

if st.button('Heart Disease Test Result'):
    data = np.asarray(
        [age, 1 if sex == 'Male' else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca,
         thal]).reshape(1, -1)
    heart_prediction = heart_disease_model.predict(data)

    if (heart_prediction[0] == 1):
        heart_diagnosis = 'The person is having heart disease'
    else:
        heart_diagnosis = 'The person does not have any heart disease'

st.success(heart_diagnosis)






