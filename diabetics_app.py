import streamlit as st
import pickle
import numpy as np

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def diabetics_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# web page
def main():
    st.set_page_config(page_title='Diabetes Prediction App', page_icon="🏥")
    st.title('Diabetes Prediction App')
    Pregnancies = st.text_input("Number of Pregnancies:")
    Glucose = st.text_input("Glucose Level:")
    BloodPressure = st.text_input("Blood Pressure Value:")
    SkinThickness = st.text_input("Skin Thickness Value:")
    Insulin = st.text_input("Insulin Value:")
    BMI = st.text_input("BMI Value:")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value:")
    Age = st.text_input("Age of the Person:")
    diagnosis = ''

    if st.button('Test Result'):
        diagnosis = diabetics_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()