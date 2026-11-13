
import streamlit as st
import joblib
import pandas as pd
import numpy as np

classifier = joblib.load('hr_promotion_model.pkl')
df = pd.read_csv('Hr promotion.csv')

def predict_promotion(employee_id, department, region, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating, length_of_service, awards_won, avg_training_score):
    input_data = pd.DataFrame({
        'employee_id': [employee_id],
        'department': [department],
        'region': [region],
        'education': [education],
        'gender': [gender],
        'recruitment_channel': [recruitment_channel],
        'no_of_trainings': [no_of_trainings],
        'age': [age],
        'previous_year_rating': [previous_year_rating],
        'length_of_service': [length_of_service],
        'awards_won?': [awards_won],
        'avg_training_score': [avg_training_score]
    })

    prediction = classifier.predict(input_data)
    list_of_predictions = ['not promoted', 'promoted']
    return list_of_predictions[prediction[0]]


def main():
    st.title("HR Promotion Prediction App")
    employee_id = st.number_input("Employee ID", min_value=1, step=1)
    department = st.selectbox("Department", df['department'].unique())
    region = st.selectbox("Region", df['region'].unique())
    education = st.selectbox("Education", df['education'].unique())
    gender = st.selectbox("Gender", df['gender'].unique())
    recruitment_channel = st.selectbox("Recruitment Channel", df['recruitment_channel'].unique())
    no_of_trainings = st.number_input("Number of Trainings", min_value=0, step=1)
    age = st.number_input("Age", min_value=0, step=1)
    previous_year_rating = st.number_input("Previous Year Rating", min_value=0.0, max_value=5.0, step=0.1)
    length_of_service = st.number_input("Length of Service", min_value=0, step=1)
    awards_won = st.checkbox("Awards Won?")
    avg_training_score = st.number_input("Average Training Score", min_value=0.0, max_value=100.0, step=0.1)

    if st.button("Predict Promotion"):
        result = predict_promotion(employee_id, department, region, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating, length_of_service, awards_won, avg_training_score)
        st.success(f"Prediction: {result}")

if __name__ == "__main__":
    main()    
