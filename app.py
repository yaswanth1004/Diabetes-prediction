from flask import Flask, render_template, request
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from joblib import load
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("Final_Forest.pkl","rb"))

@app.route('/')
def input():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    bmi = float(request.form['bmi'])
    HbA1c_level = float(request.form['HbA1c_level'])
    blood_glucose_level = float(request.form['blood_glucose_level'])
    gender = int(request.form['gender'])
    smoking_history = int(request.form['smoking_history'])
    
    # Prepare the input data in the required format
    new_data = pd.DataFrame({
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level],
        'No Info': [int(smoking_history == 0)],
        'current': [int(smoking_history == 1)],
        'ever': [int(smoking_history == 2)],
        'former': [int(smoking_history == 3)],
        'never': [int(smoking_history == 4)],
        'not current': [int(smoking_history == 5)],
        'Female': [int(gender == 1)],
        'Male': [int(gender == 2)],
        'Other': [int(gender == 3)]
    })
    print("Form Data:", request.form)
    print("Input DataFrame:")
    print(new_data)
    # Load scaler and transform input data
    #scaler = StandardScaler()
    #numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    #new_data[numerical_features] = scaler.fit_transform(new_data[numerical_features])

    # Make the prediction
    prediction = model.predict(new_data)
    print(prediction)
    result = prediction[0]
    if result == 1:
        output = "Sorry to say this, you have Diabetes" 
    else:
        output = "You are safe, You have no diabetes"

    return output


if __name__ == '__main__':
    app.run(debug=True, port=4000)
