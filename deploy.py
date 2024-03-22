import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import pickle

sc = pickle.load(open('scale1.pkl', 'rb'))
# Load the trained model
model = pickle.load(open('rfmodel1.pkl', 'rb'))

st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Create input fields for user to enter feature values
input_df = st.text_input('Input All features')

# Create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    if input_df:
        # Split the input string into a list of feature values
        input_df_lst = input_df.split(',')
        print("Input List:", input_df_lst)
        try:
            # Convert input features to numerical values
            input_array = np.array(input_df_lst, dtype=np.float64)
            print("Input Array:", input_array)
            # Standardize the input features using the previously fitted scaler
            standardized_input = sc.transform(input_array.reshape(1, -1))
            print("Standardized Input:", standardized_input)

            # Make prediction
            prediction = model.predict(standardized_input)
            print("Prediction:", prediction)
            # Display result
            if prediction[0] == 0:
                st.write("Legitimate transaction")
            else:
                st.write("Fraudulent transaction")
        except ValueError as e:
            st.write("ValueError:", e)
    else:
        st.write("Please provide input values.")
