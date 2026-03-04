import streamlit as st
import pandas as pd
import joblib

model= joblib.load("rfiris.pkl")

st.title(" IRIS FLOWER CLASSIFICATION APPLICATION")

st.write("Predict the species of an Iris Flower Using a Random Forest Model")

form=st.form("Iris form")

form.subheader("Enter Flower Measurement")

sepal_length = form.number_input(
	"sepal_length (cm)",
	min_value=4.0,
	max_value=8.0,
	value=5.1
	)

sepal_width = form.number_input(

	"sepal_width (cm)",
	min_value=1.0,
	max_value=7.5,
	value=5.1
	)

petal_length = form.number_input(

	"petal_length (cm)",
	min_value=1.0,
	max_value=7.0,
	value=5.1
	)


petal_width = form.number_input(

	"petal_width (cm)",
	min_value=4.0,
	max_value=8.0,
	value=5.1
	)

submit_button = form.form_submit_button('Predict')

if submit_button:
	input_data = pd.DataFrame( 
		[[sepal_length, sepal_width, petal_length, petal_width]], 
							  columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

	prediction = model.predict(input_data)
	st.subheader('Prediction Result')

	st.success(f" Predicted species: {prediction[0]}")


