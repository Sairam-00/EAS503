import streamlit as st
import pandas as pd
from input_processing import preprocess_input
from input_processing import model


st.title('Get your car price')

data=pd.read_csv('data_final.csv')

company = []
engine = []
color = []
transmission = []
reports = []
accidents = []

for j in data.iterrows():
    i=list(list(j)[1])
    temp1 = (i[0],i[1])
    temp2 = (i[4],i[5])
    temp3 = (i[7],i[8])
    if temp1 not in company:
        company.append(temp1)
    if temp2 not in engine:
        engine.append(temp2)
    for k in temp3:
        if k not in color:
            color.append(k)

transmission=list(data['Transmission'].unique())
reports=list(data['Accident'].unique())
accidents=list(data['Clean_title'].unique())

car_make = st.selectbox('Car Make', ['Select a make'] + company)
year_made = st.number_input('Year Made', min_value=1990, max_value=2024, step=1, format="%d", value=None)
mileage = st.number_input('Mileage', min_value=0, max_value=500000, step=1000, format="%d", value=None)
engine_model = st.selectbox('Engine Model', ['Select an engine model'] + engine)
transmission = st.selectbox('Transmission', ['Select a transmission'] + transmission)
exterior_color = st.selectbox('Exterior Color', ['Select a color'] + color)
interior_color = st.selectbox('Interior Color', ['Select a color'] + color)
reports = st.selectbox('Reports', ['Select a report'] + ['At least 1 accident or damage reported', 'None reported'])
accidents_involved = st.selectbox('Accidents Involved', ['Any Accidents'] + ['Yes','No'])

if st.button('Get Price'):

    input=[car_make[0],car_make[1],year_made,mileage,engine_model[0],engine_model[1],transmission,exterior_color,interior_color,reports,accidents_involved]
    processed_data = preprocess_input(input)
    prediction = model(processed_data)
    price = prediction

    # Display the price
    st.write(f"The estimated price of your car is: ${price}")
