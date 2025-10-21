import joblib
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# copy the path reference / path
kmeans = joblib.load('/Users/prashanth45/Downloads/kmeans_model.pkl')
scaler = joblib.load('/Users/prashanth45/Downloads/scaler.pkl')

st.title('Ecommerce Purchase behaviour')
st.write('Enter customer details to predict the segment')

Age = st.number_input("Age of Customer : ", min_value= 18, max_value=100, value=40)
Income_Level = st.selectbox("Income_level of Customer", ["Middle", "High"])
Purchase_Amount = st.number_input("purchase amount :", min_value=0, max_value=10000000, value=100000)
Frequency_of_Purchase = st.number_input("Frequency_of_Purchase : ", min_value=0, max_value=1000, value=10)
High_Value_Customer = st.number_input("High_Value_Customer : ", max_value=1, min_value=0, value=0)
Overall_Satisfaction = st.number_input("Overall_Satisfaction : ", min_value=0, max_value=10, value=4)

if Income_Level == 'Middle':
    Income_Level = 0
else:
    Income_Level = 1



input_data = pd.DataFrame({
    'Age': [Age],
    'Income_Level': [Income_Level],
    'Purchase_Amount': [Purchase_Amount],
    'Frequency_of_Purchase': [Frequency_of_Purchase],
    'High_Value_Customer': [High_Value_Customer],
    'Overall_Satisfaction': [Overall_Satisfaction],

})


input_scaled = scaler.transform(input_data)



if st.button('Predict Segment'):
    cluster = kmeans.predict(input_scaled)
    st.success(f"The predicted segment :  {cluster}")
