# Importing necessary libs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle

x="https://i4.hurimg.com/i/hurriyet/75/750x422/59bd76fc7af5070fe0b2b8a1.jpg"
st.image(x,width=800)

df= pd.read_excel("sample data - rnv.xlsx",sheet_name="Ecommerce Customers")
df.drop(['Yearly Amount Spent','Email', 'Address', 'Avatar'],inplace=True, axis=1)


s1=st.sidebar.slider("Avg. Session Length",min_value=29.53, max_value=36.14)
s2=st.sidebar.slider("Time on App",min_value=8.51, max_value=15.13)
s3=st.sidebar.slider("Time on Website",min_value=33.91, max_value=40.01)
s4=st.sidebar.slider("Length of Membership",min_value=0.27, max_value=6.92)
col_list=["Avg. Session Length","Time on App","Time on Website","Length of Membership"]
l=[s1,s2,s3,s4]
df2=pd.DataFrame(data=[l],columns=col_list)
#st.write(df2.head())

filename = 'finalized_model.model'
loaded_model = pickle.load(open(filename, 'rb'))

filename2 = 'finalized_m.model'
loaded_model2 = pickle.load(open(filename2, 'rb'))

ypred = loaded_model.predict(df2)
st.title("Yearly Amoung Spent Prediction - XGBoost Regression")
st.title(ypred[0])

ypred2 = loaded_model2.predict(df2)
st.title("Yearly Amoung Spent Prediction - Lineer Regression")
ypred2=str(ypred2).strip("[]")
st.title(ypred2)
