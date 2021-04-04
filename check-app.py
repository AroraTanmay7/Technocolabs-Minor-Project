import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.set_option('deprecation.showfileUploaderEncoding', False)

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Recency = st.sidebar.slider('Recency (months)', 0.00 , 74.00, 59.80)
        Frequency = st.sidebar.slider('Frequency (times)', 1.00, 50.00, 21.50)
        Monetary = st.sidebar.slider('Monetary (c.c. blood)', 250.00, 12500.00, 560.00)
        Time = st.sidebar.slider('Time (months)', 2.00, 98.00, 50.00)
        data = {'Recency (months)': Recency,
                 'Frequency (times)': Frequency,
                'Monetary (c.c. blood)': Monetary,
                'Time (months)': Time}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset

transfusion_raw = pd.read_csv('transfusion.data')
np.any(np.isnan(transfusion_raw))
np.all(np.isfinite(transfusion_raw))


transfusion = transfusion_raw.drop(columns=['whether he/she donated blood in March 2007'])
df = pd.concat([input_df, transfusion], axis=0)
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('transfusion_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_prob = load_clf.predict_proba(df)


st.subheader('Prediction')
transfusion_prediction = np.array(['whether he/she donated blood in March 2007'])
st.write(prediction[0])

st.subheader('Prediction Probability')
st.write(prediction_prob)
