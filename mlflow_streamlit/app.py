import streamlit as st
import mlflow.catboost
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder


# Load the trained CatBoost model from MLflow
# model_uri = "runs:/f61d27f11f684f67a94852cd60142389/catboost_model"
# model = mlflow.catboost.load_model(model_uri)

from catboost import CatBoostRegressor
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")


# Load the saved encoders (LabelEncoders and One-Hot Encoded columns)
encoders = joblib.load('encoders.pkl')
outlet_encoder = encoders['outlet_encoder']
label_encoders = encoders['label_encoders']
one_hot_columns = encoders['one_hot_columns']

# Streamlit app title and description
st.title("CatBoost Regression Model Deployment")
st.write("Enter the feature values below to predict Item_Outlet_Sales")

# Input interface for the user
Item_Weight = st.number_input("Item_Weight", min_value=0.0, max_value=50.0, value=10.0)
Item_Fat_Content = st.selectbox("Item_Fat_Content", options=label_encoders['Item_Fat_Content'].classes_)
Item_Visibility = st.slider("Item_Visibility", min_value=0.0, max_value=1.0, value=0.05)
Item_Type = st.selectbox("Item_Type", options=label_encoders['Item_Type'].classes_)
Item_MRP = st.number_input("Item_MRP", min_value=0.0, max_value=500.0, value=150.0)
Outlet_Identifier = st.selectbox("Outlet_Identifier", options=outlet_encoder.classes_)
Outlet_Establishment_Year = st.number_input("Outlet_Establishment_Year", min_value=1950, max_value=2025, value=2000)
Outlet_Size = st.selectbox("Outlet_Size", options=label_encoders['Outlet_Size'].classes_)
Outlet_Location_Type = st.selectbox("Outlet_Location_Type", options=label_encoders['Outlet_Location_Type'].classes_)
Outlet_Type = st.selectbox("Outlet_Type", options=label_encoders['Outlet_Type'].classes_)
New_Item_Type = st.selectbox("New_Item_Type", options=label_encoders['New_Item_Type'].classes_)

# Prediction button
if st.button("Predict Item Outlet Sales"):

    # Calculate the number of years the outlet has been established
    Outlet_Years = 2025 - Outlet_Establishment_Year

    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame({
        'Item_Weight': [Item_Weight],
        'Item_Fat_Content': [Item_Fat_Content],
        'Item_Visibility': [Item_Visibility],
        'Item_Type': [Item_Type],
        'Item_MRP': [Item_MRP],
        'Outlet_Identifier': [Outlet_Identifier],
        'Outlet_Establishment_Year': [Outlet_Establishment_Year],
        'Outlet_Size': [Outlet_Size],
        'Outlet_Location_Type': [Outlet_Location_Type],
        'Outlet_Type': [Outlet_Type],
        'New_Item_Type': [New_Item_Type],
        'Outlet_Years': [Outlet_Years]
    })

    # Apply the Outlet_Identifier encoder (encoded as 'Outlet')
    input_data['Outlet'] = outlet_encoder.transform(input_data['Outlet_Identifier'])

    # Apply LabelEncoders to other categorical columns
    cat_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
    for col in cat_cols:
        le = label_encoders[col]
        input_data[col] = le.transform(input_data[col])

    # Drop the original Outlet_Identifier column (replaced by 'Outlet')
    input_data.drop(columns=['Outlet_Identifier'], inplace=True)

    # Reorder columns to match the training dataset
    final_cols = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
                  'Item_Type', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size',
                  'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type', 'Outlet_Years', 'Outlet']
    input_data = input_data[final_cols]

    # Make prediction using the loaded CatBoost model
    prediction = model.predict(input_data)

    # Display the prediction result
    st.success(f"Predicted Item Outlet Sales: {prediction[0]:.2f}")
