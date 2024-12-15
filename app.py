import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "Clustered_Customer_Data.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# App Title
st.title("Cluster Prediction and Data Visualization")

# Sidebar for user inputs
st.sidebar.header("Input Data for Prediction")
input_data = {}

# Toggle between manual input and slider input
input_mode = st.sidebar.radio("Select Input Mode:", ("Slider", "Manual Input"))

for col in data.columns[:-1]:  # Exclude 'Cluster' column
    dtype = data[col].dtype
    if np.issubdtype(dtype, np.number):
        if input_mode == "Slider":
            # Use slider for numeric input
            min_val = float(data[col].min())
            max_val = float(data[col].max())
            input_data[col] = st.sidebar.slider(f"{col}", min_val, max_val, float(data[col].mean()))
        else:
            # Use text input for manual numeric input
            input_data[col] = st.sidebar.number_input(f"{col}", value=float(data[col].mean()))
    else:
        input_data[col] = st.sidebar.text_input(f"{col}", "")

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Predict Cluster
if st.sidebar.button("Predict Cluster"):
    # Find the nearest cluster (naive approach)
    distances = np.linalg.norm(data.drop(columns='Cluster').values - input_df.values, axis=1)
    predicted_cluster = data.iloc[np.argmin(distances)]['Cluster']
    st.success(f"The predicted cluster is: {int(predicted_cluster)}")

# Histogram Visualization
st.header("Data Visualization")
selected_column = st.selectbox("Select a column for histogram:", data.columns[:-1])  # Exclude 'Cluster'
if st.button("Generate Histogram"):
    fig, ax = plt.subplots()
    ax.hist(data[selected_column], bins=30, color='skyblue', edgecolor='black')
    ax.set_title(f"Histogram of {selected_column}")
    ax.set_xlabel(selected_column)
    ax.set_ylabel("Frequency")
    
    # Ensure x-axis starts at zero only if no negative values exist
    if data[selected_column].min() >= 0:
        ax.set_xlim(left=0)
    
    st.pyplot(fig)
