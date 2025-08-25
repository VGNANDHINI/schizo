import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense

# App title
st.title("EEG Schizophrenia Classification App")
st.write("Upload your EEG CSV file (combined format with 'label' column)")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.dataframe(df.head())

    # Prepare data
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Reshape for CNN-LSTM
    timesteps = 6250
    num_channels = X.shape[1]
    num_frames = X.shape[0] // timesteps
    X = X[:num_frames*timesteps].reshape(num_frames, timesteps, num_channels)
    y = y[:num_frames*timesteps:timesteps]

    st.write(f"Data reshaped: {X.shape}")

    # Build simple CNN-LSTM model
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(timesteps, num_channels)),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        LSTM(100),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    st.write("Model built successfully!")

    # Train button
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            history = model.fit(X, y, epochs=3, batch_size=2, validation_split=0.2)
        st.success("Training completed!")
        st.write("Final training accuracy:", history.history['accuracy'][-1])
