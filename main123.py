# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.title("EEG Schizophrenia Classification App")
st.write("Upload your EEG CSV file (must include 'label' column)")

# -------------------------------
# Step 1: Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.dataframe(df.head())

    # -------------------------------
    # Step 2: Prepare data
    # -------------------------------
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # For demonstration, we reduce timesteps to 1000 for speed
    timesteps = 1000
    num_channels = X.shape[1]
    num_frames = X.shape[0] // timesteps
    X = X[:num_frames*timesteps].reshape(num_frames, timesteps, num_channels)
    y = y[:num_frames*timesteps:timesteps]

    st.write(f"Data reshaped for CNN-LSTM: {X.shape}")

    # -------------------------------
    # Step 3: Train/Test split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------
    # Step 4: Build CNN-LSTM model
    # -------------------------------
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

    # -------------------------------
    # Step 5: Train button
    # -------------------------------
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            history = model.fit(
                X_train, y_train,
                epochs=3,          # increase epochs for real training
                batch_size=2,
                validation_split=0.2
            )
        st.success("Training completed!")
        st.write("Final training accuracy:", history.history['accuracy'][-1])

        # -------------------------------
        # Step 6: Evaluate model
        # -------------------------------
        loss, acc = model.evaluate(X_test, y_test)
        st.write(f"Test Accuracy: {acc*100:.2f}%")

        # Confusion Matrix
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Schizophrenia'])
        disp.plot()
        st.pyplot(plt)

        # Training plot
        plt.figure(figsize=(8,4))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training Accuracy")
        st.pyplot(plt)

    # -------------------------------
    # Step 7: Prediction on new CSV
    # -------------------------------
    st.write("---")
    st.write("Predict on new EEG CSV")

    uploaded_new = st.file_uploader("Upload new EEG CSV", type="csv", key="new")

    if uploaded_new:
        new_df = pd.read_csv(uploaded_new)
        # Simple reshaping for single frame
        X_new = new_df.values.reshape(1, X.shape[1], X.shape[2])
        pred = (model.predict(X_new) > 0.5).astype(int)
        st.write("Predicted label:", "Schizophrenia" if pred[0][0] else "Healthy")
