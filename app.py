import streamlit as st
import pickle
import os
import pandas as pd

# Set the directory where the pickled files are stored
directory = "Sample data/pkl"

# Load the model performance summary
summary_filename = os.path.join(directory, "model_performance_summary.pkl")

# Error handling for missing file
if os.path.exists(summary_filename):
    with open(summary_filename, 'rb') as file:
        results_df = pickle.load(file)
else:
    st.error("Model performance summary file not found.")
    st.stop()

st.title("Model Performance Dashboard")

# Display the summary DataFrame
st.subheader("Model Performance Summary")
st.dataframe(results_df.sort_values(by='Test Accuracy', ascending=False))

# Dropdown for model selection
model_name = st.selectbox("Select a model to view its details:", results_df['Model'])

# Initialize model, confusion matrix, classification report, and accuracy
model = confusion_matrix = classification_report = accuracy = None

# Load the selected model, confusion matrix, classification report, and accuracy
model_filename = os.path.join(directory, f"{model_name.replace(' ', '_')}_model.pkl")
cm_filename = os.path.join(directory, f"{model_name.replace(' ', '_')}_confusion_matrix.pkl")
report_filename = os.path.join(directory, f"{model_name.replace(' ', '_')}_classification_report.pkl")
accuracy_filename = os.path.join(directory, f"{model_name.replace(' ', '_')}_accuracy.pkl")

# Load the model if file exists
if os.path.exists(model_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
else:
    st.error(f"Model file not found: {model_filename}")
    st.stop()

# Load the confusion matrix if file exists
if os.path.exists(cm_filename):
    with open(cm_filename, 'rb') as file:
        confusion_matrix = pickle.load(file)
else:
    st.error(f"Confusion matrix file not found: {cm_filename}")
    st.stop()

# Load the classification report if file exists
if os.path.exists(report_filename):
    with open(report_filename, 'rb') as file:
        classification_report = pickle.load(file)
else:
    st.error(f"Classification report file not found: {report_filename}")
    st.stop()

# Load the accuracy if file exists
if os.path.exists(accuracy_filename):
    with open(accuracy_filename, 'rb') as file:
        accuracy = pickle.load(file)
else:
    st.error(f"Accuracy file not found: {accuracy_filename}")
    st.stop()

# Display model accuracy
st.subheader(f"{model_name} - Accuracy")
st.write(f"Test Accuracy: {accuracy:.4f}")

# Display confusion matrix
st.subheader(f"{model_name} - Confusion Matrix")
st.write(confusion_matrix)

# Display classification report
st.subheader(f"{model_name} - Classification Report")
st.text(classification_report)

# Option to upload a custom dataset for prediction
st.subheader("Make Predictions")
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data)

    # Check if the model and data are compatible
    if model is not None and not data.empty:
        # Assuming the model expects the data in the same format as training
        try:
            predictions = model.predict(data)
            st.write("Predictions:")
            st.write(predictions)

            # Optionally, save predictions to a CSV file
            if st.button("Save Predictions to CSV"):
                prediction_df = pd.DataFrame(predictions, columns=["Prediction"])
                prediction_df.to_csv("predictions.csv", index=False)
                st.success("Predictions saved to predictions.csv")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("No model loaded or uploaded data is empty.")


