from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
model_path = "loan_model.pkl"  # Ensure this file is in the same directory
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure it is in the correct location.")

# Route for the home page
@app.route("/")
def index():
    return render_template("index.html", result=None)

# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        applicant_income = int(request.form["applicant_income"])
        coapplicant_income = int(request.form["coapplicant_income"])
        loan_amount = int(request.form["loan_amount"])
        loan_amount_term = int(request.form["loan_amount_term"])
        credit_history = int(request.form["credit_history"])

        # Create feature array
        features = np.array([[applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history]])

        # Predict using the model
        prediction = model.predict(features)[0]
        result = "Loan Approved" if prediction == 1 else "Loan Rejected"
    except Exception as e:
        result = f"Error occurred: {e}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
