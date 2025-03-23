import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv("loan_data.csv")

# Preprocessing (ensure this aligns with your dataset)
X = data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]
y = data['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open("loan_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as loan_model.pkl")
