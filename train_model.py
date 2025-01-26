import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("diabetes.csv")  # Ensure 'diabetes.csv' is in your project folder

# Split features and target
X = df.drop(columns=['Outcome'])  # Independent variables
y = df['Outcome']  # Dependent variable (1 = Diabetic, 0 = Non-Diabetic)

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and scaler
pickle.dump(model, open("diabetes_model.pkl", "wb"))  # Save model
pickle.dump(scaler, open("scaler.pkl", "wb"))  # Save scaler
