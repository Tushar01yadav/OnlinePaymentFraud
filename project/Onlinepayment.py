import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Dataset (Replace with real dataset)
# Simulating a dataset (for demo)
np.random.seed(42)
data_size = 5000
data = pd.DataFrame({
    "amount": np.random.uniform(10, 5000, data_size),  # Transaction amount
    "time": np.random.uniform(0, 86400, data_size),  # Time of transaction
    "location": np.random.randint(1, 50, data_size),  # Encoded location ID
    "device_type": np.random.randint(1, 5, data_size),  # Encoded device ID
    "is_fraud": np.random.choice([0, 1], data_size, p=[0.97, 0.03])  # Fraud label
})

# Step 2: Split dataset into features (X) and target (y)
X = data.drop(columns=["is_fraud"])
y = data["is_fraud"]

# Step 3: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Test on New Data
new_transaction = np.array([[2500, 45000, 15, 2]])  # Example: Amount=2500, time=45000, location=15, device=2
new_transaction = scaler.transform(new_transaction)
prediction = model.predict(new_transaction)
print("Fraud Detected!" if prediction[0] == 1 else "Transaction is Safe.")
