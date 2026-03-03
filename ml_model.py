import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import random

# ==========================================
# Karnataka Cities
# ==========================================
cities = [
    "Bangalore", "Mysore", "Mangalore", "Hubli", "Belgaum",
    "Gulbarga", "Davanagere", "Tumkur", "Udupi", "Shimoga",
    "Bellary", "Bijapur", "Chikmagalur", "Mandya", "Hassan",
    "Raichur", "Bidar", "Karwar", "Kolar", "Chitradurga"
]

# Create city mapping
city_mapping = {city: index for index, city in enumerate(cities)}

# ==========================================
# Generate Synthetic Training Data
# ==========================================
data = []

for _ in range(1000):

    city = random.choice(cities)
    city_code = city_mapping[city]

    amount = random.randint(100, 150000)
    hour = random.randint(0, 23)

    # Fraud logic for training
    fraud = 0

    if amount > 80000:
        fraud = 1
    if hour < 4:
        fraud = 1
    if amount > 50000 and hour < 6:
        fraud = 1

    data.append([amount, city_code, hour, fraud])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["amount", "city_code", "hour", "fraud"])

X = df[["amount", "city_code", "hour"]]
y = df["fraud"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ New ML Model Trained Successfully!")