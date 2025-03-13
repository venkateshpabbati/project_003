import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
from data_preprocessing import load_data  # Assuming this function is adapted for file input

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
parser.add_argument("--model", type=str, required=True, help="Path to save the trained model")
args = parser.parse_args()

# Load dataset from file
df = pd.read_csv(args.data)
X = df.drop(columns=['medv'])
y = df['medv']

# Split the dataset (or use load_data if you prefer a function)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Model trained with MSE: {mse}")

# Save the model
with open(args.model, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {args.model}")