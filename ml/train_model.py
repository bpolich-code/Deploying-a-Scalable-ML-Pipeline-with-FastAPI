import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load the census.csv data
data_path = os.path.join("data", "census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Split the provided data to have a train dataset and a test dataset
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Use the process_data function to process the data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model on the training dataset
print("Training model...")
model = train_model(X_train, y_train)

# Save the model and the encoder
model_path = os.path.join("model", "model.pkl")
save_model(model, model_path)
print(f"Model saved to {model_path}")

encoder_path = os.path.join("model", "encoder.pkl")
save_model(encoder, encoder_path)
print(f"Encoder saved to {encoder_path}")

# Load the model
model = load_model(model_path)
print(f"Model loaded from {model_path}")

# Run the model inferences on the test dataset
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute the performance on model slices
print("\nComputing performance on categorical slices...")

# Clear the slice_output.txt file if it exists
if os.path.exists("slice_output.txt"):
    os.remove("slice_output.txt")

# Iterate through the categorical features
for col in cat_features:
    # Iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            test,
            col,
            slicevalue,
            cat_features,
            "salary",
            encoder,
            lb,
            model
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)

print("Slice performance metrics saved to slice_output.txt")