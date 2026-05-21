import json
import os
from sklearn.model_selection import train_test_split

with open("data/dataset.json", "r") as f:
    dataset = json.load(f)

train_data, test_data = train_test_split(
    dataset,
    test_size=0.15,
    random_state=42
)

train_data, val_data = train_test_split(
    train_data,
    test_size=0.15,
    random_state=42
)

os.makedirs("data/splits", exist_ok=True)

with open("data/splits/train.json", "w") as f:
    json.dump(train_data, f, indent=4)

with open("data/splits/val.json", "w") as f:
    json.dump(val_data, f, indent=4)

with open("data/splits/test.json", "w") as f:
    json.dump(test_data, f, indent=4)

print(f"Train Cases: {len(train_data)}")
print(f"Validation Cases: {len(val_data)}")
print(f"Test Cases: {len(test_data)}")