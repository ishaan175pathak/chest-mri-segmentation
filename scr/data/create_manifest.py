import os
import json
from pathlib import Path

DATA_DIR = Path("data/extracted/Nifti-data")

dataset = []

for patient in sorted(os.listdir(DATA_DIR)):

    patient_dir = os.path.join(DATA_DIR, patient)

    if not os.path.isdir(patient_dir):
        continue

    image_path = os.path.join(patient_dir, "image.nii.gz")
    label_path = os.path.join(patient_dir, "lung_seg.nii.gz")

    if os.path.exists(image_path) and os.path.exists(label_path):

        dataset.append({
            "image": image_path,
            "label": label_path,
            "patient_id": patient
        })

print(f"Total Valid Cases: {len(dataset)}")

with open("data/dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print("Manifest saved.")