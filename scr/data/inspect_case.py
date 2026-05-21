import nibabel as nib
import numpy as np
from pathlib import Path

IMAGE_PATH = Path("C:/USERS/ISHAANPATHAK/DESKTOP/CHEST_MRI_SEGMENTATION/DATA/EXTRACTED/Nifti-data/BPD-Neo-01/image.nii.gz")

LABEL_PATH = Path("C:/USERS/ISHAANPATHAK/DESKTOP/CHEST_MRI_SEGMENTATION/DATA/EXTRACTED/Nifti-data/BPD-Neo-01/lung_seg.nii.gz")

image = nib.load(IMAGE_PATH).get_fdata()
label = nib.load(LABEL_PATH).get_fdata()

print("=" * 50)

print("Image Shape:", image.shape)
print("Label Shape:", label.shape)

print("Image Min:", image.min())
print("Image Max:", image.max())

print("Unique Label Values:", np.unique(label))