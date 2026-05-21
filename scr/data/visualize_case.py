import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_PATH = Path("C:/USERS/ISHAANPATHAK/DESKTOP/CHEST_MRI_SEGMENTATION/DATA/EXTRACTED/Nifti-data/BPD-Neo-01/image.nii.gz")

LABEL_PATH = Path("C:/USERS/ISHAANPATHAK/DESKTOP/CHEST_MRI_SEGMENTATION/DATA/EXTRACTED/Nifti-data/BPD-Neo-01/lung_seg.nii.gz")

image = nib.load(IMAGE_PATH).get_fdata()
label = nib.load(LABEL_PATH).get_fdata()

slice_idx = image.shape[2] // 2

img_slice = image[:, :, slice_idx]
label_slice = label[:, :, slice_idx]

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_slice, cmap="gray")
plt.title("MRI")

plt.subplot(1, 3, 2)
plt.imshow(label_slice, cmap="gray")
plt.title("Mask")

plt.subplot(1, 3, 3)
plt.imshow(img_slice, cmap="gray")
plt.imshow(label_slice, alpha=0.4, cmap="Reds")
plt.title("Overlay")

plt.tight_layout()
plt.show()