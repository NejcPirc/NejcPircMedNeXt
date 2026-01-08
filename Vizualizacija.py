import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib.colors import ListedColormap

# --- POTI ---
pred_dir = "./predictions_final"
img_dir = "./data/nnUNet_raw/Dataset004_FinalTest/imagesTs"
lbl_dir = "./data/nnUNet_raw/Dataset004_FinalTest/test_labels_ref"

# --- 1. POIŠČI SLIKO ---
vse_napovedi = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
if len(vse_napovedi) == 0:
    print("❌ Ni slik v predictions_final!")
    exit()

pred_path = vse_napovedi[4] 
filename = os.path.basename(pred_path)
print(f"📸 Prikazujem (Cela slika): {filename}")

img_path = os.path.join(img_dir, filename)
lbl_filename = filename.replace("_0000.nii.gz", ".nii.gz")
lbl_path = os.path.join(lbl_dir, lbl_filename)

# --- 2. NALAGANJE ---
def load_data(path):
    if not os.path.exists(path): return None
    img = nib.load(path)
    data = img.get_fdata()
    mid = data.shape[2] // 2
    return np.rot90(data[:, :, mid])

image = load_data(img_path)
label = load_data(lbl_path)
pred = load_data(pred_path)

if image is None: exit()

# Resize (da so vse iste velikosti)
if label is not None:
    label_resized = resize(label, image.shape, order=0, preserve_range=True, anti_aliasing=False)
else:
    label_resized = np.zeros_like(image)

pred_resized = resize(pred, image.shape, order=0, preserve_range=True, anti_aliasing=False)

# Kontrast
image = np.clip(image, -500, 800)

# Barve
cmap_gt = ListedColormap(['lime']) 
cmap_pred = ListedColormap(['red']) 

# --- 3. RISANJE (Brez zooma) ---
# Povečamo višino slike, da bo več prostora za naslove
fig, axes = plt.subplots(1, 3, figsize=(20, 8)) 

# A) Original
axes[0].imshow(image, cmap='gray')
# pad=20 doda prostor med sliko in naslovom
axes[0].set_title("CT Original", fontsize=20, fontweight='bold', pad=20)
axes[0].axis('off')

# B) nnU-Net Standard (Referenca)
axes[1].imshow(image, cmap='gray')
axes[1].imshow(np.ma.masked_where(label_resized < 0.5, label_resized), cmap=cmap_gt, alpha=1.0)
axes[1].set_title("nnU-Net Standard", fontsize=20, fontweight='bold', pad=20)
axes[1].axis('off')

# C) Tvoja metoda
axes[2].imshow(image, cmap='gray')
axes[2].imshow(np.ma.masked_where(pred_resized < 0.5, pred_resized), cmap=cmap_pred, alpha=1.0)
axes[2].set_title("MedNeXt Metoda", fontsize=20, fontweight='bold', pad=20)
axes[2].axis('off')

# --- KLJUČEN POPRAVEK ZA NASLOVE ---
plt.tight_layout()
# Pustimo 10% prostora na vrhu praznega, da se naslovi ne odrežejo
plt.subplots_adjust(top=0.85) 

plt.savefig("slika_primerjava_full.png", dpi=150)
print("✅ Slika shranjena kot: slika_primerjava_full.png")