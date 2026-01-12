import os, glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib.colors import ListedColormap

# 1. Poisci datoteke
# Iscemo v mapi z rezultati
pred_files = sorted(glob.glob("./predictions_final/*.nii.gz"))

# Vzamemo 5. sliko (indeks 4)
pred_path = pred_files[1]
filename = os.path.basename(pred_path)
print(f"Prikazujem: {filename}")

# Sestavimo poti do originala in labele
img_path = f"./data/nnUNet_raw/Podatki/imagesTs/{filename}"
# Labela nima končnice _0000, zato zamenjamo string
lbl_path = f"./data/nnUNet_raw/Podatki/test_labels_ref/{filename.replace('_0000.nii.gz', '.nii.gz')}"

# 2. Funkcija za nalaganje (srednja rezina + rotacija)
def get_slice(path):
    data = nib.load(path).get_fdata()
    mid = data.shape[2] // 2
    return np.rot90(data[:, :, mid])

# Nalozi slike
img = get_slice(img_path)
lbl = get_slice(lbl_path)
pred = get_slice(pred_path)

# 3. Obdelava (Resize in Kontrast)
# Povečamo labelo in napoved na velikost originalne slike
lbl_resized = resize(lbl, img.shape, order=0, preserve_range=True, anti_aliasing=False)
pred_resized = resize(pred, img.shape, order=0, preserve_range=True, anti_aliasing=False)

# Malo izboljšamo kontrast CT slike
img = np.clip(img, -500, 800)

# Barve
green = ListedColormap(['lime'])
red = ListedColormap(['red'])

# 4. Risanje
fig, ax = plt.subplots(1, 3, figsize=(20, 8))

# A) Original
ax[0].imshow(img, cmap='gray')
ax[0].set_title("CT Original", fontsize=20, fontweight='bold', pad=20)
ax[0].axis('off')

# B) nnU-Net Standard (Zeleno)
ax[1].imshow(img, cmap='gray')
ax[1].imshow(np.ma.masked_where(lbl_resized < 0.5, lbl_resized), cmap=green, alpha=1)
ax[1].set_title("nnU-Net Standard", fontsize=20, fontweight='bold', pad=20)
ax[1].axis('off')

# C) MedNeXt Metoda (Rdeče)
ax[2].imshow(img, cmap='gray')
ax[2].imshow(np.ma.masked_where(pred_resized < 0.5, pred_resized), cmap=red, alpha=1)
ax[2].set_title("MedNeXt Metoda", fontsize=20, fontweight='bold', pad=20)
ax[2].axis('off')

# Pustimo prostor zgoraj za naslove
plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.savefig("Primerjava MedNeXt v nnUNet.png")
print("Slika shranjena.")