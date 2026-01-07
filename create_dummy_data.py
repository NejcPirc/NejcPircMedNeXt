import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Nastavitve
base_dir = "./data/nnUNet_raw/Dataset001_Dummy"
images_dir = os.path.join(base_dir, "imagesTr")
labels_dir = os.path.join(base_dir, "labelsTr")

# Koliko parov slik želimo (za test je dovolj 3)
num_samples = 3

# Velikost slike (zmanjšamo za test, da bo hitro - npr. 64x64x64)
# Prave slike so veliko večje, ampak za preverjanje kode je to super.
shape = (64, 64, 64)

print(f"Ustvarjam {num_samples} lažnih 3D slik v {base_dir}...")

for i in tqdm(range(num_samples)):
    # Ime datoteke: nnU-Net zahteva format Ime_0000.nii.gz za slike
    case_id = f"Dummy_{i:03d}"
    
    # 1. Ustvari naključno sliko (šum) - to simulira CT posnetek
    # Podatki so float32
    image_data = np.random.rand(*shape).astype(np.float32)
    
    # 2. Ustvari naključno masko (0 in 1) - to simulira označeno žilo
    # Podatki so ponavadi celi, npr. uint8
    label_data = np.random.randint(0, 2, size=shape).astype(np.uint8)
    
    # 3. Shrani kot Nifti (.nii.gz)
    # Uporabimo identično afino matriko (pove kje v prostoru je slika)
    affine = np.eye(4)
    
    img_nifti = nib.Nifti1Image(image_data, affine)
    lbl_nifti = nib.Nifti1Image(label_data, affine)
    
    # Shrani sliko (mora imeti končnico _0000.nii.gz za nnU-Net format)
    nib.save(img_nifti, os.path.join(images_dir, f"{case_id}_0000.nii.gz"))
    
    # Shrani masko (samo .nii.gz)
    nib.save(lbl_nifti, os.path.join(labels_dir, f"{case_id}.nii.gz"))

print("✅ Končano! Podatki so pripravljeni.")