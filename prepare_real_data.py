import os
import shutil
from tqdm import tqdm
import glob

# Nastavitve
source_dir = "/media/FastDataMama/izziv/data/1-200"
target_base = "./data/nnUNet_raw/Dataset002_MiniImageCAS"
images_out = os.path.join(target_base, "imagesTr")
labels_out = os.path.join(target_base, "labelsTr")

# Ustvari mape
os.makedirs(images_out, exist_ok=True)
os.makedirs(labels_out, exist_ok=True)

# Poišči vse slike (samo tiste s končnico .img.nii.gz)
# Omejimo se na prvih 10 za hiter test
all_files = sorted(glob.glob(os.path.join(source_dir, "*.img.nii.gz")))
selected_files = all_files[:10]  # Vzamemo samo 10 slik

print(f"Kopiram {len(selected_files)} slik iz ImageCAS za testiranje...")

for img_path in tqdm(selected_files):
    # img_path je npr: .../100.img.nii.gz
    filename = os.path.basename(img_path)
    
    # Dobimo ID (npr. "100")
    case_id = filename.replace(".img.nii.gz", "")
    
    # Konstruiramo ime labele (npr. .../100.label.nii.gz)
    lbl_path = os.path.join(source_dir, f"{case_id}.label.nii.gz")
    
    if not os.path.exists(lbl_path):
        print(f"⚠️ Opozorilo: Labela za {case_id} ne obstaja, preskakujem.")
        continue
    
    # --- PRIPRAVA NOVIH IMEN (nnU-Net format) ---
    # Slika: ImageCAS_100_0000.nii.gz
    # Labela: ImageCAS_100.nii.gz
    
    new_case_id = f"ImageCAS_{case_id}"
    
    new_img_name = f"{new_case_id}_0000.nii.gz"
    new_lbl_name = f"{new_case_id}.nii.gz"
    
    # Kopiranje
    shutil.copy2(img_path, os.path.join(images_out, new_img_name))
    shutil.copy2(lbl_path, os.path.join(labels_out, new_lbl_name))

print("✅ Končano! Podatki so pripravljeni v:", target_base)