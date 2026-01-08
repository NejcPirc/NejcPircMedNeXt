import os
import shutil
import glob
from tqdm import tqdm

# --- NASTAVITVE ---
# Mape, kjer so slike
data_roots = [
    "/media/FastDataMama/izziv/data/1-200",
    "/media/FastDataMama/izziv/data/201-400",
    "/media/FastDataMama/izziv/data/401-600",
    "/media/FastDataMama/izziv/data/601-800",
    "/media/FastDataMama/izziv/data/801-1000"
]

# Kam kopiramo
target_base = "./data/nnUNet_raw/Dataset004_FinalTest"

# LIMITI (Koliko slik za vsako fazo)
LIMIT_TRAIN = 20
LIMIT_INFER = 5
LIMIT_TEST = 10

# --- 1. POIŠČI VSE SLIKE ---
print("🔍 Iščem slike na disku...")
all_images = []

for folder in data_roots:
    if os.path.exists(folder):
        # Najdemo vse pare slik in label
        imgs = sorted(glob.glob(os.path.join(folder, "*.img.nii.gz")))
        for img_path in imgs:
            # Preverimo, če obstaja labela
            lbl_path = img_path.replace(".img.nii.gz", ".label.nii.gz")
            if os.path.exists(lbl_path):
                all_images.append((img_path, lbl_path))

print(f"✅ Našel sem {len(all_images)} veljavnih parov (slika + labela).")

if len(all_images) < (LIMIT_TRAIN + LIMIT_INFER + LIMIT_TEST):
    print("❌ Premalo slik na disku za zahtevane limite!")
    exit()

# --- 2. PRIPRAVA MAP ---
train_out = os.path.join(target_base, "imagesTr")
labels_out = os.path.join(target_base, "labelsTr")
test_out = os.path.join(target_base, "imagesTs")
infer_out = os.path.join(target_base, "inference_input")
infer_lbl_ref = os.path.join(target_base, "inference_labels_ref")
test_lbl_ref = os.path.join(target_base, "test_labels_ref")

for d in [train_out, labels_out, test_out, infer_out, infer_lbl_ref, test_lbl_ref]:
    os.makedirs(d, exist_ok=True)

# --- 3. RAZDELITEV IN KOPIRANJE ---
print("🚀 Začenjam ročno delitev (bypass Excel)...")

current_idx = 0

# A) TRENING (Prvih 20)
print(f"📦 Kopiram {LIMIT_TRAIN} slik za TRENING...")
for i in tqdm(range(LIMIT_TRAIN)):
    src_img, src_lbl = all_images[current_idx]
    filename = os.path.basename(src_img)
    # Iz 100.img.nii.gz -> ID je 100
    file_id = filename.replace(".img.nii.gz", "")
    
    shutil.copy2(src_img, os.path.join(train_out, f"ImageCAS_{file_id}_0000.nii.gz"))
    shutil.copy2(src_lbl, os.path.join(labels_out, f"ImageCAS_{file_id}.nii.gz"))
    current_idx += 1

# B) INFERENCA (Naslednjih 5)
print(f"📦 Kopiram {LIMIT_INFER} slik za INFERENCO...")
for i in tqdm(range(LIMIT_INFER)):
    src_img, src_lbl = all_images[current_idx]
    filename = os.path.basename(src_img)
    file_id = filename.replace(".img.nii.gz", "")
    
    shutil.copy2(src_img, os.path.join(infer_out, f"ImageCAS_{file_id}_0000.nii.gz"))
    # Labelo shranimo posebej za referenco
    shutil.copy2(src_lbl, os.path.join(infer_lbl_ref, f"ImageCAS_{file_id}.nii.gz"))
    current_idx += 1

# C) TESTIRANJE (Naslednjih 10)
print(f"📦 Kopiram {LIMIT_TEST} slik za TESTIRANJE...")
for i in tqdm(range(LIMIT_TEST)):
    src_img, src_lbl = all_images[current_idx]
    filename = os.path.basename(src_img)
    file_id = filename.replace(".img.nii.gz", "")
    
    shutil.copy2(src_img, os.path.join(test_out, f"ImageCAS_{file_id}_0000.nii.gz"))
    # Labelo shranimo posebej za test
    shutil.copy2(src_lbl, os.path.join(test_lbl_ref, f"ImageCAS_{file_id}.nii.gz"))
    current_idx += 1

print("-" * 30)
print(f"✅ Končano! Podatki so v: {target_base}")