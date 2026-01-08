import os
import shutil
import glob
from tqdm import tqdm

# --- NAVODILA ---
# Ta skripta pretvori originalni ImageCAS dataset v format, 
# ki ga zahteva nnU-Net (imagesTr, labelsTr, imagesTs).
# To omogoča neposredno uporabo nnU-Net orodij.

# Poti do originalnih podatkov
data_roots = [
    "/media/FastDataMama/izziv/data/1-200",
    "/media/FastDataMama/izziv/data/201-400",
    "/media/FastDataMama/izziv/data/401-600",
    "/media/FastDataMama/izziv/data/601-800",
    "/media/FastDataMama/izziv/data/801-1000"
]

# Izhodna pot (nnU-Net struktura)
target_base = "./data/nnUNet_raw/Dataset004_FinalTest"

# Limiti za našo implementacijo (zaradi hitrosti)
# V pravi nnU-Net uporabi bi kopirali vse.
LIMIT_TRAIN = 20
LIMIT_TEST = 10

def main():
    print("🔄 Začenjam pretvorbo ImageCAS -> nnU-Net format...")
    
    # 1. Priprava map
    train_img_dir = os.path.join(target_base, "imagesTr")
    train_lbl_dir = os.path.join(target_base, "labelsTr")
    test_img_dir = os.path.join(target_base, "imagesTs")
    # Za potrebe evalvacije kopiramo testne labele sem (nnU-Net tega sicer ne zahteva v mapi raw)
    test_lbl_ref = os.path.join(target_base, "test_labels_ref")

    for d in [train_img_dir, train_lbl_dir, test_img_dir, test_lbl_ref]:
        os.makedirs(d, exist_ok=True)

    # 2. Iskanje slik
    all_pairs = []
    for folder in data_roots:
        if os.path.exists(folder):
            imgs = sorted(glob.glob(os.path.join(folder, "*.img.nii.gz")))
            for img_path in imgs:
                lbl_path = img_path.replace(".img.nii.gz", ".label.nii.gz")
                if os.path.exists(lbl_path):
                    all_pairs.append((img_path, lbl_path))

    print(f"📄 Našel {len(all_pairs)} parov slik.")

    # 3. Kopiranje
    # Za Train (imagesTr + labelsTr)
    print(f"📦 Pripravljam 'imagesTr' in 'labelsTr' ({LIMIT_TRAIN} primerov)...")
    for i in tqdm(range(LIMIT_TRAIN)):
        src_img, src_lbl = all_pairs[i]
        case_id = os.path.basename(src_img).replace(".img.nii.gz", "")
        
        # nnU-Net konvencija: case_identifier_0000.nii.gz
        dst_img = os.path.join(train_img_dir, f"ImageCAS_{case_id}_0000.nii.gz")
        dst_lbl = os.path.join(train_lbl_dir, f"ImageCAS_{case_id}.nii.gz")
        
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

    # Za Test (imagesTs)
    print(f"📦 Pripravljam 'imagesTs' ({LIMIT_TEST} primerov)...")
    # Vzamemo slike od konca, da niso iste kot za trening
    start_idx = LIMIT_TRAIN
    for i in tqdm(range(start_idx, start_idx + LIMIT_TEST)):
        src_img, src_lbl = all_pairs[i]
        case_id = os.path.basename(src_img).replace(".img.nii.gz", "")
        
        dst_img = os.path.join(test_img_dir, f"ImageCAS_{case_id}_0000.nii.gz")
        # Labelo shranimo za našo referenco (evalvacijo)
        dst_lbl_ref = os.path.join(test_lbl_ref, f"ImageCAS_{case_id}.nii.gz")
        
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl_ref)

    print(f"✅ Pretvorba končana! Podatki so v: {target_base}")

if __name__ == "__main__":
    main()