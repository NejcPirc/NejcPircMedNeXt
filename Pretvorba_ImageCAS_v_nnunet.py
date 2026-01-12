import os, shutil, glob
from tqdm import tqdm

# Mape, kjer so originalni podatki
roots = [
    "/media/FastDataMama/izziv/data/1-200",
    "/media/FastDataMama/izziv/data/201-400",
    "/media/FastDataMama/izziv/data/401-600",
    "/media/FastDataMama/izziv/data/601-800",
    "/media/FastDataMama/izziv/data/801-1000"
]

out_dir = "./data/nnUNet_raw/Podatki"

# Priprava map
os.makedirs(f"{out_dir}/imagesTr", exist_ok=True)
os.makedirs(f"{out_dir}/labelsTr", exist_ok=True)
os.makedirs(f"{out_dir}/imagesTs", exist_ok=True)
os.makedirs(f"{out_dir}/inference_in", exist_ok=True)
os.makedirs(f"{out_dir}/test_labels_ref", exist_ok=True)

# 1. Najdi vse slike
files = []
for root in roots:
    if os.path.exists(root):
        # Iscemo pare .img in .label
        imgs = sorted(glob.glob(f"{root}/*.img.nii.gz"))
        for img in imgs:
            lbl = img.replace(".img.nii.gz", ".label.nii.gz")
            if os.path.exists(lbl):
                files.append((img, lbl))

print(f"{len(files)} Slik ")

# --- SPREMEMBA TUKAJ ---
# Skupaj 1000 slik
# 0 - 750:   Training (750 slik)
# 750 - 800: Inference (50 slik)
# 800 - 1000: Test (200 slik)
# Zanka gre zdaj do 1000 (oziroma kolikor je vseh slik, če jih je manj)
limit = min(25, len(files)) 

for i in tqdm(range(limit)):
    img_src, lbl_src = files[i]
    
    # Dobimo ID
    name = os.path.basename(img_src).replace(".img.nii.gz", "")
    
    # Imena za cilj
    dst_img_name = f"ImageCAS_{name}_0000.nii.gz"
    dst_lbl_name = f"ImageCAS_{name}.nii.gz"

    if i < 10:
        # TRENING (Prvih 750 slik)
        shutil.copy(img_src, f"{out_dir}/imagesTr/{dst_img_name}")
        shutil.copy(lbl_src, f"{out_dir}/labelsTr/{dst_lbl_name}")
        
    elif i < 15:
        # INFERENCA (Naslednjih 50 slik -> 750 + 50 = 800)
        shutil.copy(img_src, f"{out_dir}/inference_in/{dst_img_name}")
        
    else:
        # TEST (Preostalih 200 slik -> do 1000)
        shutil.copy(img_src, f"{out_dir}/imagesTs/{dst_img_name}")
        shutil.copy(lbl_src, f"{out_dir}/test_labels_ref/{dst_lbl_name}")