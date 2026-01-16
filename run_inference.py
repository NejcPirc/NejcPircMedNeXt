import os, sys, glob, argparse
import torch
import numpy as np
import nibabel as nib
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, EnsureTyped

# Dodamo pot do modela (preprosto)
sys.path.append('./mednext_lib')
from MedNextV1 import MedNeXt

# Argumenti
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default="./data/nnUNet_raw/Podatki/imagesTs")
parser.add_argument('--model_path', default="./output_final/model_final.pth")
parser.add_argument('--output_path', default="./predictions_final")
args = parser.parse_args()

# 1. Priprava seznama slik
print("Iscem slike za inferenco...")
slike = sorted(glob.glob(f"{args.input_path}/*.nii.gz"))
podatki = [{"image": s} for s in slike]

print(f"Nasel {len(slike)} slik.")

# 2. Transformacije
# Nujno pomanjsanje na 96x96x96, ker je model tako treniran
transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
    EnsureTyped(keys=["image"]),
])

loader = DataLoader(Dataset(podatki, transform=transforms), batch_size=1, shuffle=False)

# 3. Priprava modela
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedNeXt(
    in_channels=1, n_channels=32, n_classes=2, exp_r=2, kernel_size=5,
    deep_supervision=False, do_res=True, do_res_up_down=True,
    block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
).to(device)

# Nalaganje uteži
print(f"Nalagam model iz {args.model_path}...")
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# 4. Napovedovanje
os.makedirs(args.output_path, exist_ok=True)

with torch.no_grad():
    for i, batch in enumerate(loader):
        inputs = batch["image"].to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Ce model vrne seznam, vzamemo prvega
        if isinstance(outputs, list):
            outputs = outputs[0]

        # Argmax: verjetnosti spremenimo v 0 ali 1 (segmentacija)
        pred = torch.argmax(outputs, dim=1).cpu().numpy()[0].astype(np.uint8)
        
        # Ime datoteke dobimo iz originalnega seznama
        ime_datoteke = os.path.basename(slike[i])
        pot_shranjevanja = f"{args.output_path}/{ime_datoteke}"
        
        # Shranimo kot Nifti (uporabimo identity matriko za affine)
        nib.save(nib.Nifti1Image(pred, np.eye(4)), pot_shranjevanja)
        print(f"Shranil: {ime_datoteke}")

print("Koncano.")