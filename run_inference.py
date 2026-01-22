import os, sys, glob, argparse
import torch
import numpy as np
import nibabel as nib
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, EnsureTyped
from monai.inferers import sliding_window_inference
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
warnings.filterwarnings("ignore", module="torchvision")
import warnings
warnings.filterwarnings('ignore')

sys.path.append('./mednext_lib')
from MedNextV1 import MedNeXt

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default="./data/nnUNet_raw/Podatki/imagesTs") # Testne slike
parser.add_argument('--model_path', default="./Izhod_Modeli/model_final.pth")  # Naučene uteži
parser.add_argument('--output_path', default="./Napovedi_Koncne") #napovedna segmentacije
args = parser.parse_args()

 
slike = sorted(glob.glob(f"{args.input_path}/*.nii.gz")) #nifti datoteka
podatki = [{"image": s} for s in slike]

print(f"Nasel {len(slike)} slik.")

if len(slike) == 0:
    print("Ni slik! Preveri pot.")
    sys.exit()

#### TRANSFORMACIJE (Pred-procesiranje) ####
# Tukaj NE uporabljamo Resize, ker želimo ohraniti originalno ločljivost slike za natančno segmentacijo.
transforms = Compose([
    LoadImaged(keys=["image"]),         # Naloži 3D sliko
    EnsureChannelFirstd(keys=["image"]),# Postavi kanale na prvo mesto (C, H, W, D)
    ScaleIntensityd(keys=["image"]),    # Normalizira vrednosti pikslov (0-1), da model lažje računa
    EnsureTyped(keys=["image"]),        # Pretvori v PyTorch Tenzor
])

loader = DataLoader(Dataset(podatki, transform=transforms), batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Ustvarimo "prazno" MedNeXt mrežo. Parametri morajo biti IDENTIČNI tistim pri treningu!
# kernel_size=5 in deep_supervision=True 
model = MedNeXt(
    in_channels=1, n_channels=32, n_classes=2, exp_r=2, kernel_size=5,
    deep_supervision=True, do_res=True, do_res_up_down=True,
    block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
).to(device)

#### Nalaganje uteži ####

if os.path.exists(args.model_path):
    print(f"Nalagam model: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
else:
    alt_path = args.model_path.replace("best", "final")
    if os.path.exists(alt_path):
        print(f"Model 'best' ne obstaja, nalagam 'final': {alt_path}")
        model.load_state_dict(torch.load(alt_path, map_location=device))
    else:
        print("NAPAKA: Noben model ne obstaja! Najprej zazeni trening.")
        sys.exit()

model.eval() # Preklopimo v način za preverjanje (izklopi dropout in posodabljanje uteži)

#### Inferenca ####
os.makedirs(args.output_path, exist_ok=True)
print("Zacinjam obdelavo...")

with torch.no_grad():
    for i, batch in enumerate(loader):
        inputs = batch["image"].to(device)
        ime = os.path.basename(slike[i])
        
        
        #### SLIDING WINDOW INFERENCE ####
        # Ker je originalna slika prevelika za v GPU (512x512x...), jo razrežemo na kocke (96x96x96).
        # Model predela vsako kocko posebej, funkcija pa jih nato sestavi nazaj v celo sliko.
        outputs = sliding_window_inference(
            inputs, 
            roi_size=(96, 96, 96), 
            sw_batch_size=4,      # Obdela 4 kocke naenkrat
            predictor=model, 
            overlap=0.25,         # 25% prekrivanje med kockami za lepše spoje
            progress=True         
        )
        
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        pred = torch.argmax(outputs, dim=1).cpu().numpy()[0].astype(np.uint8)
        
        nib.save(nib.Nifti1Image(pred, np.eye(4)), f"{args.output_path}/{ime}")
        print(f"-> Shranjeno.")

print("Koncano.")