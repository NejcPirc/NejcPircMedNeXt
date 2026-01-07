import argparse
import os
import sys
import glob
import torch
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, EnsureTyped, Resized, AsDiscrete
)
from monai.data import Dataset, DataLoader, decollate_batch

# --- UVOZ MODELA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'mednext_lib'))

try:
    from MedNextV1 import MedNeXt
    print("✅ Model uspešno uvožen.")
except ImportError as e:
    print(f"❌ NAPAKA: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help="Mapa s slikami za napoved")
    parser.add_argument('--model_path', type=str, required=True, help="Pot do .pth datoteke")
    parser.add_argument('--output_path', type=str, required=True, help="Kam shraniti rezultate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. PRIPRAVA DATOTEK
    images = sorted(glob.glob(os.path.join(args.input_path, "*.nii.gz")))
    if len(images) == 0:
        print("❌ Nisem našel slik v input_path!")
        sys.exit(1)
        
    print(f"🔍 Našel {len(images)} slik za inferenco.")
    
    data_dicts = [{"image": img} for img in images]

    # 2. TRANSFORMS (Mora biti enako kot pri treningu!)
    # Pazi: Tukaj nimamo "label", zato transformacije samo za "image"
    infer_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"), # Ista velikost kot pri treningu
        EnsureTyped(keys=["image"]),
    ])

    ds = Dataset(data=data_dicts, transform=infer_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # 3. NALOŽI MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Pazi: Parametri morajo biti isti kot pri treningu!
    model = MedNeXt(
        in_channels=1,
        n_channels=32,
        n_classes=2,
        exp_r=2,
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
    ).to(device)

    print(f"📂 Nalagam uteži iz: {args.model_path}")
    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval() # Preklop v eval mode (izklopi dropout itd.)

    # 4. NAPOVEDOVANJE
    os.makedirs(args.output_path, exist_ok=True)
    
    print("🚀 Začenjam inferenco...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["image"].to(device)
            original_path = data_dicts[i]["image"]
            filename = os.path.basename(original_path)
            
            # Forward pass
            outputs = model(inputs)
            
            # Če model vrne listo (deep supervision), vzemi prvega
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                outputs = outputs[0]

            # Argmax: Spremeni verjetnosti v razred (0 ali 1)
            # outputs shape: [1, 2, 96, 96, 96] -> [1, 96, 96, 96]
            preds = torch.argmax(outputs, dim=1)
            
            # --- SHRANJEVANJE ---
            # Za pravi izziv bi morali resize-ati nazaj na originalno velikost,
            # ampak za demonstracijo shranimo kar pomanjšano (96x96x96).
            
            # Uporabimo affine matriko iz originalne slike (da ohranimo orientacijo)
            # Ker smo uporabili MONAI LoadImage, je to malo bolj zapleteno, 
            # zato za demo uporabimo poenostavljen shranjevalnik z Identity affine.
            
            pred_np = preds.cpu().numpy()[0].astype(np.uint8)
            
            save_name = os.path.join(args.output_path, filename)
            
            # Ustvari Nifti sliko
            nib_img = nib.Nifti1Image(pred_np, np.eye(4)) 
            nib.save(nib_img, save_name)
            
            print(f"💾 Shranil: {save_name}")

    print("✅ Končano!")

if __name__ == '__main__':
    main()