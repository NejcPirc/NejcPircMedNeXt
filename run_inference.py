import argparse
import os
import sys
import glob
import torch
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, EnsureTyped, Resized
)
from monai.data import Dataset, DataLoader

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
    # --- PRIVZETE POTI ---
    # Vzamemo slike iz TESTNEGA seta (imagesTs)
    parser.add_argument('--input_path', type=str, default="./data/nnUNet_raw/Dataset004_FinalTest/imagesTs")
    parser.add_argument('--model_path', type=str, default="./output_final/model_final.pth")
    parser.add_argument('--output_path', type=str, default="./predictions_final")
    return parser.parse_args()

def main():
    args = parse_args()
    
    images = sorted(glob.glob(os.path.join(args.input_path, "*.nii.gz")))
    if len(images) == 0:
        print(f"❌ Nisem našel slik v {args.input_path}!")
        sys.exit(1)
        
    print(f"🔍 Našel {len(images)} slik za inferenco.")
    
    data_dicts = [{"image": img} for img in images]

    infer_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
        EnsureTyped(keys=["image"]),
    ])

    ds = Dataset(data=data_dicts, transform=infer_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MedNeXt(
        in_channels=1, n_channels=32, n_classes=2, exp_r=2, kernel_size=3,
        deep_supervision=False, do_res=True, do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
    ).to(device)

    print(f"📂 Nalagam uteži iz: {args.model_path}")
    if not os.path.exists(args.model_path):
        print("❌ Model ne obstaja! Najprej zaženi run_train.py")
        sys.exit(1)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    os.makedirs(args.output_path, exist_ok=True)
    
    print("🚀 Začenjam inferenco...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["image"].to(device)
            original_path = data_dicts[i]["image"]
            filename = os.path.basename(original_path)
            
            outputs = model(inputs)
            if isinstance(outputs, list): outputs = outputs[0]

            preds = torch.argmax(outputs, dim=1)
            pred_np = preds.cpu().numpy()[0].astype(np.uint8)
            
            save_name = os.path.join(args.output_path, filename)
            nib.save(nib.Nifti1Image(pred_np, np.eye(4)), save_name)
            print(f"💾 Shranil: {save_name}")

    print("✅ Končano!")

if __name__ == '__main__':
    main()