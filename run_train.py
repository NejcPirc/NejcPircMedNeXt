import argparse
import os
import sys
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from monai.data import DataLoader, Dataset, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, EnsureTyped, Resized
)
from monai.losses import DiceCELoss

# --- UVOZ MODELA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'mednext_lib'))

try:
    from MedNextV1 import MedNeXt
    # Uvozimo vse bloke, da ne bo težav z OutBlock
    from blocks import * 
    print("✅ Model uspešno uvožen.")
except ImportError as e:
    print(f"❌ NAPAKA: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser()
    # --- TUKAJ SO NOVE PRIVZETE VREDNOSTI (DEFAULTS) ---
    parser.add_argument('--data_path', type=str, default="./data/nnUNet_raw/Dataset004_FinalTest")
    parser.add_argument('--params_path', type=str, default="./params.json")
    parser.add_argument('--output_path', type=str, default="./output_final")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. PREBERI PARAMETRE
    if not os.path.exists(args.params_path):
        # Če params.json ne obstaja, ustvari privzetega
        default_params = {"learning_rate": 0.001, "max_epochs": 50, "batch_size": 1, "model_size": "S"}
        with open(args.params_path, 'w') as f:
            json.dump(default_params, f)
            
    with open(args.params_path, 'r') as f:
        params = json.load(f)
    print(f"⚙️  Parametri: {params}")

    # 2. PRIPRAVA PODATKOV
    images_dir = os.path.join(args.data_path, "imagesTr")
    labels_dir = os.path.join(args.data_path, "labelsTr")
    
    train_images = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    data_dicts = []
    
    for img_path in train_images:
        filename = os.path.basename(img_path)
        case_id = filename.replace("_0000.nii.gz", "")
        lbl_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
        
        if os.path.exists(lbl_path):
            data_dicts.append({"image": img_path, "label": lbl_path})

    if len(data_dicts) == 0:
        print("❌ NAPAKA: Nisem našel nobenih parov slik in label!")
        sys.exit(1)
        
    print(f"✅ Najdenih {len(data_dicts)} parov za trening.")

    # 3. MONAI TRANSFORMS
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),       
        EnsureChannelFirstd(keys=["image", "label"]), 
        ScaleIntensityd(keys=["image"]),           
        Resized(keys=["image", "label"], spatial_size=(96, 96, 96), mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "label"]),      
    ])

    train_ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)

    # 4. INICIALIZACIJA MODELA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Zaganjam na: {device}")

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

    # 5. TRENING
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'])
    # DiceCELoss rabi [Batch, Channel, X, Y, Z]
    loss_function = DiceCELoss(softmax=True, to_onehot_y=True)

    print("🏁 Začenjam trening...")
    model.train()
    
    for epoch in range(params['max_epochs']):
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoha {epoch+1}/{params['max_epochs']}, Loss: {epoch_loss/step:.4f}")

    # 6. SHRANJEVANJE
    os.makedirs(args.output_path, exist_ok=True)
    save_path = os.path.join(args.output_path, "model_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model shranjen v: {save_path}")

if __name__ == '__main__':
    main()