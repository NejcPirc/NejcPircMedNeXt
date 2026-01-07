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
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--params_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. PREBERI PARAMETRE
    with open(args.params_path, 'r') as f:
        params = json.load(f)
    print(f"⚙️  Parametri: {params}")

    # 2. PRIPRAVA PODATKOV (Iskanje datotek)
    # Predvidevamo strukturo: data_path/imagesTr in data_path/labelsTr
    images_dir = os.path.join(args.data_path, "imagesTr")
    labels_dir = os.path.join(args.data_path, "labelsTr")
    
    # Poiščemo vse slike (npr. Dummy_000_0000.nii.gz)
    # Pazi: nnU-Net ima končnico _0000 za slike, labele pa ne.
    train_images = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    train_labels = []
    
    # Za vsako sliko poiščemo ustrezno labelo
    data_dicts = []
    for img_path in train_images:
        filename = os.path.basename(img_path)
        # Odstranimo _0000.nii.gz, da dobimo ID (npr. Dummy_000)
        case_id = filename.replace("_0000.nii.gz", "")
        lbl_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
        
        if os.path.exists(lbl_path):
            data_dicts.append({"image": img_path, "label": lbl_path})
        else:
            print(f"⚠️ Opozorilo: Manjka labela za {filename}")

    if len(data_dicts) == 0:
        print("❌ NAPAKA: Nisem našel nobenih parov slik in label!")
        sys.exit(1)
        
    print(f"✅ Najdenih {len(data_dicts)} parov za trening.")

    # 3. MONAI TRANSFORMS (Obdelava slik)
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),       
        EnsureChannelFirstd(keys=["image", "label"]), 
        ScaleIntensityd(keys=["image"]),           
        
        # --- DODANO: Spremenimo vse slike na fiksno velikost 128x128x128 ---
        # mode=('trilinear', 'nearest') pomeni: sliko lepo zgladi, masko (labelo) pa pusti celoštevilsko (0 ali 1)
        Resized(keys=["image", "label"], spatial_size=(96, 96, 96), mode=("trilinear", "nearest")),
        # -------------------------------------------------------------------
        
        EnsureTyped(keys=["image", "label"]),      
    ])

    # Dataset in DataLoader
    # CacheDataset je hiter, ker shrani slike v RAM
    train_ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)

    # 4. INICIALIZACIJA MODELA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Zaganjam na: {device}")

    model = MedNeXt(
        in_channels=1,      # CT slika ima 1 kanal (sivine)
        n_channels=32,      # Osnovno število filtrov (manjše za hitrejši test)
        n_classes=2,        # 2 razreda: 0=ozadje, 1=žila
        exp_r=2,            # Expansion ratio (iz MedNeXt papirja)
        kernel_size=3,      # Velikost jedra
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2] # Arhitektura (lahko prilagodimo)
    ).to(device)

    # 5. TRENING ZANKA (Loop)
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'])
    loss_function = nn.CrossEntropyLoss() # Preprosta loss funkcija za začetek

    print("🏁 Začenjam trening...")
    model.train()
    
    for epoch in range(params['max_epochs']):
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            # Labela mora biti 'long' tipa za CrossEntropy in brez kanala dimenzije
            # labels pride [Batch, 1, X, Y, Z], rabimo [Batch, X, Y, Z]
            labels = labels.squeeze(1).long() 

            optimizer.zero_grad()
            outputs = model(inputs) # Forward pass
            
            # MedNeXt lahko vrne seznam (zaradi deep supervision), vzamemo prvega
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = loss_function(outputs, labels)
            loss.backward()         # Backward pass
            optimizer.step()        # Update weights
            
            epoch_loss += loss.item()
            print(f"   Epoha {epoch+1}, Korak {step}, Loss: {loss.item():.4f}")

        print(f"📊 Konec epohe {epoch+1}, Povprečni Loss: {epoch_loss/step:.4f}")

    # 6. SHRANJEVANJE
    os.makedirs(args.output_path, exist_ok=True)
    save_path = os.path.join(args.output_path, "model_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model shranjen v: {save_path}")

if __name__ == '__main__':
    main()