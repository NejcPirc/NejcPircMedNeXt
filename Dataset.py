import os
import sys
import glob
import argparse
import json
import torch
import torch.optim as optim
from monai.data import DataLoader, CacheDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, EnsureTyped
from monai.losses import DiceCELoss
from tqdm import tqdm

# Uvoz modela
sys.path.append('./mednext_lib')
from MedNextV1 import MedNeXt

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="./data/nnUNet_raw/Podatki", help='Pot do nnU-Net podatkov')
parser.add_argument('--output_path', default="./output_final", help='Pot za shranjevanje modela')
parser.add_argument('--params_path', default="", help='Pot do params.json za hiperparametre')
args = parser.parse_args()

# Nalaganje parametrov iz JSON
params = {
    'lr': 0.001,
    'epochs': 50,
    'batch_size': 1,
    'patch_size': (96, 96, 96),
}
if args.params_path and os.path.exists(args.params_path):
    with open(args.params_path, 'r') as f:
        params.update(json.load(f))

# Ustvari mapo na začetku (za finalno shranjevanje)
os.makedirs(args.output_path, exist_ok=True)

# 1. Podatki (brez splita – vsi za trening)
print("Iščem slike...")
slike = sorted(glob.glob(f"{args.data_path}/imagesTr/*.nii.gz"))
podatki = []
for slika in slike:
    labela = slika.replace("imagesTr", "labelsTr").replace("_0000.nii.gz", ".nii.gz")
    if os.path.exists(labela):
        podatki.append({"image": slika, "label": labela})

print(f"Našel {len(podatki)} parov za trening (vsi uporabljeni).")

# 2. Transformacije
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image", "label"], spatial_size=params['patch_size'], mode=("trilinear", "nearest")),
    EnsureTyped(keys=["image", "label"]),
])

ds = CacheDataset(data=podatki, transform=transforms, cache_rate=1.0)
loader = DataLoader(ds, batch_size=params['batch_size'], shuffle=True, num_workers=2)

# 3. Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = MedNeXt(
    in_channels=1, n_channels=32, n_classes=2, exp_r=2, kernel_size=3,
    deep_supervision=False, do_res=True, do_res_up_down=True,
    block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
loss_func = DiceCELoss(softmax=True, to_onehot_y=True)

# 4. Trening (samo loss, brez sproti shranjevanja)
print("Začenjam učenje...")
for epoch in tqdm(range(params['epochs']), desc="Napredek"):
    model.train()
    epoch_loss = 0
    steps = 0
    
    for batch in loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        steps += 1
    
    avg_loss = epoch_loss / steps
    tqdm.write(f"Epoha {epoch+1}: Loss = {avg_loss:.4f}")

# Finalni model (samo na koncu)
torch.save(model.state_dict(), os.path.join(args.output_path, "model_final.pth"))
print("Finalni model shranjen.")