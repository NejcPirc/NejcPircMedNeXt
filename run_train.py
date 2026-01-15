import os
import sys
import glob
import argparse
import json
import torch
import torch.optim as optim
from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, EnsureTyped, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Uvoz modela (predpostavimo lokalno kopijo)
sys.path.append('./mednext_lib')  # Prilagodi, če je drugje
from MedNextV1 import MedNeXt  # Če je to custom, preveri ime modula

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="./data/nnUNet_raw/Podatki", help='Pot do nnU-Net podatkov')
parser.add_argument('--output_path', default="./output_final", help='Pot za shranjevanje modela')
parser.add_argument('--params_path', default="", help='Pot do params.json za hiperparametre')
args = parser.parse_args()

# Nalaganje parametrov iz JSON (če obstaja)
params = {
    'lr': 0.001,
    'epochs': 50,
    'batch_size': 1,
    'patch_size': (96, 96, 96),  # Povečaj na 128, če imaš dovolj GPU spomina
    'patience': 10  # Za early stopping
}
if args.params_path and os.path.exists(args.params_path):
    with open(args.params_path, 'r') as f:
        params.update(json.load(f))

# 1. Priprava podatkov
print("Iščem slike...")
slike = sorted(glob.glob(f"{args.data_path}/imagesTr/*.nii.gz"))
podatki = []
for slika in slike:
    labela = slika.replace("imagesTr", "labelsTr").replace("_0000.nii.gz", ".nii.gz")
    if os.path.exists(labela):
        podatki.append({"image": slika, "label": labela})

print(f"Našel {len(podatki)} parov za trening.")

# Split na train/val (80/20)
train_data, val_data = train_test_split(podatki, test_size=0.2, random_state=42)

# 2. Transformacije
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image", "label"], spatial_size=params['patch_size'], mode=("trilinear", "nearest")),
    EnsureTyped(keys=["image", "label"]),
])

# 3. Dataloaderji
train_ds = CacheDataset(data=train_data, transform=transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=2)

val_ds = CacheDataset(data=val_data, transform=transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

# 4. Priprava modela
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uporabljam device: {device}")

model = MedNeXt(
    in_channels=1, n_channels=32, n_classes=2, exp_r=2, kernel_size=3,
    deep_supervision=False, do_res=True, do_res_up_down=True,
    block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
loss_func = DiceCELoss(softmax=True, to_onehot_y=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")  # Za val metrike

# 5. Zanka učenja z validacijo
print("Začenjam učenje...")
best_val_dice = 0
patience_counter = 0

for epoch in tqdm(range(params['epochs']), desc="Napredek"):
    model.train()
    epoch_loss = 0
    steps = 0
    
    for batch in train_loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]  # Vzemi glavni output
        
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        steps += 1
    
    avg_loss = epoch_loss / steps
    tqdm.write(f"Epoha {epoch+1}: Train Loss = {avg_loss:.4f}")
    
    # Validacija
    model.eval()
    with torch.no_grad():
        for val_batch in val_loader:
            val_inputs = val_batch["image"].to(device)
            val_labels = val_batch["label"].to(device)
            
            # Sliding window za večje slike, če patch_size < original
            val_outputs = sliding_window_inference(val_inputs, params['patch_size'], 4, model)
            
            # Post-process za metrike
            post_trans = AsDiscrete(argmax=True, to_onehot=2)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_trans(i) for i in decollate_batch(val_labels)]
            
            dice_metric(y_pred=val_outputs, y=val_labels)
        
        val_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        tqdm.write(f"Epoha {epoch+1}: Val Dice = {val_dice:.4f}")
        
        # Early stopping
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output_path}/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= params['patience']:
                print("Early stopping sprožen.")
                break

# 6. Shranjevanje finalnega modela
os.makedirs(args.output_path, exist_ok=True)
torch.save(model.state_dict(), f"{args.output_path}/model_final.pth")
print("Model shranjen.")