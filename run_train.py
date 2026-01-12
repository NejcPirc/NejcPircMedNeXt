import os, sys, glob, argparse
import torch
import torch.optim as optim
from monai.data import DataLoader, CacheDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, EnsureTyped
from monai.losses import DiceCELoss
from tqdm import tqdm

# Uvoz modela (preprosta pot)
sys.path.append('./mednext_lib')
from MedNextV1 import MedNeXt

# Argumenti (nastavljeni na tvoje slovenske mape)
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="./data/nnUNet_raw/Podatki")
parser.add_argument('--output_path', default="./output_final")
# Params path sploh ne rabimo, ker bomo parametre napisali kar spodaj
parser.add_argument('--params_path', default="") 
args = parser.parse_args()

# --- PARAMETRI ---
lr = 0.001
epochs = 50
batch_size = 1

# 1. Priprava podatkov
print("Iscem slike...")
# Iscemo direktno v imagesTr
slike = sorted(glob.glob(f"{args.data_path}/imagesTr/*.nii.gz"))
podatki = []

for slika in slike:
    # Zamenjamo mapo in koncnico za labelo
    labela = slika.replace("imagesTr", "labelsTr").replace("_0000.nii.gz", ".nii.gz")
    if os.path.exists(labela):
        podatki.append({"image": slika, "label": labela})

print(f"Nasel {len(podatki)} parov za trening.")

# 2. Transformacije
# Nujno pomanjsanje na 96x96x96 zaradi spomina
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image", "label"], spatial_size=(96, 96, 96), mode=("trilinear", "nearest")),
    EnsureTyped(keys=["image", "label"]),
])

# 3. Dataloader
# CacheDataset nalozi slike v RAM, da je hitreje
ds = CacheDataset(data=podatki, transform=transforms, cache_rate=1.0)
loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

# 4. Priprava modela
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedNeXt(
    in_channels=1, n_channels=32, n_classes=2, exp_r=2, kernel_size=3,
    deep_supervision=False, do_res=True, do_res_up_down=True,
    block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr)
loss_func = DiceCELoss(softmax=True, to_onehot_y=True)

# 5. Zanka učenja
print("Zacinjam ucenje...")
model.train()

for epoch in tqdm(range(epochs), desc="Napredek"):
    epoch_loss = 0
    korak = 0
    
    for batch in loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Ce model vrne seznam, vzamemo prvega
        if isinstance(outputs, list):
            outputs = outputs[0]

        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        korak += 1
    
    tqdm.write(f" Epoha {epoch+1}: Loss = {epoch_loss/korak:.4f}")

# 6. Shranjevanje
os.makedirs(args.output_path, exist_ok=True)
torch.save(model.state_dict(), f"{args.output_path}/model_final.pth")
print("Model shranjen.")