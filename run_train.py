import os
# --- SPREMEMBA: Prisilimo uporabo GPU 1 (ker je GPU 0 zaseden) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
warnings.filterwarnings("ignore", module="torchvision")
import sys
import glob
import argparse
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, EnsureTyped,
    RandCropByPosNegLabeld, RandRotate90d, RandFlipd, RandShiftIntensityd, RandGaussianNoised
)
from monai.losses import DiceCELoss
from tqdm import tqdm

# Uvoz modela
sys.path.append('./mednext_lib')
from MedNextV1 import MedNeXt

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="./data/nnUNet_raw/Podatki", help='Pot do podatkov')
parser.add_argument('--output_path', default="./Izhod_Modeli", help='Pot za shranjevanje')
parser.add_argument('--params_path', default="", help='Pot do params.json')
args = parser.parse_args()

#### PARAMETRI ####
lr = 0.001          # Hitrost učenja
epochs = 200        # Število ponovitev 
batch_size = 1      # Ena slika naenkrat 
patch_size = (96, 96, 96) # Velikost izrezane kocke 


slike = sorted(glob.glob(f"{args.data_path}/imagesTr/*.nii.gz"))
podatki = []


# Povežemo vsako sliko z njeno labelo (rešitvijo)
for slika in slike:
    labela = slika.replace("imagesTr", "labelsTr").replace("_0000.nii.gz", ".nii.gz")
    if os.path.exists(labela):
        podatki.append({"image": slika, "label": labela})

#### Transformacije ####
transforms = Compose([
    LoadImaged(keys=["image", "label"]),           # Naloži Nifti datoteko
    EnsureChannelFirstd(keys=["image", "label"]),  # Kanali na prvo mesto
    ScaleIntensityd(keys=["image"]),               # Normalizacija svetlosti
    
    # Namesto cele slike (ki je prevelika),
    # izrežemo naključno kocko 96x96x96 v polni ločljivosti.
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=patch_size, pos=2, neg=1, num_samples=1,
        image_key="image", image_threshold=0,
    ),
    
    # AVGMENTACIJE: Slike naključno vrtimo in zrcalimo,
    # da se model nauči oblik
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]), #Vrtenje
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),          #Zrcaljenje
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),            #Svetlost
    RandGaussianNoised(keys=["image"], prob=0.1),                          #Šum
    
    EnsureTyped(keys=["image", "label"]),
])

ds = Dataset(data=podatki, transform=transforms)
loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

#### Model MedNeXt ####

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedNeXt(
    in_channels=1, n_channels=32, n_classes=2, exp_r=2, 
    kernel_size=5,          # Večje jedro (5) bolje vidi dolge, povezane žile --> probal s 3x3x3
    deep_supervision=True,  # Model se uči na 5 nivojih hkrati (hitrejše učenje)
    do_res=True, do_res_up_down=True,
    block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
loss_func = DiceCELoss(softmax=True, to_onehot_y=True)

#### Trening ####
print("Začenjam trening...")
os.makedirs(args.output_path, exist_ok=True)

best_loss = 1.0
pbar = tqdm(range(epochs), desc="Napredek")

for epoch in pbar:
    model.train()
    epoch_loss = 0
    steps = 0
    
    for batch in loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        # Deep Supervision
        loss = 0
        if isinstance(outputs, list):  # Če imamo več izhodov, seštejemo napake z utežmi
            weights = [1.0, 0.5, 0.25, 0.125, 0.0625]
            for i, out in enumerate(outputs):
                if i < len(weights):
                    if out.shape[2:] != labels.shape[2:]:  # Pomanjšamo labelo, da ustreza velikosti izhoda
                        target = F.interpolate(labels, size=out.shape[2:], mode='nearest')
                    else:
                        target = labels
                    loss += weights[i] * loss_func(out, target)
        else:
            loss = loss_func(outputs, labels)

        # Posodabljanje uteži (Backpropagation)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        steps += 1
    
    scheduler.step()
    avg_loss = epoch_loss / steps
    
    # Shrani najboljšega
    msg = ""
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"{args.output_path}/model_best.pth")
        msg = " (BEST)"
    
    pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Info": msg})

# Shrani zadnjega
torch.save(model.state_dict(), f"{args.output_path}/model_final.pth")
print("Trening končan.")