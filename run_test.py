import os, glob, json, torch, argparse
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Resized, AsDiscreted
from monai.data import Dataset, DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
warnings.filterwarnings("ignore", module="torchvision")

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="./data/nnUNet_raw/Podatki/testne_labele")
parser.add_argument('--predictions_path', default="./Napovedi_Koncne") #rezultati inference
parser.add_argument('--output_path', default="./metrics_final")
args = parser.parse_args()

pred_files = sorted(glob.glob(os.path.join(args.predictions_path, "*.nii.gz"))) # Poiščemo vse .nii.gz datoteke v mapi z napovedmi

# Ustvarimo seznam parov: {napoved, labela}
# Iz imena napovedi (ImageCAS_123_0000.nii.gz) izluščimo ime labele (ImageCAS_123.nii.gz)
data_list = [{"pred": p, "label": os.path.join(args.data_path, os.path.basename(p).replace("_0000", ""))} for p in pred_files]

print(f"Testiram {len(data_list)} primerov...")

#### Transformacije ####
transforms = Compose([
    LoadImaged(keys=["pred", "label"]),
    EnsureChannelFirstd(keys=["pred", "label"]),
    Resized(keys=["pred", "label"], spatial_size=(96, 96, 96), mode="nearest"),
    EnsureTyped(keys=["pred", "label"]),
    AsDiscreted(keys=["pred", "label"], threshold=0.5) # Threshold: Vse verjetnosti nad 0.5 spremenimo v 1 (žila), ostalo v 0 (ozadje).
])
 
loader = DataLoader(Dataset(data_list, transform=transforms), batch_size=1, shuffle=False)


#### DICE ####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = {}

with torch.no_grad():
    for i, batch in enumerate(loader):
        pred = batch["pred"].to(device)
        label = batch["label"].to(device)
        
        # Preprosta formula: 2 * presek / vsota
        dice = (2.0 * (pred * label).sum()) / (pred.sum() + label.sum() + 1e-5)
        
        # Ime dobimo kar iz seznama 
        name = os.path.basename(pred_files[i])
        print(f"{name} -> {dice:.4f}")
        results[name] = dice.item()

avg = sum(results.values()) / len(results)
print(f"--- Povprecje: {avg:.4f} ---")

os.makedirs(args.output_path, exist_ok=True)
with open(os.path.join(args.output_path, "metrics.json"), "w") as f:  #JSON ZA RISANJE GRAFOV
    json.dump({"mean_dice": avg, "details": results}, f, indent=4)