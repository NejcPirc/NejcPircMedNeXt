import os, glob, json, torch, argparse
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Resized, AsDiscreted
from monai.data import Dataset, DataLoader

# Preprosti argumenti
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="./data/nnUNet_raw/Podatki/test_labels_ref")
parser.add_argument('--predictions_path', default="./predictions_final")
parser.add_argument('--output_path', default="./metrics_final")
args = parser.parse_args()

# 1. Priprava seznama (List comprehension)
pred_files = sorted(glob.glob(os.path.join(args.predictions_path, "*.nii.gz")))
data_list = [{"pred": p, "label": os.path.join(args.data_path, os.path.basename(p).replace("_0000", ""))} for p in pred_files]

print(f"Testiram {len(data_list)} primerov...")

# 2. Transformacije (Nujni resize na 96x96x96)
transforms = Compose([
    LoadImaged(keys=["pred", "label"]),
    EnsureChannelFirstd(keys=["pred", "label"]),
    Resized(keys=["pred", "label"], spatial_size=(96, 96, 96), mode="nearest"),
    EnsureTyped(keys=["pred", "label"]),
    AsDiscreted(keys=["pred", "label"], threshold=0.5)
])

# 3. Zanka
loader = DataLoader(Dataset(data_list, transform=transforms), batch_size=1, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = {}

with torch.no_grad():
    for i, batch in enumerate(loader):
        pred = batch["pred"].to(device)
        label = batch["label"].to(device)
        
        # Preprosta formula: 2 * presek / vsota
        dice = (2.0 * (pred * label).sum()) / (pred.sum() + label.sum() + 1e-5)
        
        # Ime dobimo kar iz seznama (ker je shuffle=False)
        name = os.path.basename(pred_files[i])
        print(f"{name} -> {dice:.4f}")
        results[name] = dice.item()

# 4. Povprečje in shranjevanje
avg = sum(results.values()) / len(results)
print(f"--- Povprecje: {avg:.4f} ---")

os.makedirs(args.output_path, exist_ok=True)
with open(os.path.join(args.output_path, "metrics.json"), "w") as f:
    json.dump({"mean_dice": avg, "details": results}, f, indent=4)