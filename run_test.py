import argparse
import os
import sys
import glob
import json
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, EnsureTyped, Resized, AsDiscreted
)
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    # --- PRIVZETE POTI ---
    # Pazi: Tu moramo pokazati na mapo, kjer so testne labele (test_labels_ref)
    parser.add_argument('--data_path', type=str, default="./data/nnUNet_raw/Dataset004_FinalTest/test_labels_ref")
    parser.add_argument('--model_path', type=str, default="./output_final/model_final.pth")
    parser.add_argument('--output_path', type=str, default="./metrics_final")
    parser.add_argument('--predictions_path', type=str, default="./predictions_final")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Preverimo, kje so labele
    if len(glob.glob(os.path.join(args.data_path, "*.nii.gz"))) > 0:
        gt_dir = args.data_path
    else:
        gt_dir = os.path.join(args.data_path, "labelsTr")

    pred_dir = args.predictions_path
    print(f"🔍 Primerjam: {pred_dir} <---> {gt_dir}")

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    
    if len(pred_files) == 0:
        print("❌ Ni napovedi! Najprej zaženi run_inference.py")
        sys.exit(1)

    data_dicts = []
    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        label_filename = filename.replace("_0000.nii.gz", ".nii.gz")
        gt_path = os.path.join(gt_dir, label_filename)
        
        if os.path.exists(gt_path):
            data_dicts.append({"pred": pred_path, "label": gt_path})
        else:
            print(f"⚠️ Manjka labela za: {label_filename}")

    if len(data_dicts) == 0:
        print("❌ Ni parov!")
        sys.exit(1)

    print(f"✅ Ocenjujem {len(data_dicts)} primerov.")

    eval_transforms = Compose([
        LoadImaged(keys=["pred", "label"]),
        EnsureChannelFirstd(keys=["pred", "label"]),
        Resized(keys=["label"], spatial_size=(96, 96, 96), mode="nearest"),
        Resized(keys=["pred"], spatial_size=(96, 96, 96), mode="nearest"),
        EnsureTyped(keys=["pred", "label"]),
        AsDiscreted(keys=["pred", "label"], threshold=0.5)
    ])

    ds = Dataset(data=data_dicts, transform=eval_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            pred = batch["pred"].to(device)
            label = batch["label"].to(device)
            
            intersection = (pred * label).sum()
            union = pred.sum() + label.sum()
            epsilon = 1e-5
            my_dice = (2. * intersection + epsilon) / (union + epsilon)
            
            filename = os.path.basename(data_dicts[i]["pred"])
            print(f"📄 {filename} -> Dice: {my_dice.item():.4f}")
            results[filename] = my_dice.item()

    avg_dice = sum(results.values()) / len(results)
    print("-" * 30)
    print(f"🏆 POVPREČNI DICE SCORE: {avg_dice:.4f}")
    print("-" * 30)

    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "metrics.json"), "w") as f:
        json.dump({"mean_dice": avg_dice, "details": results}, f, indent=4)

if __name__ == '__main__':
    main()