import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Argumenti za CLI (pot do JSON-ov)
parser = argparse.ArgumentParser()
parser.add_argument('--mednext_json', default="./metrics_final/metrics_mednext.json", help='Pot do MedNeXt metrics JSON')
parser.add_argument('--nnunet_json', default="./metrics_final/metrics_nnunet.json", help='Pot do nnU-Net metrics JSON')
parser.add_argument('--output_dir', default="./graphs", help='Pot za shranjevanje grafov')
args = parser.parse_args()

# Ustvari mapo za grafe
os.makedirs(args.output_dir, exist_ok=True)

# Branje JSON-ov
with open(args.mednext_json, 'r') as f:
    mednext_data = json.load(f)
with open(args.nnunet_json, 'r') as f:
    nnunet_data = json.load(f)

# Povprečni Dice
mednext_avg = mednext_data["mean_dice"]
nnunet_avg = nnunet_data["mean_dice"]

# Detajli (Dice po slikah – sortiraj po imenu za primerjavo)
mednext_details = sorted(mednext_data["details"].items())
nnunet_details = sorted(nnunet_data["details"].items())

mednext_names, mednext_dices = zip(*mednext_details)
nnunet_names, nnunet_dices = zip(*nnunet_details)

# Preveri, če so imena enaka (za primerjavo)
assert mednext_names == nnunet_names, "Imena slik se ne ujemajo med metodama!"

# 1. Bar graf za povprečne Dice
fig, ax = plt.subplots(figsize=(8, 6))
methods = ['MedNeXt', 'nnU-Net']
averages = [mednext_avg, nnunet_avg]
ax.bar(methods, averages, color=['blue', 'orange'])
ax.set_ylabel('Povprečni Dice koeficient')
ax.set_title('Primerjava povprečnega Dice med MedNeXt in nnU-Net')
ax.set_ylim(0, 1)
for i, v in enumerate(averages):
    ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.savefig(os.path.join(args.output_dir, "bar_dice_comparison.png"))
plt.close()

# 2. Line graf za Dice po slikah
x = np.arange(len(mednext_names))  # Številke slik
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, mednext_dices, label='MedNeXt', marker='o', color='blue')
ax.plot(x, nnunet_dices, label='nnU-Net', marker='x', color='orange')
ax.set_xlabel('Številka slike')
ax.set_ylabel('Dice koeficient')
ax.set_title('Primerjava Dice po posameznih slikah med MedNeXt in nnU-Net')
ax.set_xticks(x)
ax.set_xticklabels(mednext_names, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "line_dice_comparison.png"))
plt.close()

print("Grafa shranjena v:", args.output_dir)