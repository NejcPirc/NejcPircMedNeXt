import json
import matplotlib.pyplot as plt
import os
import numpy as np

# --- POTI ---
my_metrics_path = "./metrics_final/metrics.json"

# --- 1. PREBERI TVOJ REZULTAT ---
my_score = 0.0
if os.path.exists(my_metrics_path):
    with open(my_metrics_path, 'r') as f:
        data = json.load(f)
        my_score = data.get("mean_dice", 0.0)
else:
    print(f"⚠️ Ne najdem datoteke: {my_metrics_path}")
    # Za testiranje damo fiktivno, če datoteke ni
    my_score = 0.5 

# --- 2. NNUNET BASELINE (Iz literature za ImageCAS) ---
# Uradni rezultat nnU-Net na ImageCAS je cca 88.5% Dice
nnunet_score = 0.885 

print(f"Tvoj MedNeXt Dice: {my_score:.4f}")
print(f"nnU-Net Baseline:  {nnunet_score:.4f}")

# --- 3. RISANJE GRAFA ---
methods = ['Tvoj MedNeXt\n(Naša implementacija)', 'nnU-Net\n(State-of-the-Art)']
scores = [my_score, nnunet_score]
colors = ['#FF5733', '#33FF57'] # Rdeča (ti), Zelena (oni)

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, scores, color=colors, alpha=0.9, width=0.6)

# Dodajanje črte za 0.88 (Cilj)
plt.axhline(y=nnunet_score, color='gray', linestyle='--', alpha=0.5, label='nnU-Net standard')

# Izpis vrednosti na stolpcih
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('Dice Score (Višje je bolje)', fontsize=12)
plt.title('Primerjava: Naš MedNeXt vs. nnU-Net Baseline', fontsize=14, fontweight='bold')
plt.ylim(0, 1.05) # Dice je max 1.0
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("primerjava_baseline.png", dpi=150)
print("✅ Graf shranjen: primerjava_baseline.png")