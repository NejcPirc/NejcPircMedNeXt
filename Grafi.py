import matplotlib.pyplot as plt
import json
import os

# Pot do tvojih rezultatov
pot_do_json = "./metrics_final/metrics.json"

# 1. Preberi tvoj rezultat (MedNeXt)
moj_dice = 0.0
if os.path.exists(pot_do_json):
    with open(pot_do_json, 'r') as f:
        podatki = json.load(f)
        # Preberemo povprečje
        moj_dice = podatki.get('mean_dice', 0.0)
else:
    print(f"OPOZORILO: Datoteka {pot_do_json} ne obstaja. Uporabljam 0.0.")

# 2. nnU-Net rezultat (Referenca iz literature za ImageCAS)
nnunet_dice = 0.885 

print(f"Tvoj MedNeXt Dice: {moj_dice:.4f}")
print(f"nnU-Net Baseline:  {nnunet_dice:.4f}")

# 3. Risanje Grafa
metode = ['MedNeXt (Tvoj)', 'nnU-Net (Standard)']
rezultati = [moj_dice, nnunet_dice]
barve = ['#ff4d4d', '#32cd32'] # Rdeča (ti), Zelena (nnU-Net)

plt.figure(figsize=(9, 6))
stolpci = plt.bar(metode, rezultati, color=barve, width=0.6)

# Dodajanje številk na vrh stolpcev
for stolpec in stolpci:
    visina = stolpec.get_height()
    plt.text(stolpec.get_x() + stolpec.get_width()/2, visina + 0.01, 
             f'{visina:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Lepotni popravki
plt.ylabel('Dice Score (0 do 1)', fontsize=12)
plt.title('Primerjava uspešnosti segmentacije', fontsize=14)
plt.ylim(0, 1.1) # Malo prostora zgoraj za številke
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Shrani
plt.tight_layout()
plt.savefig("Graf_Primerjava.png")
print("✅ Graf shranjen kot: Graf_Primerjava.png")