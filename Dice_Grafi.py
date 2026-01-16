import matplotlib.pyplot as plt
import json
import os

# --- NASTAVITVE ---
# Pot do tvojih rezultatov (ki jih ustvari run_test.py)
pot_do_json = "./metrics_final/metrics.json"

# Uradni rezultat za nnU-Net na ImageCAS (iz literature)
nnunet_dice = 0.885 

# --- 1. PREBERI TVOJ REZULTAT ---
moj_dice = 0.0

if os.path.exists(pot_do_json):
    with open(pot_do_json, 'r') as f:
        podatki = json.load(f)
        # Preberemo povprecje iz jsona
        moj_dice = podatki.get("mean_dice", 0.0)
    print(f"Najden rezultat v datoteki: {moj_dice:.4f}")
else:
    print(f"OPOZORILO: Datoteka {pot_do_json} se ne obstaja.")
    print("Uporabljam vrednost 0.0 za prikaz grafa.")

# --- 2. RISANJE GRAFA ---
metode = ['MedNeXt (Tvoj)', 'nnU-Net (Standard)']
rezultati = [moj_dice, nnunet_dice]
barve = ['#ff4d4d', '#32cd32'] # Rdeca in Zelena

plt.figure(figsize=(8, 6))

# Narisemo stolpce
stolpci = plt.bar(metode, rezultati, color=barve, width=0.6)

# Dodamo stevilke na vrh vsakega stolpca
for stolpec in stolpci:
    visina = stolpec.get_height()
    plt.text(stolpec.get_x() + stolpec.get_width()/2, visina + 0.01, 
             f'{visina:.4f}', ha='center', va='bottom', fontweight='bold')

# Oznake in naslov
plt.ylabel('Dice Score (0 do 1)')
plt.title('Primerjava uspesnosti segmentacije (Split-1)')
plt.ylim(0, 1.1) # Da je na vrhu malo prostora
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Shranimo
ime_slike = "Graf_Primerjava_Dice.png"
plt.savefig(ime_slike, dpi=150)
print(f"✅ Graf shranjen kot: {ime_slike}")