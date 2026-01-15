import os, glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib.colors import ListedColormap

# --- POTI ---
# Tvoje napovedi (rezultati)
pred_dir = "./predictions_final"  # Ali "./Napovedi_Koncne" - preveri, kje imas


# Kje so originalne slike (na FastDataMama)
izvorne_mape = [
    "/media/FastDataMama/izziv/data/1-200",
    "/media/FastDataMama/izziv/data/201-400",
    "/media/FastDataMama/izziv/data/401-600",
    "/media/FastDataMama/izziv/data/601-800",
    "/media/FastDataMama/izziv/data/801-1000"
]

# --- 1. IZBERI NAPOVED ---
napovedi = sorted(glob.glob(f"{pred_dir}/*.nii.gz"))

if len(napovedi) == 0:
    print(f"NAPAKA: V mapi {pred_dir} ni slik. Preveri ime mape!")
    exit()

# Izberemo sliko (spremeni indeks [0], [1], [2] za druge slike)
pot_napovedi = napovedi[2] 
ime_datoteke = os.path.basename(pot_napovedi) # npr. ImageCAS_113_0000.nii.gz
print(f"Prikazujem: {ime_datoteke}")

# --- 2. POIŠČI ORIGINALNO SLIKO IN LABELO ---
# Iz imena moramo dobiti ID. 
# ImageCAS_113_0000.nii.gz  -->  113
id_slike = ime_datoteke.replace("ImageCAS_", "").replace("_0000.nii.gz", "")

pot_originala = None
pot_labele = None

# Iscemo po vseh mapah na disku
for mapa in izvorne_mape:
    mozna_slika = f"{mapa}/{id_slike}.img.nii.gz"
    mozna_labela = f"{mapa}/{id_slike}.label.nii.gz"
    
    if os.path.exists(mozna_slika):
        pot_originala = mozna_slika
        pot_labele = mozna_labela
        break

if pot_originala is None:
    print(f"NAPAKA: Ne najdem originalne slike za ID {id_slike} na FastDataMama!")
    exit()

# --- 3. NALAGANJE IN OBDELAVA ---
def nalozi_rezino(pot):
    img = nib.load(pot)
    data = img.get_fdata()
    mid = data.shape[2] // 2
    return np.rot90(data[:, :, mid])

slika = nalozi_rezino(pot_originala)
labela = nalozi_rezino(pot_labele)
napoved = nalozi_rezino(pot_napovedi)

# Resize (ker je tvoj model delal na 96x96, original je 512x512)
labela_resized = resize(labela, slika.shape, order=0, preserve_range=True, anti_aliasing=False)
napoved_resized = resize(napoved, slika.shape, order=0, preserve_range=True, anti_aliasing=False)

# Kontrast
slika = np.clip(slika, -500, 800)

# Barve
zelena = ListedColormap(['lime'])
rdeca = ListedColormap(['red'])

# --- 4. RISANJE ---
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# A) Original
axes[0].imshow(slika, cmap='gray')
axes[0].set_title("CT Original", fontsize=20, fontweight='bold', pad=20)
axes[0].axis('off')

# B) nnU-Net Standard (Referenca)
axes[1].imshow(slika, cmap='gray')
axes[1].imshow(np.ma.masked_where(labela_resized < 0.5, labela_resized), cmap=zelena, alpha=1)
axes[1].set_title("nnU-Net Standard", fontsize=20, fontweight='bold', pad=20)
axes[1].axis('off')

# C) Tvoja metoda
axes[2].imshow(slika, cmap='gray')
axes[2].imshow(np.ma.masked_where(napoved_resized < 0.5, napoved_resized), cmap=rdeca, alpha=1)
axes[2].set_title("MedNeXt Metoda", fontsize=20, fontweight='bold', pad=20)
axes[2].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.85) 

plt.savefig("slika_primerjava_full.png")
print("Slika shranjena: slika_primerjava_full.png")