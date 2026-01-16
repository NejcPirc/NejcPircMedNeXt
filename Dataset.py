import os, shutil, glob
from tqdm import tqdm

# --- 1. NASTAVITVE POTI ---
izvorne_mape = [
    "/media/FastDataMama/izziv/data/1-200",
    "/media/FastDataMama/izziv/data/201-400",
    "/media/FastDataMama/izziv/data/401-600",
    "/media/FastDataMama/izziv/data/601-800",
    "/media/FastDataMama/izziv/data/801-1000"
]

# Glavna mapa za podatke (Slovensko ime)
glavna_mapa = "./data/nnUNet_raw/Podatki"

# Priprava map (Počistimo staro, če obstaja)
if os.path.exists(glavna_mapa):
    shutil.rmtree(glavna_mapa)

# Ustvarimo podmape
# Opomba: imagesTr/Ts so standardna imena, ki jih raje pustimo zaradi kompatibilnosti
os.makedirs(f"{glavna_mapa}/imagesTr", exist_ok=True)       # Trening slike
os.makedirs(f"{glavna_mapa}/labelsTr", exist_ok=True)       # Trening maske
os.makedirs(f"{glavna_mapa}/imagesTs", exist_ok=True)       # Testne slike (15 kom)
os.makedirs(f"{glavna_mapa}/vhod_inferenca", exist_ok=True) # Inferenca slike (5 kom)
os.makedirs(f"{glavna_mapa}/testne_labele", exist_ok=True)  # Rešitve za test

# --- 2. ISKANJE IN SORTIRANJE ---
datoteke = []
print("Iščem datoteke na disku...")
for koren in izvorne_mape:
    if os.path.exists(koren):
        slike = glob.glob(f"{koren}/*.img.nii.gz")
        for slika in slike:
            labela = slika.replace(".img.nii.gz", ".label.nii.gz")
            if os.path.exists(labela):
                datoteke.append((slika, labela))

# Sortiranje po številki v imenu (nujno za ponovljivost!)
datoteke.sort(key=lambda x: int(os.path.basename(x[0]).split('.')[0]))

print(f"Našel {len(datoteke)} parov.")

# --- 3. KOPIRANJE (20 + 5 + 15 = 40) ---
limit = min(20, len(datoteke))
print(f"Pripravljam {limit} slik...")

for i in tqdm(range(limit)):
    izvorna_slika, izvorna_labela = datoteke[i]
    
    # ID slike (npr. 100)
    original_id = os.path.basename(izvorna_slika).split('.')[0]
    
    # Nova imena
    nova_slika = f"ImageCAS_{original_id}_0000.nii.gz"
    nova_labela = f"ImageCAS_{original_id}.nii.gz"

    # --- LOGIKA RAZDELITVE ---
    
    # 1. TRENING: Prvih 20 (indeksi 0-19)
    if i < 750:
        shutil.copy(izvorna_slika, f"{glavna_mapa}/imagesTr/{nova_slika}")
        shutil.copy(izvorna_labela, f"{glavna_mapa}/labelsTr/{nova_labela}")
        
    # 2. INFERENCA: Naslednjih 5 (indeksi 20-24)
    elif i < 800:
        shutil.copy(izvorna_slika, f"{glavna_mapa}/vhod_inferenca/{nova_slika}")
        
    # 3. TEST: Preostalih 15 (indeksi 25-39)
    else:
        shutil.copy(izvorna_slika, f"{glavna_mapa}/imagesTs/{nova_slika}")
        shutil.copy(izvorna_labela, f"{glavna_mapa}/testne_labele/{nova_labela}")

print(f"Priprava zaključena. Podatki so v: {glavna_mapa}")      