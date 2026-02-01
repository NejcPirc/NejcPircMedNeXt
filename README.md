# Segmentacija koronarnih arterij - MedNeXt
**Izziv AMS 2025**  
**Študent:** Nejc Pirc

Repozitorij vsebuje rešitev za avtomatsko segmentacijo koronarnih arterij na 3D CTA slikah z uporabo arhitekture MedNeXt. Rešitev je optimizirana za delovanje v Docker okolju na strežniški infrastrukturi.

---

##  1. Podatki

Zaradi velikosti dataseta se podatki ne kopirajo lokalno v kontejner, ampak se berejo neposredno z diska strežnika. Skripte so prilagojene za to strukturo.

*   **Lokacija na strežniku:** `/media/FastDataMama/izziv/data`
*   **Dataset:** ImageCAS (1000 3D slik)
*   **Uporabljena razdelitev (Split-1):**
    *   **Train (Učenje):** 750 slik (ID 1–750)
    *   **Inference/Val:** 50 slik (ID 751–800)
    *   **Test (Evalvacija):** 200 slik (ID 801–1000)

---

##  2. Navodila za zagon (Docker)

Celoten postopek je zapakiran v Docker kontejner. Za dostop do podatkov in grafične kartice so potrebne spodnje nastavitve.

```bash
docker build -t nejcpircmednext
```


---

## 3. Zagon programa

```bash
docker run --gpus all --ipc=host --rm \
  -v .:/workspace/ \
  -v /media/FastDataMama:/media/FastDataMama \
  nejcpircmednext python3 run_all.py
 ```


## 4. Opis datotek in skript

| Datoteka | Opis |
| :--- | :--- |
| `Priprava_Podatkov.py` | Pripravi strukturo map (`imagesTr`, `labelsTr`, `imagesTs`) v mapi `./data` in razporedi slike glede na Split-1. |
| `run_train.py` | Izvaja učenje modela. Uporablja Patch-based training (96×96×96),*Deep Supervision in napredne augmentacije. Shrani `model_best.pth`. |
| `run_inference.py` | Naloži naučen model in izvede segmentacijo na testnih slikah z uporabo metode Sliding Window Inference (drsno okno). |
| `run_test.py` | Primerja napovedi z referenčnimi maskami. Izračuna Dice Score ter shrani rezultate v `metrics.json`. |
| `run_all.py` | Krovna skripta, ki požene zgornje štiri korake v pravilnem vrstnem redu. |
| `Vizualizacija.py` | Generira sliko `slika_primerjava_full.png` za vizualno primerjavo (Original vs. GT vs. Napoved). |
| `Izris_Loss_Log.py` | Iz log datoteke izlušči podatke in izriše graf poteka učenja (Loss curve). |
| `Primerjava_Dice.py` | Izriše graf primerjave našega rezultata z nnU-Net benchmarkom. |
| `mednext_lib/` | Mapa, ki vsebuje definicijo arhitekture MedNeXt in gradnike (Blocks). |


## 5. O metodi (MedNeXt)

Za rešitev izziva je uporabljena arhitektura MedNeXt, ki je specializirana za segmentacijo tankih in povezanih struktur.

### Specifike naše implementacije:

* Kernel Size: 5×5×5 (zajame širši kontekst za ohranjanje kontinuitete žil).
* Arhitektura: ConvNeXt bloki z "Inverted Bottleneck" zasnovo.
* Deep Supervision: Učenje na 5 nivojih globine za boljše zaznavanje detajlov in hitrejšo konvergenco.

### Primerjava z nnU-Net (Baseline):

Rezultati so primerjani z uradnim rezultatom nnU-Net na ImageCAS datasetu (Dice ~0.885).
Zaradi časovnih omejitev nismo trenirali nnU-Neta, so pa vključena navodila za njegovo reprodukcijo:


