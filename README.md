# Segmentacija koronarnih arterij - MedNeXt
**Izziv AMS 2025**  
**Študent:** Nejc Pirc

Repozitorij vsebuje rešitev za avtomatsko segmentacijo koronarnih arterij na 3D CTA slikah z uporabo arhitekture **MedNeXt**. Rešitev je optimizirana za delovanje v Docker okolju na strežniški infrastrukturi.

---

## 📂 1. Podatki

Zaradi velikosti dataseta se podatki **ne kopirajo lokalno** v kontejner, ampak se berejo neposredno z diska strežnika. Skripte so prilagojene za to strukturo.

*   **Lokacija na strežniku:** `/media/FastDataMama/izziv/data`
*   **Dataset:** ImageCAS (1000 3D slik)
*   **Uporabljena razdelitev (Split-1):**
    *   **Train (Učenje):** 750 slik (ID 1–750)
    *   **Inference/Val:** 50 slik (ID 751–800)
    *   **Test (Evalvacija):** 200 slik (ID 801–1000)

---

## 🐳 2. Navodila za zagon (Docker)

Celoten postopek je zapakiran v Docker kontejner. Za dostop do podatkov in grafične kartice so potrebne spodnje nastavitve.

### 1. Priprava slike (Build)
```bash
docker build -t nejcpircmednext .


| **`Izris_Loss_Log.py`** | Iz log datoteke izlušči podatke in izriše graf poteka učenja (Loss curve). |
| **`Primerjava_Dice.py`** | Izriše graf primerjave našega rezultata z nnU-Net benchmarkom. |
| **`mednext_lib/`** | Mapa, ki vsebuje definicijo arhitekture MedNeXt in gradnike (Blocks). |

---

## 🧠 4. O metodi (MedNeXt)

Za rešitev izziva je uporabljena arhitektura **MedNeXt Small**, ki je specializirana za segmentacijo tankih in povezanih struktur.

**Specifike naše implementacije:**
*   **Kernel Size:** 5x5x5 (zajame širši kontekst za ohranjanje kontinuitete žil).
*   **Arhitektura:** ConvNeXt bloki z "Inverted Bottleneck" zasnovo.
*   **Deep Supervision:** Učenje na 5 nivojih globine za boljše zaznavanje detajlov in hitrejšo konvergenco.

**Primerjava z nnU-Net (Baseline):**
Rezultati so primerjani z uradnim rezultatom nnU-Net na ImageCAS datasetu (Dice ~0.885).
Zaradi časovnih omejitev nismo trenirali nnU-Neta, so pa vključena navodila za njegovo reprodukcijo:
1. `nnUNetv2_plan_and_preprocess -d 001`
2. `nnUNetv2_train 001 3d_fullres 0`
