# Segmentacija koronarnih arterij - MedNeXt
**Izziv AMS 2025**  
**Študent:** Nejc Pirc

Repozitorij vsebuje rešitev za avtomatsko segmentacijo koronarnih arterij na 3D CTA slikah z uporabo arhitekture **MedNeXt**. Rešitev je optimizirana za delovanje v Docker okolju na strežniški infrastrukturi.

---

##  1. Podatki

Zaradi velikosti dataseta se podatki **ne kopirajo lokalno** v kontejner, ampak se berejo neposredno z diska strežnika. Skripte so prilagojene za to strukturo.

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
docker build -t nejcpircmednext .



