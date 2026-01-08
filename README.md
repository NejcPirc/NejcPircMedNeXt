# AMS 2025 Izziv: MedNeXt za segmentacijo koronarnih arterij (ImageCAS)

Implementacija MedNeXt (MIC-DKFZ/MedNeXt) za avtomatsko segmentacijo na CTA slikah. Uporabljen Split-1 za razvoj, primerjava z nnU-Net.

## Instalacija
(kasneje dodamo)

## Uporaba
- run_train.py: Trening
- run_test.py: Testiranje
- run_inference.py: Inferenca

## Dataset
ImageCAS (pretvorjen v nnU-Net format).

## Docker
Uporabi Dockerfile za okolje.

## Primerjava z nnU-Net (Baseline)

Za primerjavo smo uporabili uradni nnU-Net framework. Zaradi dolgotrajnosti postopka učenja na celotnem ImageCAS datasetu (več dni), v poročilu primerjamo naše rezultate z uradno objavljenimi rezultati za nnU-Net na tem datasetu (Dice ~ 0.885).

### Reprodukcija nnU-Net rezultatov

Če želite zagnati nnU-Net na teh podatkih, sledite spodnjim korakom:

1. **Namestitev nnU-Net:**
   ```bash
   pip install nnunetv2