# Segmentacija koronarnih arterij - MedNeXt
Izziv AMS 2025
Študent: Nejc Pirc
Rešitev za avtomatsko segmentacijo koronarnih arterij na 3D CTA slikah z uporabo arhitekture MedNeXt.

## Podatki

-Lokacija: /media/FastDataMama/izziv/data
-Struktura: Dataset ImageCAS (1000 slik).
-Razdelitev (Split-1):
-Train: 750 slik.
-Inference/Val: 50 slik.
-Test: 200 slik.

## Zagon celotnega postopka

docker run --gpus all --ipc=host --rm \
  -v .:/workspace/ \
  -v /media/FastDataMama:/media/FastDataMama \
  nejcpircmednext python3 run_all.py


## Opis datotek in skript

Datoteka	                 Opis
Priprava_Podatkov.py	     Pripravi strukturo map (imagesTr, labelsTr, imagesTs) in simbolične povezave do originalnih podatkov (Split-1).
run_train.py	             Izvaja učenje modela. Uporablja Patch-based training (96x96x96), Deep Supervision in augmentacije (rotacije, šum). Shrani model_best.pth.
run_inference.py	         Naloži naučen model in izvede segmentacijo na testnih slikah (200 kom) z uporabo metode Sliding Window Inference.
run_test.py	               Primerja napovedi z referenčnimi maskami. Izračuna Dice Score in clDice (topologija) ter shrani rezultate v metrics.json.
run_all.py	               Krovna skripta, ki požene zgornje štiri skripte v pravilnem vrstnem redu.
Vizualizacija.py	         Generira sliko slika_primerjava_full.png za vizualno primerjavo (Original vs. GT vs. Napoved).
Dockerfile	               Konfiguracija okolja (PyTorch, MONAI, sistemske knjižnice).

## O metodi (MedNeXt)
Uporabljena je arhitektura MedNeXt Small, ki je specializirana za segmentacijo tankih struktur.

Ključne značilnosti:

Velika jedra (5x5x5): Zajamejo širši kontekst za ohranjanje kontinuitete žil.
ConvNeXt bloki: Uporaba "Inverted Bottleneck" za učinkovitost.
Deep Supervision: Učenje na 5 nivojih globine hkrati.

