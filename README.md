# Segmentacija koronarnih arterij - MedNeXt (Izziv AMS 2025/2026)
Študent: Nejc Pirc
Predmet: AMS (Analiza Medicinskih Slik)
Metoda: MedNeXt (Convolutional Neural Network)

## Predstavitev izziva
Bolezni srca in ožilja so vodilni vzrok smrti po vsem svetu. Natančna segmentacija koronarnih
arterij (CAS - Coronary Artery Segmentation) je ključna za odkrivanje stenoze (zoženja),
načrtovanje stentiranja in simulacijo pretoka krvi.

Cilj tega izziva je razviti metodo za avtomatsko segmentacijo koronarnih arterij na 3D medicinskih slikah (CTA). Izziv je težak zaradi tanke, vijugaste strukture žil in prisotnosti šuma ter
artefaktov.

Za reševanje tega problema je uporabljen obsežen javni nabor podatkov ImageCAS (1000 3D slik).

## Metoda: MedNeXt

Za rešitev izziva je uporabljena arhitektura MedNeXt, ki nadgrajuje klasični U-Net s sodobnimi ConvNeXt bloki za učinkovitejšo obdelavo 3D podatkov.

## Ključne prednosti za segmentacijo žil:
-Velika jedra (5x5x5): Zajamejo širši kontekst, kar je nujno za ohranjanje kontinuitete dolgih in tankih žil.

-Deep Supervision: Model se uči na 5 nivojih hkrati, kar izboljša zaznavanje finih detajlov in preprečuje izgubo gradientov.

-Učinkovitost: Z uporabo Inverted Bottleneck in Residual Connections omogoča stabilno učenje kompleksnih značilnosti ob manjši porabi spomina.

## Implementacija in Arhitektura

-Vhodni podatki: 3D izrezi (patches) velikosti 96x96x96. Učenje na celotni sliki ni mogoče zaradi omejitev GPU pomnilnika, zato uporabljamo patch-based training.

-Encoder: Zaporedje MedNeXt blokov zmanjšuje ločljivost in ekstrahira značilnosti na več ravneh.

-Decoder: Rekonstruira segmentacijsko masko z združevanjem značilnosti iz encoderja.

-Trening: Uporabljen je DiceCELoss za reševanje problema neuravnoteženih razredov ter AdamW optimizator z CosineAnnealing urnikom učenja.
<img src="MedNeXt.png" alt="Arhitektura MedNeXt" width="500">
