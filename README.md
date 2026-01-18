# Segmentacija koronarnih arterij - MedNeXt (Izziv AMS 2025)
Študent: Nejc Pirc
Predmet: AMS (Analiza Medicinskih Slik)
Metoda: MedNeXt (Convolutional Neural Network)


## Segmentacija koronarnih arterij z metodo MedNeXt
![alt text](MedNeXt.png)
## Predstavitev izziva
Bolezni srca in ožilja so vodilni vzrok smrti po svetu. Natančna segmentacija koronarnih arterij na 3D CTA (Computed Tomography Angiography) sgit commit -m "Finalna verzija: Vse skripte (Train, Infer, Test) in Dockerfile"likah je ključna za diagnozo.

Gre za tehnično zahteven problem zaradi specifike anatomije:
-Tanke in vijugaste strukture: Žile so ozke in se razvejano širijo skozi volumen.
-Neuravnoteženi razredi (Class Imbalance): Žile zavzemajo izjemno majhen del volumna (< 1 %) v primerjavi z ozadjem, kar otežuje učenje modela.

Za reševanje tega problema je uporabljen obsežen javni nabor podatkov ImageCAS (1000 3D slik).

## Metoda: MedNeXt
Za rešitev izziva sem uporabil arhitekturo MedNeXt, ki predstavlja sodobno evolucijo konvolucijskih nevronskih mrež za medicinsko segmentacijo.

Kako deluje?
MedNeXt ohranja preverjeno strukturo kodirnika in dekodirnika (Encoder-Decoder), značilno za U-Net, vendar standardne konvolucijske bloke nadomešča s ConvNeXt bloki. Ti so zasnovani za boljšo skalabilnost in učinkovitost pri obdelavi 3D podatkov.

Zakaj sem izbrali MedNeXt?
V primerjavi s klasičnimi metodami (U-Net) ali novejšimi Transformerji (Swin-UNetr), MedNeXt ponuja specifične prednosti za segmentacijo tankih žil:

-Velika jedra (Large Kernels): Uporabljena so konvolucijska jedra velikosti 5x5x5. To modelu omogoča zajemanje širšega konteksta, kar je ključno za ohranjanje kontinuitete dolgih žil (preprečuje "prekinjene" segmentacije).

-Inverted Bottleneck: Uporaba blokov, ki kanale najprej razširijo (expansion ratio = 2) in nato stisnejo. To omogoča učenje kompleksnejših značilnosti brez prevelike porabe spomina.

-Deep Supervision: Model se uči na 5 različnih nivojih globine hkrati. To izboljša pretok gradientov in prisili model, da se nauči detajlov tudi v globokih plasteh mreže.

-Residual Connections: Dodatne povezave v blokih (ResBlock) omogočajo stabilnejše učenje in boljši prenos informacij.

## Implementacija in Arhitektura

-Vhodni podatki: 3D izrezi (patches) velikosti 96x96x96. Učenje na celotni sliki ni mogoče zaradi omejitev GPU pomnilnika, zato uporabljamo patch-based training.
-Encoder: Zaporedje MedNeXt blokov zmanjšuje ločljivost in ekstrahira značilnosti na več ravneh.
-Decoder: Rekonstruira segmentacijsko masko z združevanjem značilnosti iz encoderja.
-Trening: Uporabljen je DiceCELoss za reševanje problema neuravnoteženih razredov ter AdamW optimizator z CosineAnnealing urnikom učenja.
