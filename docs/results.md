# Rezultate pe MVTec AD

*Măsurat pe 6 noiembrie 2026.*

## `metal_nut`, EfficientAD small, 256, cu autoencoder, 70.000 de pași

Scorul e AUROC la nivel de imagine: dintr-o piesă bună și una defectă, cât de des primește
cea defectă scorul mai mare.

| Măsură | Valoare |
|---|---:|
| AUROC, acest model, o singură antrenare | **0,9868** |
| Publicat, articolul EfficientAD, medie pe 5 rulări | 0,9949 |
| Publicat, `nelson1425/EfficientAD`, `results/mvtec_ad_small.json` | 0,9951 |

Modelul e sub scorurile publicate cu aproximativ 0,8 puncte procentuale și peste minimul de
0,90 al planului.

| Tip de defect | AUROC |
|---|---:|
| `bent` | 0,9927 |
| `color` | 0,9897 |
| `flip` | 1,0000 |
| `scratch` | 0,9644 |

## La pragul confirmat

Pragul e **0,3334**, calculat doar din cele 22 de imagini de validare: scorul lor cel
mai mare, plus 10%. Confirmat de Tudor pe 7 noiembrie 2026.

| Imagini de test | Total | Marcate ca defecte |
|---|---:|---:|
| bune | 22 | 1, adică 4,5% |
| defecte | 93 | 91, adică 97,8% |

- Imaginea bună marcată greșit e `test/good/016.png`, cu scorul 0,917. Următoarea imagine bună
  are 0,261. Din cele 27 de perechi bună-defectă ordonate greșit, din 2046, 21 vin de la ea. Fără
  ea, AUROC ar fi 0,9969. Calculul arată cât cântărește o imagine, iar scorul raportat rămâne 0,9868.
- Defectele ratate sînt `test/scratch/000.png`, cu 0,299, și `test/scratch/020.png`, cu 0,167.

## Cum s-a verificat

Scorurile au fost recalculate pe laptop, din `model.onnx` descărcat, cu `onnxruntime` pe
procesor, nu cu codul de antrenare. AUROC, ratele și scorul pe fiecare tip de defect ies
identice cu cele din Colab. Pragul diferă cu 0,0000004.

## Limite

- **Datele:** site-ul MVTec nu răspundea pe 6 noiembrie 2026, așa că setul vine dintr-o copie
  neoficială: Hugging Face `MSherbinii/mvtec-ad-metal-nut`, revizia
  `8db7371c35a94709cbb3e69ac69f80d2716e2f8a`. Copia nu a fost comparată încă cu originalul.
- **Comparația:** e o singură antrenare, față de o medie pe 5 rulări, cu 22 de imagini bune
  în test. Cauza diferenței nu e stabilită.
- **Pregătirea imaginii:** aici se folosește OpenCV `INTER_AREA`, iar scorul e luat direct din
  harta de 56 x 56. Sursa folosește redimensionarea din PIL și ia scorul din harta mărită la
  rezoluția imaginii. Efectul acestor diferențe nu a fost măsurat.
- **Ce nu arată:** e un rezultat pe un set de referință, cu imagini de laborator, nu pe
  obiectele și camera stației. Timpul și memoria pe Nano sînt în `proba-viteza.md`.

## Surse

- [EfficientAD, articolul](https://arxiv.org/abs/2303.14535), fișierul
  [`per_scenario_results.json`](https://www.mydrive.ch/shares/79401/c41dcdb937972fb43d5cdd7bfa7072f8/download/449203483-1690527455/per_scenario_results.json),
  `efficientad_s/mvtec_ad/detection/auc_roc/metal_nut`.
- [`nelson1425/EfficientAD`](https://github.com/nelson1425/EfficientAD),
  `results/mvtec_ad_small.json`, `metal_nut/classification_au_roc`.

---

# Rezultate pe piulițe M10, cu stația

*Pozele: 7 noiembrie 2026, 12:06–12:40. Antrenarea: 8 noiembrie 2026. Pragul confirmat de Tudor pe
15 noiembrie 2026.*

Scorul de aici și cel de pe MVTec AD se raportează separat, după spec, §5.

## Setul de date

Webcam-ul laptopului, la 1920 x 1080, cu imaginea tăiată la zona de interes `[603, 399, 505, 450]`.
Setul e împărțit pe piese, nu pe poze: toate pozele unei piulițe stau de aceeași parte.

| Parte | Piese | Imagini |
|---|---:|---:|
| antrenare, bune | 27 | 135 |
| validare, bune | 3 | 15 |
| test, bune | 10 | 50 |
| test, defecte: zgârietură, marker, lovitură | 5 + 5 + 5 | 45 |

## `piulita-m10`, EfficientAD small, 256, cu autoencoder, 70.000 de pași

| Măsură | Valoare |
|---|---:|
| AUROC, toate defectele | **0,905** |
| marker | 0,997 |
| zgârietură | 0,976 |
| lovitură | 0,741 |

## La pragul confirmat

Pragul e **0,1834**, calculat doar din cele 15 imagini de validare.

| Imagini de test | Total | Marcate ca defecte |
|---|---:|---:|
| bune | 50 | 2, adică 4%, ambele de la piesa 28 |
| marker | 15 | 15 |
| zgârietură | 15 | 13 |
| lovitură | 15 | 2 |
| **defecte vizibile, marker și zgârietură** | 30 | **28, adică 93%** |
| toate defectele | 45 | 30, adică 67% |

Ținta din spec, §5, adică cel puțin 9 din 10 defecte vizibile prinse cu cel mult 1 din 10 piese
bune acuzate, e atinsă pentru marker și zgârietură. Loviturile din set au fost făcute fără ciocan
și se văd slab chiar și cu ochiul liber. Un prag mai jos nu le salvează: la 0,15 se prind 2 în plus
și sînt acuzate 7 piese bune din 50.

## Lovituri cu ciocanul, în altă lumină

Pe 15 noiembrie 2026, la 14:28–14:32, s-au pozat 5 piulițe lovite cu ciocanul și, ca martor, 5
piulițe bune, câte 3 poze, în aceeași lumină de după-amiază. Fundalul zonei a avut 81–84, față de
87–93 în pozele de antrenare. Pozele nu intră în setul de mai sus.

| Grupă | Scoruri |
|---|---|
| martor, bune | 0,21 – 0,38 |
| lovituri cu ciocanul | 0,26 – 0,43 |

- AUROC lovitură față de martor: 0,83 pe imagini și 0,84 pe piese.
- Toate cele 30 de imagini ies peste prag, inclusiv piesele bune. Pragul nu se poate judeca din
  aceste poze.
- O piuliță bună, pe centru, a ieșit 0,35–0,40 chiar și cu fundalul adus la 88–90. Nici culoarea
  hârtiei nu e cauza: corectată spre cea de la prânz, scorul urcă la 0,73.
- Între poze piesele n-au fost mutate, deci fiecare piesă are de fapt o singură imagine.

## Limite

- **Lumina:** stația depinde de lumina zilei. Modelul a învățat piese pozate la prânz, iar
  după-amiaza sau seara chiar și piesele bune ies peste prag. Pe 9 noiembrie, seara, o piuliță
  bună a ieșit 1,13.
- **Loviturile:** cele cu ciocanul se deosebesc de piesele bune mai bine decât cele fără ciocan,
  cu AUROC 0,83 față de 0,74, dar nu au fost testate în lumina în care a învățat modelul.
- **Mărimea:** în test sînt 10 piese bune și 5 piese pe fiecare defect. O singură piesă schimbă
  rata alarmelor false cu 10 puncte procentuale.

## Cum s-a verificat

Scorurile au fost recalculate pe laptop din `model.onnx`, cu `onnxruntime` pe procesor, și ies
identice cu cele din Colab.
