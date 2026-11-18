# Pretrained relative-depth accuracy

FP32 CPU; released weights; no training or test-time augmentation.
Best: **bold**. Second distinct best: <ins>underlined</ins>. Ties share rank.

Dense metrics use per-image least-squares scale/shift in inverse depth,
then clipped depth metrics from the pinned ZipDepth evaluator.
These scores do not establish metric scale accuracy.

[Protocol, checkpoints and input shapes](README.md) · [Recorded environment](results/environment.json)

## DA2K

| Model | Pair accuracy ↑ |
| --- | ---: |
| GuideDepth-S | <ins>0.6862</ins> |
| ZipDepth | **0.8670** |
| RT-MonoDepth-S | 0.5919 |

## DIODE

| Model | AbsRel ↓ | SqRel ↓ | Aligned RMSE (m) ↓ | RMSElog ↓ | log10 ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GuideDepth-S | <ins>0.2755</ins> | <ins>2.5777</ins> | <ins>5.2843</ins> | <ins>0.3726</ins> | <ins>0.1210</ins> | <ins>0.6249</ins> | <ins>0.8122</ins> | <ins>0.8938</ins> |
| ZipDepth | **0.2260** | 2.7768 | **4.5376** | **0.2905** | **0.0894** | **0.7391** | **0.8699** | **0.9266** |
| RT-MonoDepth-S | 0.2942 | **2.3839** | 5.4509 | 0.4016 | 0.1319 | 0.5781 | 0.7899 | 0.8819 |

## DIODE indoors

| Model | AbsRel ↓ | SqRel ↓ | Aligned RMSE (m) ↓ | RMSElog ↓ | log10 ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GuideDepth-S | <ins>0.1568</ins> | 0.9462 | <ins>1.0124</ins> | <ins>0.2108</ins> | <ins>0.0661</ins> | <ins>0.8059</ins> | <ins>0.9432</ins> | <ins>0.9768</ins> |
| ZipDepth | **0.0888** | **0.0939** | **0.5990** | **0.1364** | **0.0388** | **0.9135** | **0.9738** | **0.9913** |
| RT-MonoDepth-S | 0.1878 | <ins>0.2754</ins> | 1.0416 | 0.2594 | 0.0859 | 0.7156 | 0.9061 | 0.9597 |

## DIODE outdoor

| Model | AbsRel ↓ | SqRel ↓ | Aligned RMSE (m) ↓ | RMSElog ↓ | log10 ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GuideDepth-S | <ins>0.3621</ins> | **3.7666** | <ins>8.3973</ins> | <ins>0.4904</ins> | <ins>0.1611</ins> | <ins>0.4930</ins> | <ins>0.7168</ins> | <ins>0.8333</ins> |
| ZipDepth | **0.3261** | 4.7319 | **7.4076** | **0.4027** | **0.1263** | **0.6121** | **0.7942** | **0.8795** |
| RT-MonoDepth-S | 0.3718 | <ins>3.9203</ins> | 8.6640 | 0.5052 | 0.1654 | 0.4779 | 0.7052 | 0.8252 |

## ETH3D

| Model | AbsRel ↓ | SqRel ↓ | Aligned RMSE (m) ↓ | RMSElog ↓ | log10 ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GuideDepth-S | <ins>0.1664</ins> | <ins>0.7095</ins> | <ins>2.0345</ins> | <ins>0.2212</ins> | <ins>0.0717</ins> | <ins>0.7650</ins> | <ins>0.9299</ins> | <ins>0.9724</ins> |
| ZipDepth | **0.0969** | **0.4144** | **1.4010** | **0.1295** | **0.0399** | **0.9236** | **0.9831** | **0.9930** |
| RT-MonoDepth-S | 0.2299 | 0.7749 | 2.4453 | 0.3095 | 0.1025 | 0.6177 | 0.8511 | 0.9408 |

## Recorded runs

- [da2k/guidedepth-s](results/da2k/guidedepth-s/result.json): 1033 images; 2026-11-18T08:32:44.838491+00:00
- [da2k/zipdepth](results/da2k/zipdepth/result.json): 1033 images; 2026-11-18T08:34:15.969362+00:00
- [da2k/rt-monodepth-s](results/da2k/rt-monodepth-s/result.json): 1033 images; 2026-11-18T08:35:37.958867+00:00
- [diode/guidedepth-s](results/diode/guidedepth-s/result.json): 771 images; 2026-11-18T08:40:26.465573+00:00
- [diode/zipdepth](results/diode/zipdepth/result.json): 771 images; 2026-11-18T08:42:45.679452+00:00
- [diode/rt-monodepth-s](results/diode/rt-monodepth-s/result.json): 771 images; 2026-11-18T08:44:18.351795+00:00
- [eth3d/guidedepth-s](results/eth3d/guidedepth-s/result.json): 454 images; 2026-11-18T10:38:20.861033+00:00
- [eth3d/zipdepth](results/eth3d/zipdepth/result.json): 454 images; 2026-11-18T10:41:36.009196+00:00
- [eth3d/rt-monodepth-s](results/eth3d/rt-monodepth-s/result.json): 454 images; 2026-11-18T10:43:03.729732+00:00
