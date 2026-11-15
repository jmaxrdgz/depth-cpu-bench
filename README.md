# Depth CPU benchmarks

Compare **FastDepth, GuideDepth, GuideDepth-S, RT-MonoDepth, RT-MonoDepth-S,
ZipDepth, ZipDepth-NPU, YOLO26n-depth, and an explicitly labelled LNCA reconstruction**
on pre-generated RGB noise. Batch one, CPU only, FP32, random weights, no datasets
or checkpoint downloads. Measures latency, not depth accuracy.

## CPU results

Local run on **Apple M3**, macOS 15.7.3 ARM64, **13 November 2026**.
PyTorch 2.14.1 eager, FP32, batch one, **192×640 RGB**, one intra-op thread and
one inter-op thread, contiguous memory format. Each configuration received
10 warm-ups and 30 timed forwards using a pre-generated ring of 16 uniform-noise
images. All official upstream revisions matched `upstreams.json`.

| Model | Implementation | Parameters (M) | Mean (ms) | Median (ms) | p95 (ms) | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LNCA reconstruction† | Paper reconstruction† | 0.041 | 6.54 | 6.44 | 7.01 | 152.94 |
| GuideDepth-S | Official, unfused | 5.723 | 32.36 | 31.77 | 36.13 | 30.90 |
| ZipDepth (NPU-friendly path) | Official, unfused | 6.787 | 42.42 | 40.82 | 54.09 | 23.57 |
| ZipDepth (convex) | Official, unfused | 6.788 | 43.71 | 44.46 | 45.65 | 22.88 |
| FastDepth | Official, unfused | 3.961 | 52.84 | 52.65 | 55.03 | 18.92 |
| RT-MonoDepth-S | Official, unfused | 1.234 | 61.25 | 60.94 | 64.32 | 16.33 |
| GuideDepth | Official, unfused | 5.825 | 74.09 | 73.21 | 82.49 | 13.50 |
| RT-MonoDepth | Official, unfused | 2.591 | 148.24 | 146.72 | 166.93 | 6.75 |

**GuideDepth-S was fastest among the official implementations in this run**, at 30.90 FPS.
FPS is calculated from mean latency, not median latency. All models, including
both ZipDepth paths, run on CPU without reparameterization or Conv–BN fusion.

† LNCA is a **local paper reconstruction**, not official author code or a
validated reproduction. Its 40,832 parameters match the paper’s component
counts; unresolved details are listed in [LNCA_NOTES.md](LNCA_NOTES.md). Treat its
timing as the performance of this reconstruction, not verification of the paper.

These are short CPU forward-pass measurements with **random weights**, not depth
accuracy results or end-to-end camera throughput. FastDepth is unpruned;
RT-MonoDepth variants omit auxiliary training prediction heads. This table does
not reproduce TVM/TensorRT or the LNCA paper’s smartphone ONNX benchmarks.

Recorded data: [JSON with samples and provenance](benchmarks/cpu-2026-11-13.json)
and [CSV summary](benchmarks/cpu-2026-11-13.csv).

Reproduce the run:

```bash
.venv/bin/python benchmark.py --threads 1 --warmup 10 --iterations 30 --output results/verified-cpu.json
```

## Setup

Use Python 3.10–3.13 and a compatible PyTorch/torchvision pair. On Linux, install
the CPU PyTorch wheels if desired; this script always places models on CPU even
when a CUDA-capable wheel is installed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python setup_upstreams.py
```

The setup script clones five official repositories into ignored `upstream/`
directories and checks out exact revisions from `upstreams.json`. LNCA is implemented locally
from the paper and needs no external clone. Standard and
small variants share their upstream project. Existing checkouts are verified;
the script refuses to replace changed revisions or tracked local edits.

## Run

```bash
python benchmark.py --threads 1 --output results/threads-1.json
python benchmark.py --threads 4 --output results/threads-4.json
python benchmark.py --threads 4 --channels-last --output results/threads-4-nhwc.json
```

All models use **192×640** by default for a common workload. Override using
`--height` and `--width`; both must be multiples of 32 and at least 64. This is a
shared shape constraint for these implementations, not their individual paper
evaluation protocol. A quick functional check:

```bash
python benchmark.py --height 64 --width 96 --warmup 1 --iterations 3
```

Useful options: `--models guidedepth-s rt-monodepth-s`, `--warmup 100`,
`--iterations 1000`, `--noise-images 16`, `--seed 0`.

## What is measured

- Separate fresh process per model, `eval()` and `torch.inference_mode()`.
- One untimed output shape/finite-value validation, then 30 warm-ups and 200
  measured forwards by default. No CUDA calls or synchronization.
- Uniform RGB noise in `[0,1)` is allocated before timing. A ring of 16 images
  varies the input without including image generation or disk I/O in latency.
- Timing includes the model's own normalization and decoding. It excludes
  imports, initialization, external preprocessing and output validation.
- Intra-op CPU threads are explicit; inter-op threads are fixed to one.
- JSON includes every timing sample, mean, median, p90/p95, min/max, standard
  deviation, FPS (`1000 / mean_ms`), parameter counts, shapes, runtime build,
  platform and source revision/dirty status. A CSV summary is written beside it.
- Failures are recorded and produce a nonzero exit status; successful variants
  still appear in the report. Missing source directories fail before timing.

The default backend is **PyTorch eager**, without compilation, quantization or
BN folding. These results do not reproduce TVM/TensorRT paper benchmarks. Compare
CPU runtimes at identical shapes, precision, thread count and memory format.
Random inputs and weights cannot establish accuracy, calibrated metric scale,
trained-weight numerical behavior, or end-to-end camera throughput.

## Model details

| Name | Implementation |
| --- | --- |
| `fastdepth` | Official `MobileNetSkipAdd`, `pretrained=False`; unpruned model, no NetAdapt/TVM |
| `guidedepth` | Official `GuideDepth`, `pretrained=False` |
| `guidedepth-s` | Official small configuration: both feature lists `[32,8,4]` |
| `rt-monodepth` | Official full encoder + decoder, `scales=[0]`, default skips |
| `rt-monodepth-s` | Official small encoder + decoder, `scales=[0]`, default no skips |
| `zipdepth` | Official base model, balanced global context, convex/unfold upsampling, unfused |
| `zipdepth-npu` | Official base model with learned nearest/bilinear blend, unfused; executed on CPU |
| `yolo26n-depth` | Official Ultralytics depth YAML with explicit nano scale; unfused `DepthModel` plus timed full-resolution bilinear resize |
| `lnca-reimplementation` | Local paper reconstruction, 32 channels, four shared NCA updates, 40,832 parameters; see [assumptions](LNCA_NOTES.md) |

The RT-MonoDepth wrapper hardcodes CUDA. We instantiate its encoder and decoder
directly, retaining their upstream operations. `scales=[0]` omits auxiliary
training prediction heads while retaining the final prediction head. We do not
modify upstream source files. Checkpoints are intentionally not loaded.

ZipDepth uses the same unfused PyTorch eager policy as the existing models;
reparameterization and Conv–BN fusion are not applied. Both upsampling paths run
on CPU: the `-npu` suffix identifies the architecture path, not the execution
device. No additional training/export dependencies are required.

YOLO26n-depth is instantiated from the pinned Ultralytics source YAML with
`scale='n'`, random weights and identity calibration. We call `DepthModel`
directly, without the high-level predictor, fusion, TTA, letterboxing or checkpoint
loading. Its eval head emits H/4×W/4; a bilinear 4× upsample with
`align_corners=True` (matching the official export path) is included in the timed
forward so every model produces H×W. Values already include the upstream head's
clamping, exponentiation and calibration buffers. Inputs remain RGB noise in
`[0,1)`. The pinned clone supplies Ultralytics itself; `requirements.txt` supplies
its model-import dependencies. Automatic package installation and online checks
are disabled for this adapter; settings use the ignored `results/ultralytics/`
directory by default.

LNCA is **not official author code**. Its parameter total and component counts
match the paper, but unresolved implementation details are documented in
[LNCA_NOTES.md](LNCA_NOTES.md). JSON records its implementation SHA-256, local
Git revision, and `paper_equivalence_verified=false`; `source.verified` for this
model means only that its source file matches the local committed file.
Its timings cannot validate the paper’s reported accuracy or deployment speed.

## Private GitHub repository

Private repository: https://github.com/jmaxrdgz/depth-cpu-bench

The five upstream repositories are cloned locally at their pinned revisions.
For another machine:

```bash
gh repo clone jmaxrdgz/depth-cpu-bench
cd depth-cpu-bench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python setup_upstreams.py
python benchmark.py --threads 1
```

Upstream dependencies are locked external clones rather than vendored source;
the parent repository records their URLs and commits. Respect upstream licenses,
including RT-MonoDepth's non-commercial restrictions and Ultralytics' AGPL-3.0 license.

## Sources

- https://github.com/dwofk/fast-depth
- https://github.com/mic-rud/GuidedDecoding
- https://github.com/Ecalpal/RT-MonoDepth
- https://github.com/fabiotosi92/ZipDepth
- https://github.com/ultralytics/ultralytics
- https://docs.ultralytics.com/tasks/depth
- https://arxiv.org/html/2607.08771v1
- https://link.springer.com/article/10.1007/s42979-026-05327-4

## Validation

All nine configurations passed CPU
output shape/finite checks and JSON/CSV export. Architecture contract tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

See `VALIDATION.md` for the environment and run conditions. The local `.venv`
uses Python 3.13 and cached PyTorch packages; the previous incomplete environment
is preserved at `.venv.previous/`. Neither environment is committed.
