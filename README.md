# Depth CPU benchmarks

Compare **FastDepth, GuideDepth, GuideDepth-S, RT-MonoDepth, RT-MonoDepth-S,
ZipDepth, ZipDepth-NPU, YOLO26n-depth, and an explicitly labelled LNCA reconstruction**
on pre-generated RGB noise. Batch one, CPU only, FP32, random weights, no datasets
or checkpoint downloads. Measures latency, not depth accuracy.

## CPU results

### MacBook (Apple M3, one CPU thread)

Local run on **Apple M3**, macOS 15.7.3 ARM64, **15 November 2026**.
PyTorch 2.14.1 eager, FP32, batch one, **192×640 RGB**, one intra-op thread and
one inter-op thread, contiguous memory format. Each configuration received
**10 warm-ups and 1,000 timed forwards per model**, using **1,000 distinct
pre-generated uniform-noise images**, each used once during timing.
All official upstream revisions matched `upstreams.json`.

| Model | Parameters (M) | Mean (ms) | FPS |
| --- | ---: | ---: | ---: |
| LNCA reconstruction† | 0.041 | 6.93 | 144.36 |
| GuideDepth-S | 5.723 | 29.63 | 33.75 |
| YOLO26n-depth | 5.175 | 33.40 | 29.94 |
| ZipDepth (NPU-friendly path) | 6.787 | 34.16 | 29.27 |
| ZipDepth (convex) | 6.788 | 39.03 | 25.62 |
| FastDepth | 3.961 | 43.69 | 22.89 |
| RT-MonoDepth-S | 1.234 | 61.99 | 16.13 |
| GuideDepth | 5.825 | 76.04 | 13.15 |
| RT-MonoDepth | 2.591 | 143.58 | 6.96 |

**GuideDepth-S was fastest among the official implementations in this run**, at 33.75 FPS.
**YOLO26n-depth reached 29.94 FPS**, including full-resolution bilinear upsampling.
FPS is calculated from mean latency, not median latency. All models run on CPU
without reparameterization or Conv–BN fusion.

† LNCA is a **local paper reconstruction**, not official author code or a
validated reproduction. Its 40,832 parameters match the paper’s component
counts; unresolved details are listed in [LNCA_NOTES.md](LNCA_NOTES.md). Treat its
timing as the performance of this reconstruction, not verification of the paper.

These are CPU forward-pass measurements with **random weights**, not depth
accuracy results or end-to-end camera throughput. FastDepth is unpruned;
RT-MonoDepth variants omit auxiliary training prediction heads. YOLO26n-depth
uses its official nano architecture with identity calibration, without TTA or
letterboxing. This table does not reproduce published ONNX/TensorRT, smartphone
or trained-checkpoint accuracy benchmarks.

Recorded data: [JSON with samples and provenance](benchmarks/cpu-2026-11-15-1000.json)
and [CSV summary](benchmarks/cpu-2026-11-15-1000.csv). Earlier short runs remain
archived: [nine models, 30 forwards](benchmarks/cpu-2026-11-15.json)
and [eight models, 30 forwards](benchmarks/cpu-2026-11-13.json).

Reproduce the run:

```bash
.venv/bin/python benchmark.py --threads 1 --warmup 10 --iterations 1000 --noise-images 1000 --output results/verified-cpu.json
```

### Raspberry Pi 5 (two CPU cores)

Run on **Raspberry Pi 5 Model B Rev 1.1, 8 GB RAM**, four Cortex-A76 cores
(stock maximum **2.4 GHz**), **15 November 2026**. Debian 13.7 (64-bit ARM),
Linux 6.18.50+rpt-rpi-2712, Python 3.13.5, PyTorch **2.14.1+cpu**,
torchvision 0.29.1+cpu and NumPy 2.5.4.

All nine models passed a full-resolution two-thread preflight, so every model
used **two intra-op threads and one inter-op thread**, with Linux CPU affinity
restricted to **cores 2–3** (`taskset -c 2,3`). This is a two-core run; the
MacBook table above uses one thread. FP32 eager, batch one, **192×640 RGB**,
contiguous memory format, random weights, seed 0, **1,000 distinct pre-generated
uniform-noise images**, each used once during timing, **10 warm-ups and 1,000
timed forwards per model**. Models ran sequentially
in fresh processes with the same verified upstream revisions and adapters as
the MacBook run, including YOLO's timed full-resolution resize.

| Model | Parameters (M) | Mean (ms) | FPS |
| --- | ---: | ---: | ---: |
| LNCA reconstruction† | 0.041 | 25.05 | 39.92 |
| GuideDepth-S | 5.723 | 188.84 | 5.30 |
| ZipDepth (NPU-friendly path) | 6.787 | 195.83 | 5.11 |
| RT-MonoDepth-S | 1.234 | 205.42 | 4.87 |
| ZipDepth (convex) | 6.788 | 209.94 | 4.76 |
| YOLO26n-depth | 5.175 | 273.80 | 3.65 |
| FastDepth | 3.961 | 281.49 | 3.55 |
| GuideDepth | 5.825 | 388.94 | 2.57 |
| RT-MonoDepth | 2.591 | 499.86 | 2.00 |

**GuideDepth-S was fastest among the official implementations at 5.30 FPS.**
The LNCA row retains the reconstruction caveat above; these measurements do not
establish depth accuracy. FPS is `1000 / mean_ms`.

**Power instability was detected during this Pi run.** The 10-second telemetry
samples of `vcgencmd get_throttled` changed from `0x0` to `0x50000` at
14:20:42 UTC, recording
undervoltage and throttling events. Additional live spot checks observed
`0x50005` (active undervoltage and throttling) during RT-MonoDepth and YOLO26n-depth;
see [validation notes](VALIDATION.md). These flags follow the
[official Raspberry Pi definitions](https://www.raspberrypi.com/documentation/computers/os.html#get_throttled).
The table includes all 1,000 samples per model, including any effects of these
events; their impact was not isolated. Treat it as the observed performance of
this Pi under these power conditions.

The governor remained `ondemand`. Ten-second telemetry samples recorded
**41.35–60.05°C** and CPU 0's `scaling_cur_freq` reported 2.4 GHz at every sample;
that sampled value does not establish uninterrupted effective clock speed.
A separate monitoring process sampled telemetry during the run; its overhead
remains part of the machine's background load.

Recorded data: [JSON with samples and provenance](benchmarks/rpi5-2026-11-15-1000.json),
[CSV summary](benchmarks/rpi5-2026-11-15-1000.csv), and
[hardware, package versions, command and telemetry](benchmarks/rpi5-2026-11-15-1000-conditions.json).

The previous [30-forward Pi run](benchmarks/rpi5-2026-11-15.json) and its
[conditions](benchmarks/rpi5-2026-11-15-conditions.json) remain archived.

Reproduce on the Pi after setup (CPU-only wheels were installed from the
PyTorch CPU index, with torch 2.14.1, torchvision 0.29.1 and NumPy 2.5.4):

```bash
taskset -c 2,3 .venv/bin/python benchmark.py --threads 2 --warmup 10 --iterations 1000 --noise-images 1000 --output results/rpi5.json
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

All nine configurations passed CPU output shape/finite checks and JSON/CSV export. Architecture contract tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

See `VALIDATION.md` for the environment and run conditions. The local `.venv`
uses Python 3.13 and cached PyTorch packages; the previous incomplete environment
is preserved at `.venv.previous/`. Neither environment is committed.
