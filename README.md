# Depth CPU benchmarks

Compare **FastDepth, GuideDepth, GuideDepth-S, RT-MonoDepth, RT-MonoDepth-S**
on pre-generated RGB noise. Batch one, CPU only, FP32, random weights, no datasets
or checkpoint downloads. Measures latency, not depth accuracy.

## CPU results

Local run on **Apple M3**, macOS 15.7.3 ARM64, **13 November 2026**
(23:03 UTC on 12 November). PyTorch 2.14.1 eager, FP32, batch one, **192×640 RGB**,
one intra-op thread and one inter-op thread, contiguous memory format.
Each model received 10 warm-ups and 30 timed forwards using a pre-generated ring
of 16 uniform-noise images. All upstream revisions matched `upstreams.json`.

| Model | Parameters (M) | Mean (ms) | Median (ms) | p95 (ms) | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| GuideDepth-S | 5.723 | 31.93 | 31.26 | 34.31 | 31.32 |
| FastDepth | 3.961 | 52.67 | 52.95 | 60.07 | 18.99 |
| RT-MonoDepth-S | 1.234 | 64.09 | 63.82 | 66.87 | 15.60 |
| GuideDepth | 5.825 | 77.63 | 75.27 | 89.78 | 12.88 |
| RT-MonoDepth | 2.591 | 151.59 | 149.09 | 171.97 | 6.60 |

**GuideDepth-S was fastest in this run**, at 31.32 FPS, about 1.65× FastDepth’s
throughput. FPS is calculated from mean latency, not median latency.

These are short CPU forward-pass measurements with **random weights**, not depth
accuracy results or end-to-end camera throughput. FastDepth is unpruned;
RT-MonoDepth variants omit auxiliary training prediction heads. The table does
not reproduce the papers’ TVM/TensorRT benchmarks.

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

The setup script clones three official repositories into ignored `upstream/`
directories and checks out exact revisions from `upstreams.json`. Standard and
small variants share their upstream project. Existing checkouts are verified;
the script refuses to replace changed revisions or dirty files.

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

The RT-MonoDepth wrapper hardcodes CUDA. We instantiate its encoder and decoder
directly, retaining their upstream operations. `scales=[0]` omits auxiliary
training prediction heads while retaining the final prediction head. We do not
modify upstream source files. Checkpoints are intentionally not loaded.

## Private GitHub repository

Private repository: https://github.com/jmaxrdgz/depth-cpu-bench

The three upstream repositories are cloned locally at their pinned revisions.
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
including RT-MonoDepth's non-commercial restrictions.

## Sources

- https://github.com/dwofk/fast-depth
- https://github.com/mic-rud/GuidedDecoding
- https://github.com/Ecalpal/RT-MonoDepth

## Validation

All five variants passed CPU output shape/finite checks and JSON/CSV export.
See `VALIDATION.md` for the environment and run conditions. The local `.venv`
uses Python 3.13 and cached PyTorch packages; the previous incomplete environment
is preserved at `.venv.previous/`. Neither environment is committed.
