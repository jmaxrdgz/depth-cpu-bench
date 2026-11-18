# Pretrained relative-depth accuracy

Independent of the root CPU timing benchmark. Reuses pinned ZipDepth dataset
loaders, dense evaluator and metrics unchanged. Adds DA-2K ordinal scoring.
Released weights, FP32, no training, pruning, quantization, flips or TTA.

[Comparison table](RESULTS.md).

## Sequential CPU workflow

CPU is the default backend, as requested. The existing CPU speed benchmark is
unchanged. Run the complete download/evaluate/delete workflow from the repo root:

```bash
.venv/bin/python -m pip install -r accuracy/requirements.txt
.venv/bin/python -m accuracy.sequential --threads 4
```

This downloads DA-2K, evaluates all three models, verifies saved per-image CSVs
and input manifests, then deletes only its own temporary dataset directory. It
repeats for DIODE validation, then each of ETH3D's 13 scenes. ETH3D's final means
are computed from all 454 per-image scores, not unweighted scene means. The
retained scene results and manifests provide provenance after raw data is removed.
It also downloads/extracts the RT small monocular release checkpoint using gdown
when missing, and preserves all checkpoints outside temporary dataset storage.

The workflow requires at least 16 GiB free before each batch (currently 27 GiB).
This is a conservative working allowance, not a measured extraction peak. Whole
ETH3D would require about 44.3 GB for depth alone, but only one scene is retained
at a time. Pre-existing user dataset directories are never deleted. Failures
retain downloaded files and completed scores for diagnosis; successful batches
are skipped on restart after verification. A failed model's partial output
without result.json must be moved aside before retrying that model.

The command environment cannot resolve download hosts, but Chrome can access
them. This run therefore obtains official archives through Chrome over HTTPS
and executes the same CPU evaluator against the extracted local files. The
automatic sequential downloader requires working DNS in its shell environment.
Browser-supplied archives can be evaluated with the individual commands below;
ETH3D scene results are combined using `python -m accuracy.merge_eth3d`.

For browser downloads, `python -m accuracy.browser_queue --threads 4` waits for
each scene's two final `*_dslr_undistorted.7z` / `*_dslr_depth.7z` filenames in
`accuracy/data/`. Download there, or move only those completed archives from
Downloads into that directory. The queue extracts, evaluates all three models,
verifies matching results, and deletes the scene and archives before waiting for
the next one. It resumes verified scenes and preserves partial outputs from an
interrupted model before retrying. Run `python -m accuracy.report` after completion
to regenerate the final GitHub-formatted tables.

MPS remains optional via `--device mps` for individual runs; explicit MPS runs
still refuse CPU fallback. CPU runs do not require MPS. The table requires the
same backend throughout all nine full evaluations. No timing is recorded.

## Setup

From the repository root, use the existing environment:

```bash
python -m pip install -r requirements.txt -r accuracy/requirements.txt
python setup_upstreams.py
python -c 'import torch; print(torch.__version__)'
```

Every run checks the three source clones against `upstreams.json`, loads all
weights strictly, and records checkpoint, harness and dataset SHA-256 hashes.

| Model | Official weights | Training background | Input |
| --- | --- | --- | --- |
| GuideDepth-S | [NYU Half](https://drive.google.com/file/d/1ZA80WcgKJsOWaOeBuSn3oupzKHV4eonv/view) | Supervised NYUv2 | 240×320 |
| ZipDepth | `upstream/ZipDepth/checkpoints/zipdepth_base.pth` | Broad teacher distillation | Shorter side 384, aspect retained, rounded to 32 |
| RT-MonoDepth-S | [Archive](https://drive.google.com/file/d/1Jf5K3m0DfAqVcVCE6y0cKufEKIHu86sz/view), `RTMonoDepth/s/m_640_192/` | Monocular self-supervised KITTI | Encoder metadata, expected 192×640 |

GuideDepth-S is currently at `/tmp/NYU_Half_GuideDepth-S.pth`; copy it to durable
storage when space permits. RT requires both `encoder.pth` and `depth.pth` from
the small monocular `m_640_192` variant. ZipDepth uses the convex base checkpoint
and official inference fusion after strict loading. No random initialized keys
are accepted. RT retains all decoder heads to load the full checkpoint strictly.

Checkpoint-appropriate preprocessing and input shapes compare released systems,
not architectures with equal training/pixel budgets. GuideDepth uses RGB [0,1]
with tensor bilinear antialiased resize, RT uses RGB [0,1] and PIL Lanczos, and
ZipDepth uses its official OpenCV bilinear resize. RT/Zip normalization is inside
their models. GuideDepth's raw output is inverse depth (`maxDepth/depth` training
target); RT sigmoid disparity is affine-converted as in `disp_to_depth(0.1,100)`;
ZipDepth emits inverse depth. Larger means closer for every adapter.

These accuracy runs use different input policies from the shared 192×640 CPU
timing benchmark. Do not combine the two tables as measured accuracy/FPS pairs.

## Datasets

Download evaluation data only and arrange as follows:

```text
DATA/
  DA-2K/
    annotations.json
    ... images at paths written in annotations.json ...
  DIODE/val/
    indoors/scene_*/scan_*/*.png
    outdoor/scene_*/scan_*/*.png
    ... matching *_depth.npy and *_depth_mask.npy ...
  ETH3D/
    images/SCENE/images/dslr_images_undistorted/*.JPG
    depth/SCENE/ground_truth_depth/dslr_images/*
```

- [DA-2K](https://huggingface.co/datasets/depth-anything/DA-2K/tree/main): extract
  `DA-2K.zip` (~681 MB compressed). Use the directory containing annotations.json.
  The released annotations contain 1,033 images and 2,068 pairs across all eight scenarios (the dataset card rounds these counts).
- [DIODE](https://diode-dataset.org/): use depth validation,
  `https://diode-dataset.s3.amazonaws.com/val.tar.gz` (~2.6 GB compressed),
  published MD5 `5c895d09201b88973c8fe4552a67dd85`. Keep both domains:
  325 indoors + 446 outdoor = 771 images.
- [ETH3D](https://www.eth3d.net/datasets): use all 13 high-resolution public
  **training** scenes, undistorted DSLR JPGs plus each `*_dslr_depth.7z`, for 454
  pairs. These are evaluation-only here; hidden test depth is unavailable.
  Use per-scene DSLR depth archives, not the site's “all ground truth (for
  evaluation)” scan archive. The ZipDepth loader requires full float32 depth
  rasters, not scan point clouds or low-resolution rig data.

Input file hashes and DA-2K annotations define a common dataset fingerprint.
Malformed/missing GT, non-finite predictions or silently skipped images reject
the run. The final table also checks full expected dataset sizes.

## Run

Use a fresh process per model to avoid upstream module-name collisions.
Example paths below must be replaced by actual storage locations:

```bash
DATA=/Volumes/DepthData/datasets
GUIDE=/Volumes/DepthData/checkpoints/NYU_Half_GuideDepth-S.pth
RT=/Volumes/DepthData/checkpoints/RTMonoDepth/s/m_640_192
for dataset in da2k diode eth3d; do
  python -m accuracy.benchmark --model guidedepth-s --dataset "$dataset" --checkpoint "$GUIDE" --data "$DATA"
  python -m accuracy.benchmark --model zipdepth --dataset "$dataset" --checkpoint upstream/ZipDepth/checkpoints/zipdepth_base.pth --data "$DATA"
  python -m accuracy.benchmark --model rt-monodepth-s --dataset "$dataset" --checkpoint "$RT" --data "$DATA"
done
python -m accuracy.report
```

First test each model on DA-2K using `--limit 2 --output accuracy/smoke`.
Limited runs are excluded from comparison. Existing output directories are never
overwritten; select another `--output` for reruns. Failed runs may leave partial
directories without a result JSON; use a new output directory for the retry.

Small result JSONs and per-image CSVs go to `accuracy/results/` and may be
committed. No prediction maps are saved. Dataset/checkpoint/smoke directories are
ignored. The report requires nine complete runs on the same backend, identical dataset/protocol
and evaluation/source hashes across models, and consistent checkpoint hashes and
input sizes across datasets. Best values are **bold**, second distinct best
`<ins>underlined</ins>`; exact ties share rank.

## Protocol

- DA-2K: pair-weighted accuracy; integer `[row,column]` coordinates. The annotated
  closer point must have strictly greater inverse depth. Ties count as wrong.
  No alignment, calibration or GT-driven polarity selection.
- DIODE/ETH3D: upstream per-image least-squares scale **and shift** in inverse
  depth, inverse conversion and dataset-range clipping. All eight upstream
  metrics: AbsRel, SqRel, RMSE, RMSElog, log10, δ1/δ2/δ3. Means are image-weighted;
  DIODE also has separate domain means. Full-resolution fit, no crop, native
  validity masks. DIODE range (0.6,80)m; ETH3D (1e-5,60)m.
- Upstream fitting uses valid GT pixels with positive predictions and
  unconstrained least squares (potentially a negative scale). Retained for
  reproducibility. DA-2K independently checks polarity. GT-valid pixels remain
  included in the post-alignment scoring.
- Dense aligned metrics evaluate relative structure, not metric scale accuracy.
  Training histories differ. Unseen-data claims need a teacher/training audit.
  Do not tune checkpoint selection on these datasets.

Validation without dataset downloads:

```bash
python -m unittest discover -s accuracy/tests -v
```
