# Validation — 18 November 2026

The released weights were evaluated on CPU in FP32, with four intra-op threads
and one inter-op thread. No inference timing, training, pruning or quantization
was performed. ZipDepth uses its official inference fusion.

## Execution status

- DA-2K: all three models complete; 1,033 images and 2,068 annotated pairs.
- DIODE validation: all three models complete; 325 indoor + 446 outdoor images.
- ETH3D: all three models complete; all 13 public high-resolution scenes,
  454 images. Scene results were aggregated from per-image scores.

Every completed model uses strictly loaded released checkpoints. Model inputs
and GT files are fingerprinted with SHA-256; counts, ordering, masks, source
revisions and evaluation code signatures are checked across models. Per-image
CSVs and manifests are retained locally and excluded from version control.
A subset cannot enter the final table.

## Protocol checks

Ten tests pass, covering ordinal polarity/coordinates/ties, the released DA-2K
ZIP's UTF-8 filename flag issue, path containment, the official DIODE loader,
known affine inverse-depth recovery, strict RT checkpoint loading and rejection
of missing weights, ranking directions/ties, incompatible-run rejection,
ETH3D coverage and image-weighted scene aggregation, and the cleanup gate.

The local environment lacks an installed tqdm package, so execution and tests
used the existing real cached tqdm 4.70.1 package without downloading a substitute:

```bash
PYTHONPATH=/Users/jean-max/.cache/uv/archive-v0/lrwMN6VLRJAJEBcVQknle .venv/bin/python -m unittest discover -s accuracy/tests -v
```

Normal installations should install `accuracy/requirements.txt`. Recorded run
environment: macOS 15.7.3 ARM64, Python 3.13.3, PyTorch 2.14.1, torchvision
0.29.1, NumPy 2.5.4, OpenCV 5.0.0 and Pillow 13.0.0. Detailed environment JSON
is retained locally under the ignored `accuracy/results/` directory.

## Downloads and storage

The command environment cannot resolve download hosts, but Chrome can. Official
archives were downloaded through Chrome over HTTPS. The DIODE validation archive
matched its published MD5, `5c895d09201b88973c8fe4552a67dd85`. ETH3D extraction
checks archive integrity, and locally retained per-scene receipts record archive hashes.

DA-2K's actual annotation counts differ from the rounded dataset-card description;
all released annotations were retained. A deterministic UTF-8/CP437 filename
fallback resolves malformed ZIP filename flags without omitting images.

Dataset/scene RGB, GT and archives are deleted only after all three completed
results are verified. Checkpoints and result/provenance files are retained locally.
No unrelated user files are deleted. No data or checkpoint binaries are published.

Dense errors use the pinned ZipDepth evaluator's per-image scale-and-shift fit
in inverse depth. They measure relative structure, not recovered metric scale.
Input policies and training histories differ across released systems; see the
[protocol](README.md). The existing synthetic-noise CPU speed benchmark is unchanged.

All nine final result records passed the comparison checks. Downloaded datasets,
scene archives and failed transfer fragments were removed. Approximately 24 GiB
was free after cleanup. Checkpoints remain in ignored local storage.
