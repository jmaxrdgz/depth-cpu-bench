# Validation, 2026-11-12

Local validation used Python 3.13 and cached PyTorch 2.14.1, torchvision 0.29.1,
and NumPy 2.5.4 on macOS ARM64. No dependencies were downloaded or globally
installed. The temporary source tree contained the upstream model files read
through the GitHub connector, not Git clones. Results correctly record
`source.verified=false` for these snapshots.

Passed for **all five variants**:

- 64×96, one CPU thread, contiguous layout, one warm-up, three timed forwards.
- 192×640, one CPU thread, contiguous layout, five warm-ups, twenty forwards.
- 64×96, two CPU threads, channels-last, one warm-up, three forwards.
- CPU model construction without checkpoints or CUDA, full-resolution single
  channel output, finite predictions, JSON/CSV generation, correct sample counts
  and reported intra-op/inter-op thread settings.

Also checked Python compilation, CLI help without importing PyTorch, and rejection
of a 65-pixel input height. Timing runs were functional validation; some overlapped
in wall time and the samples are not a controlled performance comparison. The
local full-resolution report is ignored at `results/local-validation.json`.

## Completed follow-up, 2026-11-13

- Created https://github.com/jmaxrdgz/depth-cpu-bench and confirmed private visibility.
- Verified all three actual upstream Git clones against `upstreams.json`; all were clean.
- Preserved the incomplete Python 3.12 environment at `.venv.previous/` and built a
  working Python 3.13 `.venv` from complete locally cached package contents.
- Ran all five models sequentially at 192×640, batch one, FP32, contiguous layout,
  one intra-op thread and one inter-op thread: ten warm-ups and thirty measured
  forwards each. All outputs were finite and correctly shaped; all source records
  were verified. JSON and CSV are at `results/verified-cpu.*` (ignored).
- Dependency installation from the online registry remains untested. Shell GitHub
  access remains unavailable; repository publishing uses the GitHub connector.

No paper accuracy results, Raspberry Pi measurements, trained checkpoints, TVM,
TensorRT, ONNX Runtime, quantized models, or compiled backends were evaluated.

## ZipDepth and LNCA extension, 2026-11-13

- Cloned official ZipDepth at `91f3fd21e131641f51e8d35736d1958350180e3a`.
- Added base/balanced convex-unfold and NPU-friendly blending configurations,
  both unfused and executed on CPU without loading checkpoints.
- Added an explicitly labelled LNCA paper reconstruction; no official code was
  found. See `LNCA_NOTES.md` for uncertainties. Four tests passed, covering exact
  component parameter budgets, four shared updates, positive depth, independent
  local construction, both ZipDepth upsampling paths, rectangular output shape,
  finite CPU outputs, and unknown-model rejection.
- All eight configurations passed a fresh sequential 192×640, FP32, batch-one,
  one-thread run with 10 warm-ups and 30 samples. Each report source was verified
  against its upstream lock or committed local LNCA file; the latter does not
  establish paper equivalence. JSON/CSV report validation passed.
- Results and provenance are committed in `benchmarks/cpu-2026-11-13.*`.
- The upstream setup check now tolerates untracked import caches while still
  refusing tracked edits or wrong revisions; existing checkouts are never rewritten.
