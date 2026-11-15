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

## YOLO26n-depth extension, 2026-11-15

- Cloned Ultralytics at `f6296469e088e228eb6776a28f1060fbc993ef62` (8.4.213).
- Constructed official `DepthModel` from the depth YAML with explicit nano scale;
  no checkpoint loading, automatic package installation, predictor or fusion.
- Included bilinear 4× upsampling with `align_corners=True` inside the timed
  forward to match the official export resize and produce full-resolution depth.
- Installed model-import dependencies from PyPI into the existing `.venv`;
  PyTorch 2.14.1, torchvision 0.29.1 and NumPy 2.5.4 remained unchanged.
- Five architecture tests passed. The new test checks nano scale, depth task,
  unfused CPU construction, native quarter-resolution output, exact resize
  equivalence, final rectangular shape, finite values and positive depth.
- All five upstream locks passed setup verification. All nine configurations
  passed a sequential CPU run at 192×640, FP32, batch one, one intra-op and one
  inter-op thread, 10 warm-ups and 30 timed forwards. Source verification,
  shape, sample-count, thread-count, FPS and CSV/JSON checks passed.
- New results: `benchmarks/cpu-2026-11-15.json` and `.csv`. The earlier run is
  retained. Model predictions still use random weights, so no accuracy comparison
  or metric-scale validation is implied.

## Raspberry Pi 5 two-core comparison, 2026-11-15

- Connected to `jmaxrdgz-rpi` through Tailscale; cloned this private repository
  into `/home/jmaxrdgz/depth-cpu-bench` and all five pinned upstream repositories.
- Used an isolated Python 3.13.5 environment on Debian 13.7 ARM64 with CPU-only
  PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu and NumPy 2.5.4.
- All nine models passed a full-resolution, two-thread preflight. The recorded
  comparison then used two intra-op threads, one inter-op thread, CPU affinity
  2–3, 192×640, FP32, batch one, contiguous memory, seed 0, 16 noise images,
  ten warm-ups and thirty timed forwards per model, sequential fresh processes.
- All nine output checks and source verifications passed. Report checks confirmed
  sample counts, thread settings, shapes, finite timings, FPS calculations,
  CSV/JSON agreement and identical parameter counts/upstream commits to the Mac run.
- One-second telemetry recorded 40.8–53.5°C, CPU 0 at 2.4 GHz, and throttling
  flags 0x0 throughout; governor `ondemand`. Full conditions and installed package
  versions are archived in `benchmarks/rpi5-2026-11-15-conditions.json`.
- Results are in `benchmarks/rpi5-2026-11-15.json` and `.csv`; README contains
  a separate table below the MacBook comparison, with the differing core count explicit.
- After timing completed, all five architecture tests passed on the Pi and
  `pip check` reported no broken requirements.

## 1,000-image comparison on both machines, 2026-11-15

- Reran all nine models on the MacBook and the Raspberry Pi 5. Each configuration
  used 1,000 distinct, pre-generated uniform-noise RGB images, one timed forward
  per image, and ten untimed warm-ups. Other conditions were retained: FP32 eager,
  batch one, 192×640, contiguous layout, seed 0, one inter-op thread, random weights,
  unfused official adapters and the explicitly labelled LNCA reconstruction.
- MacBook: one intra-op thread. Pi: two intra-op threads, affinity cores 2–3.
  The runs overlapped across machines; models on each machine ran sequentially
  in isolated workers. No model implementation or pinned upstream revision changed.
- Validated all 18 successful model/machine configurations and 18,000 finite,
  positive timing samples. Checks covered settings, output shapes, source verification,
  unchanged parameter counts and upstream revisions, means, medians, interpolated
  p95 values, FPS calculations, sample counts and CSV/JSON agreement.
- Archived `benchmarks/cpu-2026-11-15-1000.{json,csv}` and
  `benchmarks/rpi5-2026-11-15-1000.{json,csv}`. Each JSON contains every measured
  sample. Earlier short runs remain unchanged. Both README tables now use the
  1,000-image runs and include commands with `--iterations 1000 --noise-images 1000`.
- Pi telemetry is preserved in `benchmarks/rpi5-2026-11-15-1000-conditions.json`:
  236 samples at approximately ten-second intervals, 14:17:22–14:56:23 UTC,
  41.35–60.05°C, governor `ondemand`, CPU 0 `scaling_cur_freq` at 2.4 GHz.
  The first twenty samples reported flags 0x0; the remaining 216 reported
  0x50000, first observed at 14:20:42 UTC. These history bits establish that
  undervoltage and throttling occurred during the run; the sample interval
  does not resolve event durations or their impact on individual timings.
- Separate live SSH spot checks observed active flags `0x50005` during
  RT-MonoDepth (worker elapsed 493 seconds, 58.7°C) and YOLO26n-depth
  (worker elapsed 197 seconds, 59.3°C). These observations fell between the
  archived ten-second samples. The README explicitly discloses power instability;
  no timing samples were filtered out. Flag meanings were checked against
  https://www.raspberrypi.com/documentation/computers/os.html#get_throttled.
- Pi memory spot checks showed no swap use. The MacBook had sufficient memory
  for the image pool and was kept awake with `caffeinate -i` during its run.
  No new architecture tests were needed because implementation code was unchanged;
  verification focused on the recorded measurements and table values.
