# Video inference with ncnn FP16

Run from the repository root. First create the bundle using
[`../export/README.md`](../export/README.md). The inference machine needs only
the `inference/` folder and the model bundle; it does not need PyTorch, pnnx,
the training checkpoint, or the upstream sources.

```bash
python3 -m venv .venv-inference
.venv-inference/bin/python -m pip install -r inference/requirements.txt
```

Use an OpenCV-compatible camera (index 0), with the measured two-core Pi setup:

```bash
.venv-inference/bin/python inference/video.py \
  --model export/models/zipdepth --source 0 --cores 2,3
```

The default is two ncnn threads, CPU only, FP16 storage/arithmetic, packed
layout, and default Winograd/SGEMM settings. `--cores` sets Linux CPU affinity
before loading native libraries; omit it on macOS. Press **q**, **Escape**, or
**Ctrl+C** to stop.

For a video file, optionally save the RGB/depth visualization:

```bash
.venv-inference/bin/python inference/video.py --source input.mp4 \
  --headless --output inference/outputs/depth.mp4 --cores 2,3
```

For a network feed or a short unattended test:

```bash
.venv-inference/bin/python inference/video.py --source 'rtsp://camera/feed' \
  --headless --frames 300 --cores 2,3
```

Stream/camera access depends on your OpenCV backend and camera setup. The
script requests a one-frame capture buffer for live sources where supported;
backends may ignore it, so this does not guarantee latest-frame capture or
bounded RTSP latency. Video files are processed sequentially. For an SSH-only
Pi, use `--headless`; if OpenCV's GUI dependencies are unavailable, install
`opencv-python-headless` **instead of** `opencv-python` in the inference venv.

The preprocessing converts BGR to RGB, scales to [0,1], and uses the bundle's
fixed input dimensions. ImageNet normalization is already inside the model.
By default, letterboxing preserves aspect ratio and replicates edge pixels;
the output padding is removed before resizing predictions to source dimensions.
Use `--resize stretch` for direct resizing to the network dimensions, matching
the square-input preprocessing used for the example-image numerical checks.

The displayed depth is **relative inverse depth: larger values mean closer**.
Colors use per-frame percentile contrast and do not represent meters or fixed
distances across frames. The checkpoint has not been calibrated to your camera.

Terminal output distinguishes model-forward FPS from pipeline FPS. Pipeline
FPS includes frame reads, preprocessing, output resizing, visualization, and
optional recording/display, so it differs from model-only benchmark FPS.
Saved MP4 files use the source's reported FPS (30 if unavailable),
not processing FPS. Existing output files are never overwritten.

Generated recordings under `inference/outputs/` are ignored by Git. You can
bound a run with `--frames`, or select a different exported bundle with `--model`.

Small preprocessing checks:

```bash
.venv-inference/bin/python -m unittest discover -s inference/tests -v
```
