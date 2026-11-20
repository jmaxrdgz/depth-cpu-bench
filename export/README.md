# Export ZipDepth for ncnn CPU FP16

Run commands from the repository root. Use the pinned `upstream/ZipDepth`
checkout prepared by the main README, and its original
`checkpoints/zipdepth_base.pth` checkpoint (the convex/unfold version).

```bash
.venv/bin/python -m pip install -r export/requirements.txt
.venv/bin/python export/zipdepth_ncnn.py
```

The default output is `export/models/zipdepth/`:

- `model.ncnn.param`: the ncnn graph.
- `model.ncnn.bin`: the weights.
- `model.json`: input dimensions and deployment/checkpoint information.

Copy all three files together. The output directory must not already exist;
use `--output` to create another bundle. Generated bundles are ignored by Git.

The default fixed input is **384 high × 384 wide**. For another aspect ratio,
export a new static graph, for example:

```bash
.venv/bin/python export/zipdepth_ncnn.py --height 384 --width 512 \
  --output export/models/zipdepth-384x512
```

Dimensions must be multiples of 32 and at least 64. A larger input changes
inference cost and should be timed separately on the target device.

The exporter strictly loads the checkpoint, calls `fuse_for_inference()`,
traces the original forward, and converts it with pnnx. It preserves learned
global attention and the original convex upsampling. It does **not** use the
upstream ONNX exporter's attention-to-average-pooling replacement.

The serialized weights remain FP32 (`pnnx fp16=0`), matching the successful
benchmark configuration. **FP16 packing, storage, and arithmetic are enabled
by the inference runtime**. This also permits a faithful FP32 parity check
from the same bundle. A half-sized weight file is not required for FP16 inference.

Before publishing the bundle, the exporter checks ncnn FP32 against PyTorch
and checks FP16 prediction drift on noise and a gradient image. Conversion
errors or relative RMS drift above 5% abort the export. These are numerical
smoke checks, not an evaluation of depth accuracy on a dataset. On hosts
without native FP16 kernels, ncnn can fall back to supported precision;
validate deployment on the target Pi as well.

Inference installation and camera/video commands are in
[`../inference/README.md`](../inference/README.md).
