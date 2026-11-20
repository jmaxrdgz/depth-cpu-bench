#!/usr/bin/env python3
"""Export the original fused ZipDepth-base graph for CPU ncnn FP16 inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'upstream/ZipDepth/checkpoints/zipdepth_base.pth')
    parser.add_argument('--upstream-root', type=Path, default=ROOT / 'upstream')
    parser.add_argument('--output', type=Path, default=ROOT / 'export/models/zipdepth')
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--width', type=int, default=384)
    args = parser.parse_args()
    if any(v < 64 or v % 32 for v in (args.height, args.width)):
        parser.error('height and width must be multiples of 32 and at least 64')
    if args.output.exists():
        parser.error(f'Output already exists; choose a new directory: {args.output}')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')

    import numpy as np
    import ncnn
    import pnnx
    import torch
    from adapters import build_model
    from inference.zipdepth_ncnn import ZipDepthNCNN

    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = build_model('zipdepth', args.upstream_root.resolve()).float().eval()
    from zipdepth.utils.model_utils import strip_state_dict_prefixes
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(strip_state_dict_prefixes(state.get('model_state_dict', state)), strict=True)
    model.fuse_for_inference()
    x = torch.rand(1, 3, args.height, args.width)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.zipdepth-export-', dir=args.output.parent) as directory:
        work = Path(directory)
        bundle = work / 'bundle'
        bundle.mkdir()
        trace_path = work / 'zipdepth.pt'
        with torch.inference_mode():
            traced = torch.jit.trace(model, x)
            traced.save(str(trace_path))
        # Preserve learned attention and convex upsampling; no ONNX exporter patches.
        # FP32 serialized weights match the measured setup; ncnn packs/runs FP16 at load/runtime.
        subprocess.run([pnnx.EXEC_PATH, str(trace_path),
                        f'inputshape=[1,3,{args.height},{args.width}]', 'fp16=0',
                        f'ncnnparam={bundle / "model.ncnn.param"}',
                        f'ncnnbin={bundle / "model.ncnn.bin"}'], check=True)
        config = dict(format='zipdepth-ncnn-v1', height=args.height, width=args.width,
                      variant='base', global_mode='balanced', upsampling='convex-unfold',
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      serialized_weights='float32', runtime_precision='float16',
                      color='RGB', input_range=[0, 1], normalization='inside model',
                      output='relative inverse depth; larger means closer',
                      pnnx_version=pnnx.__version__, ncnn_version=ncnn.__version__,
                      torch_version=torch.__version__)
        (bundle / 'model.json').write_text(json.dumps(config, indent=2) + '\n')
        # Catch conversion mistakes before publishing a deployable bundle.
        fp32, fp16 = ZipDepthNCNN(bundle, fp16=False), ZipDepthNCNN(bundle)
        errors = []
        gradient = torch.linspace(0, 1, args.width).view(1, 1, 1, -1).expand_as(x)
        with torch.inference_mode():
            for tensor in (x, gradient):
                reference = model(tensor)[0, 0].numpy()
                full, _ = fp32.forward(tensor[0].numpy())
                half, _ = fp16.forward(tensor[0].numpy())
                np.testing.assert_allclose(full, reference, rtol=1e-4, atol=1e-5)
                error = float(np.linalg.norm(half - reference) /
                              max(np.linalg.norm(reference), 1e-12))
                if error > .05:
                    raise RuntimeError(f'FP16 prediction drift exceeds 5%: {error:.2%}')
                errors.append(error)
        del fp32, fp16
        bundle.rename(args.output)
    print(f'Exported {args.output}; FP16 relative RMS smoke-check errors: '
          + ', '.join(f'{error:.3%}' for error in errors))


if __name__ == '__main__':
    main()
