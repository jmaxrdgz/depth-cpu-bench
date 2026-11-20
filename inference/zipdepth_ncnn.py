"""CPU ZipDepth inference without PyTorch or the upstream source checkout."""
import json
from pathlib import Path
import time

import cv2
import ncnn
import numpy as np


def prepare_frame(bgr, height, width, resize='letterbox'):
    """Return RGB CHW in [0,1] and the unpadded model-space rectangle."""
    if bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) == 0:
        raise ValueError('Expected a nonempty BGR image with three channels')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if resize == 'letterbox':
        scale = min(height / rgb.shape[0], width / rgb.shape[1])
        h = max(1, min(height, round(rgb.shape[0] * scale)))
        w = max(1, min(width, round(rgb.shape[1] * scale)))
        top, left = (height - h) // 2, (width - w) // 2
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.copyMakeBorder(rgb, top, height - h - top, left, width - w - left,
                                 cv2.BORDER_REPLICATE)
    elif resize == 'stretch':
        top, left, h, w = 0, 0, height, width
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f'Unknown resize mode: {resize}')
    return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255, (top, left, h, w)


class ZipDepthNCNN:
    def __init__(self, model_dir, threads=2, fp16=True):
        if threads < 1:
            raise ValueError('threads must be positive')
        folder = Path(model_dir)
        self.config = json.loads((folder / 'model.json').read_text())
        if self.config.get('format') != 'zipdepth-ncnn-v1':
            raise ValueError('Unsupported model bundle; use export/zipdepth_ncnn.py')
        self.height, self.width = self.config['height'], self.config['width']
        if any(type(v) is not int or v < 64 or v % 32 for v in (self.height, self.width)):
            raise ValueError('Invalid input dimensions in model.json')
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False
        self.net.opt.use_bf16_storage = False
        self.net.opt.use_packing_layout = True
        self.net.opt.use_fp16_packed = fp16
        self.net.opt.use_fp16_storage = fp16
        self.net.opt.use_fp16_arithmetic = fp16
        for filename, loader in [('model.ncnn.param', self.net.load_param),
                                 ('model.ncnn.bin', self.net.load_model)]:
            path = folder / filename
            if not path.is_file() or loader(str(path)) != 0:
                raise RuntimeError(f'Could not load {path}')

    def forward(self, rgb_chw):
        if rgb_chw.shape != (3, self.height, self.width):
            raise ValueError(f'Expected RGB CHW shape {(3, self.height, self.width)}')
        # Keep the NumPy buffer alive until clone completes (ncnn.Mat borrows it).
        array = np.ascontiguousarray(rgb_chw, dtype=np.float32)
        mat = ncnn.Mat(array).clone()
        with self.net.create_extractor() as extractor:
            if extractor.input('in0', mat) != 0:
                raise RuntimeError('ncnn rejected the input tensor')
            start = time.perf_counter()
            code, output = extractor.extract('out0')
            milliseconds = (time.perf_counter() - start) * 1000
            if code != 0:
                raise RuntimeError(f'ncnn inference failed: {code}')
            depth = np.array(output, copy=True)
        if depth.shape != (1, self.height, self.width) or not np.isfinite(depth).all():
            raise RuntimeError(f'Invalid depth output: {depth.shape}')
        return depth[0], milliseconds

    def predict(self, bgr, resize='letterbox'):
        array, (top, left, h, w) = prepare_frame(bgr, self.height, self.width, resize)
        depth, milliseconds = self.forward(array)
        depth = depth[top:top + h, left:left + w]
        return cv2.resize(depth, (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_LINEAR), milliseconds


def colorize(depth):
    """Per-frame display contrast; output values are relative inverse depth."""
    low, high = np.percentile(depth, [2, 98])
    normalized = np.zeros_like(depth) if high <= low else np.clip((depth - low) / (high - low), 0, 1)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
