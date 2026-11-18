"""Strict pretrained adapters. All outputs are inverse depth (larger = closer)."""
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters import build_model, load_file

MODELS = ('guidedepth-s', 'zipdepth', 'rt-monodepth-s')


class Predictor:
    def __init__(self, name, checkpoint, upstream, device='cpu'):
        self.name, self.device = name, device
        self.files = []
        if name == 'rt-monodepth-s':
            module = load_file('accuracy_rt_small', upstream / 'RT-MonoDepth/networks/RTMonoDepth/RTMonoDepth_s.py')
            self.encoder = module.DepthEncoder()
            enc_path, dec_path = checkpoint / 'encoder.pth', checkpoint / 'depth.pth'
            enc = torch.load(enc_path, map_location='cpu', weights_only=True)
            self.size = (int(enc['height']), int(enc['width']))
            if min(self.size) < 64 or any(v % 32 for v in self.size):
                raise ValueError(f'Unsupported checkpoint input shape: {self.size}')
            # Official checkpoints include non-parameter metadata in encoder.pth.
            metadata = {'height', 'width', 'use_stereo'}
            self.encoder.load_state_dict({k: v for k, v in enc.items() if k not in metadata}, strict=True)
            # Retain ALL training heads so strict loading verifies the entire checkpoint.
            self.decoder = module.DepthDecoder(self.encoder.num_ch_enc)
            self.decoder.load_state_dict(torch.load(dec_path, map_location='cpu', weights_only=True), strict=True)
            self.encoder.to(device).eval()
            self.decoder.to(device).eval()
            self.files = [enc_path, dec_path]
        else:
            self.model = build_model(name, upstream)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if name == 'zipdepth':
                from zipdepth.utils.model_utils import strip_state_dict_prefixes
                state = strip_state_dict_prefixes(state.get('model_state_dict', state))
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
            if name == 'zipdepth':
                # Same reparameterization/BN fusion as the official predictor, on CPU first.
                self.model.fuse_for_inference()
                self.size = None  # official shorter-side 384, preserve aspect, round to 32
            else:
                self.size = (240, 320)  # official NYU Half GuideDepth-S checkpoint
            self.model.to(device).eval()
            self.files = [checkpoint]

    @torch.inference_mode()
    def infer_image(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.name == 'zipdepth':
            scale = 384 / min(h, w)
            nh, nw = [max(32, round(v * scale / 32) * 32) for v in (h, w)]
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        elif self.name == 'rt-monodepth-s':
            rgb = np.asarray(Image.fromarray(rgb).resize(self.size[::-1], Image.Resampling.LANCZOS))
        else:
            # GuideDepth's official evaluation uses torchvision Resize on RGB tensors.
            pass
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255)
        if self.name == 'guidedepth-s':
            from torchvision.transforms.functional import resize
            x = resize(x, list(self.size), antialias=True)
        x = x.to(self.device)
        if self.name == 'rt-monodepth-s':
            pred = self.decoder(self.encoder(x))[('disp', 0)]
            # Official disp_to_depth(0.1,100); affine conversion preserves relative order.
            pred = 0.01 + 9.99 * pred
        else:
            pred = self.model(x)
        while pred.ndim < 4:
            pred = pred.unsqueeze(0)
        if pred.ndim != 4 or pred.shape[:2] != (1, 1):
            raise ValueError(f'Invalid output shape: {pred.shape}')
        pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=self.name == 'zipdepth')
        result = pred[0, 0].float().cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Non-finite prediction; refusing to discard pixels')
        return result
