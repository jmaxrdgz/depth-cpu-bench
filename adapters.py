"""Official upstream adapters plus an explicitly labelled LNCA reconstruction."""
import importlib.util
import os
from pathlib import Path
import sys

MODELS = ('fastdepth', 'guidedepth', 'guidedepth-s', 'rt-monodepth', 'rt-monodepth-s',
          'zipdepth', 'zipdepth-npu', 'lnca-reimplementation', 'yolo26n-depth')
REPOS = dict(zip(MODELS, ('fast-depth', 'GuidedDecoding', 'GuidedDecoding', 'RT-MonoDepth',
                         'RT-MonoDepth', 'ZipDepth', 'ZipDepth', None, 'ultralytics')))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_model(name, upstream_root, output_size=(192, 640)):
    import torch.nn as nn
    if name == 'lnca-reimplementation':
        from lnca import LNCADepth
        return LNCADepth()
    if name not in MODELS:
        raise ValueError(f'Unknown model: {name}')
    repo = Path(upstream_root) / REPOS[name]
    # Each benchmark variant runs in its own process, avoiding generic upstream
    # package-name collisions (model, models, imagenet, networks).
    sys.path.insert(0, str(repo))
    if name == 'yolo26n-depth':
        os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parent / 'results/ultralytics'))
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ['YOLO_OFFLINE'] = 'true'
        from ultralytics.nn.tasks import DepthModel
        from ultralytics.utils import YAML
        config = YAML.load(repo / 'ultralytics/cfg/models/26/yolo26-depth.yaml')
        config['scale'] = 'n'
        model = DepthModel(cfg=config, ch=3, verbose=False)
        # The native eval head emits H/4 x W/4. Match its official export
        # resize to restore full input resolution inside the timed forward.
        return nn.Sequential(model, nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
    if name.startswith('zipdepth'):
        from zipdepth.model.architecture import create_model
        return create_model(variant='base', global_mode='balanced', pretrained=False,
                            upsample_unfold=name == 'zipdepth')
    if name == 'fastdepth':
        module = load_file('fastdepth_upstream', repo / 'models.py')
        return module.MobileNetSkipAdd(output_size=output_size, pretrained=False)
    if name.startswith('guidedepth'):
        from model.GuideDepth import GuideDepth
        if name.endswith('-s'):
            return GuideDepth(pretrained=False, up_features=[32, 8, 4], inner_features=[32, 8, 4])
        return GuideDepth(pretrained=False)

    filename = 'RTMonoDepth_s.py' if name.endswith('-s') else 'RTMonoDepth.py'
    module = load_file('rt_depth_upstream', repo / 'networks' / 'RTMonoDepth' / filename)

    class DepthNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = module.DepthEncoder()
            # Paper inference uses only final-resolution prediction. Passing
            # scales=[0] prevents computation AND construction of auxiliary heads.
            self.decoder = module.DepthDecoder(self.encoder.num_ch_enc, scales=[0])

        def forward(self, x):
            return self.decoder(self.encoder(x))[('disp', 0)]

    return DepthNetwork()
