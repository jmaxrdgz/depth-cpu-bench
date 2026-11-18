#!/usr/bin/env python3
"""Relative-depth accuracy only. Run: python -m accuracy.benchmark --help."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('guidedepth-s', 'zipdepth', 'rt-monodepth-s')
DATASETS = ('da2k', 'diode', 'eth3d')
REPOS = ('GuidedDecoding', 'ZipDepth', 'RT-MonoDepth')
ETH3D_SCENES = dict(courtyard=38, delivery_area=44, electro=45, facade=76,
                   kicker=31, meadow=15, office=26, pipes=14, playground=38,
                   relief=31, relief_2=31, terrace=23, terrains=42)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_upstreams(upstream):
    specs = json.loads((ROOT / 'upstreams.json').read_text())
    for repo in REPOS:
        path = upstream / repo
        if not (path / '.git').exists():
            raise ValueError(f'Missing pinned clone: {path}; run setup_upstreams.py')
        revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if revision != specs[repo]['revision'] or dirty:
            raise ValueError(f'Upstream {repo} differs from upstreams.json or has tracked edits')
    return {repo: specs[repo] for repo in REPOS}


def da2k_samples(root):
    annotations = json.loads((root / 'annotations.json').read_text())
    samples = []
    for name, pairs in sorted(annotations.items()):
        image = (root / name).resolve()
        if not image.is_relative_to(root.resolve()):
            raise ValueError(f'Image escapes DA-2K directory: {name}')
        # Released ZIP has a few UTF-8 filenames without the UTF-8 flag. Standard
        # ZIP extraction decodes those bytes as CP437, unlike annotations.json.
        if not image.is_file():
            repaired = root / name.encode('utf-8').decode('cp437')
            if repaired.is_file():
                image = repaired.resolve()
        if not pairs:
            raise ValueError(f'No annotations for {name}')
        samples.append(dict(name=name, image_path=str(image), pairs=pairs))
    return samples


def ordinal_metrics(pred, pairs):
    """DA-2K [row,column] integer coordinates; ties are incorrect."""
    import numpy as np
    if pred.ndim != 2 or not np.isfinite(pred).all():
        raise ValueError('Invalid ordinal prediction')
    correct = 0
    for pair in pairs:
        values = {}
        for key in ('point1', 'point2'):
            point = pair[key]
            if len(point) != 2 or any(int(v) != v for v in point):
                raise ValueError(f'Invalid DA-2K point: {point}')
            row, col = map(int, point)
            if not (0 <= row < pred.shape[0] and 0 <= col < pred.shape[1]):
                raise ValueError(f'DA-2K coordinate outside image: {point}')
            values[key] = pred[row, col]
        closer = pair['closer_point']
        if closer not in values:
            raise ValueError(f'Unknown closer_point: {closer}')
        other = 'point2' if closer == 'point1' else 'point1'
        correct += int(values[closer] > values[other])
    return correct, len(pairs)


def validate_dense(samples, cfg, load_gt):
    """Fail before inference instead of upstream's permissive skip-on-GT-error."""
    import numpy as np
    for sample in samples:
        for key in ('image_path', 'depth_path', 'mask_path'):
            if key in sample and not Path(sample[key]).is_file():
                raise FileNotFoundError(sample[key])
        gt, mask = load_gt(sample, cfg)
        if gt.ndim != 2 or mask.shape != gt.shape or mask.sum() < 10:
            raise ValueError(f'Insufficient valid GT: {sample["name"]}')
        if not np.isfinite(gt[mask]).all():
            raise ValueError(f'Non-finite GT: {sample["name"]}')


def evaluate_ordinal(predictor, samples, output):
    import cv2
    total, correct, rows = 0, 0, []
    for i, sample in enumerate(samples):
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise ValueError(f'Cannot read {sample["image_path"]}')
        ncorrect, n = ordinal_metrics(predictor.infer_image(image), sample['pairs'])
        correct += ncorrect
        total += n
        rows.append(dict(filename=sample['name'], correct=ncorrect, pairs=n))
        if (i + 1) % 25 == 0:
            print(f'DA-2K: {i + 1}/{len(samples)} images', flush=True)
    with open(output / 'per_sample.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'correct', 'pairs'])
        writer.writeheader()
        writer.writerows(rows)
    return dict(pair_accuracy=correct / total), dict(images=len(samples), pairs=total)


def sample_manifest(samples):
    """Hash every input file and annotation; identical image/GT set across models."""
    manifest = []
    for sample in samples:
        entry = dict(name=sample['name'])
        for key in ('image_path', 'depth_path', 'mask_path'):
            if key in sample:
                entry[key.replace('_path', '_sha256')] = digest(sample[key])
        if 'pairs' in sample:
            entry['pairs'] = sample['pairs']
        manifest.append(entry)
    return manifest


def run(args):
    if args.device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Unset PYTORCH_ENABLE_MPS_FALLBACK; CPU operation fallback is not allowed')
    import torch
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable in this Python session. No CPU fallback performed.')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sources = verify_upstreams(args.upstream)
    sys.path.insert(0, str(args.upstream / 'ZipDepth'))
    from zipdepth.evaluation.datasets import DATASET_CONFIGS, discover_samples, load_gt
    from zipdepth.evaluation.evaluator import evaluate
    from .models import Predictor
    dataset_root = args.data / {'da2k': 'DA-2K', 'diode': 'DIODE/val', 'eth3d': 'ETH3D'}[args.dataset]
    if args.dataset == 'da2k':
        samples = da2k_samples(dataset_root)
        cfg = None
    else:
        samples = discover_samples(args.dataset, str(dataset_root))
        cfg = DATASET_CONFIGS[args.dataset].copy()
    if args.scene:
        samples = [s for s in samples if s['name'].split('/')[0] == args.scene]
        if not args.limit and len(samples) != ETH3D_SCENES[args.scene]:
            raise ValueError(f'Incorrect ETH3D scene size for {args.scene}: {len(samples)}')
    if not samples:
        raise ValueError(f'No samples found at {dataset_root}')
    if len({s['name'] for s in samples}) != len(samples):
        raise ValueError('Duplicate sample names')
    if args.limit:
        samples = samples[:args.limit]
    if cfg:
        validate_dense(samples, cfg, load_gt)
    manifest = sample_manifest(samples)
    manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    torch.manual_seed(0)
    predictor = Predictor(args.model, args.checkpoint, args.upstream, args.device)
    out = args.output / args.dataset / args.model
    if args.scene:
        out = args.output / 'eth3d-scenes' / args.scene / args.model
    if out.exists():
        raise FileExistsError(f'{out} already exists; select a fresh --output run directory')
    out.mkdir(parents=True)
    if args.dataset == 'da2k':
        metrics, counts = evaluate_ordinal(predictor, samples, out)
    else:
        metrics = evaluate(predictor, samples, cfg, alignment='least_square_disparity',
                           output_dir=str(out), save_predictions=False)
        with open(out / 'per_sample.csv', newline='') as f:
            rows = list(csv.DictReader(f))
        if [r['filename'] for r in rows] != [s['name'] for s in samples]:
            raise ValueError('Upstream evaluator skipped/reordered samples; rejecting run')
        import math
        if any(not math.isfinite(float(r[k])) for r in rows for k in metrics):
            raise ValueError('Non-finite per-image metrics; rejecting run')
        counts = dict(images=len(rows))
        if args.dataset == 'diode':
            # Same per-image averaging as upstream, also report each domain.
            for domain in ('indoors', 'outdoor'):
                domain_rows = [r for r in rows if r['filename'].startswith(domain + '/')]
                counts[domain] = len(domain_rows)
                if not domain_rows:
                    raise ValueError(f'DIODE missing {domain}; refusing partial benchmark')
                for key in list(metrics):
                    if '/' not in key:
                        metrics[f'{domain}/{key}'] = sum(float(r[key]) for r in domain_rows) / len(domain_rows)
    record = dict(model=args.model, dataset=args.dataset, metrics=metrics, counts=counts,
                  sample_manifest_sha256=manifest_sha, scene=args.scene,
                  subset=bool(args.limit), backend=args.device, threads=args.threads, dtype='float32',
                  protocol='ordinal-pairs-v1' if cfg is None else 'zipdepth-least-square-disparity-v1',
                  dataset_config=cfg, checkpoint_sha256={p.name: digest(p) for p in predictor.files},
                  input_size=predictor.size or 'short-side-384-round32', sources=sources,
                  torch=torch.__version__, platform=platform.platform(),
                  timestamp=datetime.now(timezone.utc).isoformat(),
                  evaluator_sha256=digest(args.upstream / 'ZipDepth/zipdepth/evaluation/evaluator.py'),
                  harness_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='File; RT uses directory containing encoder.pth and depth.pth')
    parser.add_argument('--data', type=Path, required=True, help='Dataset parent; may be on an external volume')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--threads', type=int, default=4, help='CPU intra-op threads (accuracy only)')
    parser.add_argument('--scene', choices=tuple(ETH3D_SCENES), help='Evaluate one ETH3D scene for later exact aggregation')
    parser.add_argument('--upstream', type=Path, default=ROOT / 'upstream')
    parser.add_argument('--output', type=Path, default=ROOT / 'accuracy/results')
    parser.add_argument('--limit', type=int, help='Smoke test only; excluded from comparison table')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.scene and args.dataset != 'eth3d':
        parser.error('--scene requires --dataset eth3d')
    run(args)


if __name__ == '__main__':
    main()
