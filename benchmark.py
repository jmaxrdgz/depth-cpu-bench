#!/usr/bin/env python3
"""Batch-one FP32 PyTorch CPU forward latency on pre-generated uniform RGB noise."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

from adapters import MODELS, REPOS

ROOT = Path(__file__).resolve().parent


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--height', type=positive, default=192)
    parser.add_argument('--width', type=positive, default=640)
    parser.add_argument('--threads', type=positive, default=1)
    parser.add_argument('--warmup', type=nonnegative, default=30)
    parser.add_argument('--iterations', type=positive, default=200)
    parser.add_argument('--noise-images', type=positive, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--channels-last', action='store_true')
    parser.add_argument('--upstream-root', type=Path, default=ROOT / 'upstream')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'cpu.json')
    parser.add_argument('--worker', choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.height % 32 or args.width % 32 or min(args.height, args.width) < 64:
        parser.error('height and width must be multiples of 32 and at least 64 (shared upstream constraints)')
    return args


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def source_info(args):
    repo_name = REPOS[args.worker]
    path = args.upstream_root / repo_name
    expected = json.loads((ROOT / 'upstreams.json').read_text())[repo_name]['revision']
    def git(*options):
        return subprocess.check_output(['git', '-C', str(path), *options], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        # Avoid accidentally reporting a parent checkout's revision.
        if not (path / '.git').exists():
            raise ValueError('source snapshot, not a clone')
        actual, dirty = git('rev-parse', 'HEAD'), bool(git('status', '--porcelain', '--untracked-files=no'))
        return dict(revision=actual, expected_revision=expected, dirty=dirty,
                    verified=actual == expected and not dirty)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return dict(revision=None, expected_revision=expected, dirty=None, verified=False)


def worker(args):
    import torch
    from adapters import build_model

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    model = build_model(args.worker, args.upstream_root, (args.height, args.width)).to(device='cpu', dtype=torch.float32).eval()
    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
    model = model.to(memory_format=memory_format)
    inputs = [torch.rand(1, 3, args.height, args.width, device='cpu', dtype=torch.float32)
              .contiguous(memory_format=memory_format) for _ in range(args.noise_images)]
    durations = []
    with torch.inference_mode():
        first = model(inputs[0])
        if tuple(first.shape) != (1, 1, args.height, args.width) or not torch.isfinite(first).all().item():
            raise RuntimeError(f'invalid output: {tuple(first.shape)}')
        output_shape = list(first.shape)
        del first
        for index in range(args.warmup):
            output = model(inputs[index % len(inputs)])
            del output
        for index in range(args.iterations):
            image = inputs[index % len(inputs)]
            start = time.perf_counter_ns()
            output = model(image)
            end = time.perf_counter_ns()
            durations.append((end - start) / 1e6)
            del output
    mean = statistics.mean(durations)
    result = dict(model=args.worker, status='ok', mean_ms=mean,
                  median_ms=statistics.median(durations), p90_ms=percentile(durations, .90),
                  p95_ms=percentile(durations, .95), min_ms=min(durations), max_ms=max(durations),
                  stdev_ms=statistics.pstdev(durations), fps=1000 / mean,
                  parameters=sum(p.numel() for p in model.parameters()),
                  output_shape=output_shape, samples_ms=durations,
                  source=source_info(args), torch_version=torch.__version__,
                  torch_config=torch.__config__.show(),
                  intraop_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
    args.worker_output.write_text(json.dumps(result, indent=2) + '\n')


def main():
    args = parse_args()
    if args.worker:
        worker(args)
        return
    args.upstream_root = args.upstream_root.resolve()
    missing = [name for name in set(REPOS[m] for m in args.models) if not (args.upstream_root / name).is_dir()]
    if missing:
        raise SystemExit(f'Missing upstream clones: {", ".join(sorted(missing))}. Run python setup_upstreams.py first.')
    report = dict(schema_version=1, timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  machine=dict(platform=platform.platform(), processor=platform.processor(),
                               machine=platform.machine(), logical_cpus=os.cpu_count(), python=sys.version),
                  settings=dict(height=args.height, width=args.width, batch_size=1, dtype='float32',
                                device='cpu', backend='pytorch-eager', weights='random',
                                noise='uniform [0,1), pre-generated ring', noise_images=args.noise_images,
                                seed=args.seed, warmup=args.warmup, iterations=args.iterations,
                                threads=args.threads, channels_last=args.channels_last,
                                timing='forward only; excludes generation, imports, initialization and validation'),
                  results=[])
    print(f'CPU FP32 | 1x3x{args.height}x{args.width} | threads={args.threads} | random weights', flush=True)
    with tempfile.TemporaryDirectory(prefix='depth-bench-') as directory:
        for name in args.models:
            output = Path(directory) / f'{name}.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', name,
                       '--worker-output', str(output), '--upstream-root', str(args.upstream_root)]
            for option in ('height', 'width', 'threads', 'warmup', 'iterations', 'noise_images', 'seed'):
                command.extend(['--' + option.replace('_', '-'), str(getattr(args, option))])
            if args.channels_last:
                command.append('--channels-last')
            env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS=str(args.threads), CUDA_VISIBLE_DEVICES='')
            process = subprocess.run(command, text=True, capture_output=True, env=env)
            if process.returncode:
                result = dict(model=name, status='error', error=process.stderr or process.stdout)
                print(f'{name:18} ERROR: {result["error"].strip()}', flush=True)
            else:
                result = json.loads(output.read_text())
                print(f'{name:18} median={result["median_ms"]:8.2f} ms  p95={result["p95_ms"]:8.2f} ms '
                      f' FPS={result["fps"]:7.2f}', flush=True)
                if not result['source']['verified']:
                    print('  WARNING: source revision is unverified; see JSON provenance.', flush=True)
            report['results'].append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    fields = ['model', 'status', 'mean_ms', 'median_ms', 'p90_ms', 'p95_ms', 'min_ms', 'max_ms',
              'stdev_ms', 'fps', 'parameters']
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['results'])
    print(f'Wrote {args.output} and {args.output.with_suffix(".csv")}', flush=True)
    if any(row['status'] != 'ok' for row in report['results']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
