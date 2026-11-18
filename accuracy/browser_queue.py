"""Evaluate ETH3D archives downloaded by the browser into owned scratch storage.

No browser automation or network requests here. Waits for final archive filenames
(not .crdownload), evaluates each scene on CPU, verifies scores, and cleans up.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .benchmark import ROOT, ETH3D_SCENES, MODELS, digest
from .sequential import verify_complete, extract
from .merge_eth3d import merge
from .report import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, default=ROOT / 'accuracy/data')
    parser.add_argument('--data', type=Path, default=ROOT / 'accuracy/data/browser-session')
    parser.add_argument('--output', type=Path, default=ROOT / 'accuracy/results')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    args.archives, args.data, args.output = [p.resolve() for p in (args.archives, args.data, args.output)]
    if args.output.is_relative_to(args.data):
        parser.error('--output must be outside temporary data')
    weights = dict(zip(MODELS, (ROOT / 'accuracy/checkpoints/NYU_Half_GuideDepth-S.pth',
                   ROOT / 'upstream/ZipDepth/checkpoints/zipdepth_base.pth',
                   ROOT / 'accuracy/checkpoints/RTMonoDepth/s/m_640_192')))
    logs = args.output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    for scene in ETH3D_SCENES:
        result = args.output / 'eth3d-scenes' / scene
        if all((result / m / 'result.json').is_file() for m in MODELS):
            verify_complete(args.output, 'eth3d', scene)
            print(f'Already complete: {scene}', flush=True)
            continue
        archives = [args.archives / f'{scene}_dslr_{kind}.7z' for kind in ('undistorted', 'depth')]
        print(f'WAITING: {scene}; download both archives into {args.archives}', flush=True)
        deadline = time.monotonic() + 3600
        while not all(p.is_file() for p in archives):
            if time.monotonic() > deadline:
                raise TimeoutError(f'Timed out waiting for {scene} browser downloads')
            time.sleep(3)
        if shutil.disk_usage(args.data).free < 12 * 1024**3:
            raise RuntimeError('Need at least 12 GiB free before ETH3D extraction; data retained')
        receipt = {p.name: dict(sha256=digest(p), bytes=p.stat().st_size,
                               source=f'https://www.eth3d.net/data/{p.name}') for p in archives}
        print(f'EXTRACTING: {scene}', flush=True)
        for archive, subdir in zip(archives, ('images', 'depth')):
            extract(archive, args.data / 'ETH3D' / subdir)
        for model, checkpoint in weights.items():
            if (result / model / 'result.json').is_file():
                continue
            if (result / model).exists():
                # Preserve an interrupted writer's partial outputs, then retry it.
                (result / model).rename(logs / f'interrupted-{scene}-{model}-{time.time_ns()}')
            print(f'EVALUATING: {scene}/{model}', flush=True)
            with open(logs / f'eth3d-{scene}-{model}.log', 'w') as log:
                subprocess.run([sys.executable, '-m', 'accuracy.benchmark', '--device', 'cpu',
                                '--threads', str(args.threads), '--dataset', 'eth3d', '--scene', scene,
                                '--model', model, '--checkpoint', str(checkpoint), '--data', str(args.data),
                                '--output', str(args.output)], cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        verify_complete(args.output, 'eth3d', scene)
        (result / 'downloads.json').write_text(json.dumps(receipt, indent=2) + '\n')
        for subdir in ('images', 'depth'):
            shutil.rmtree(args.data / 'ETH3D' / subdir / scene)
        for archive in archives:
            archive.unlink()
        print(f'COMPLETE: {scene}; three results verified; owned images, depth and archives deleted', flush=True)
    for model in MODELS:
        if not (args.output / 'eth3d' / model / 'result.json').is_file():
            merge(args.output, model)
    (ROOT / 'accuracy/RESULTS.md').write_text(render(args.output))
    print('COMPLETE: all ETH3D scenes merged; final comparison table written', flush=True)


if __name__ == '__main__':
    main()
