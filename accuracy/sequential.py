"""Download one dataset/ETH3D scene, evaluate all models, then delete its raw data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import zipfile

from .benchmark import ROOT, MODELS, ETH3D_SCENES, digest, verify_upstreams
from .merge_eth3d import merge
from .report import render, evaluation_signature

DA2K = 'https://huggingface.co/datasets/depth-anything/DA-2K/resolve/main/DA-2K.zip'
DIODE = 'https://diode-dataset.s3.amazonaws.com/val.tar.gz'


def download(url, dest, md5=None):
    print(f'Downloading {url}', flush=True)
    with urlopen(url, timeout=60) as response, open(dest, 'xb') as f:
        shutil.copyfileobj(response, f, length=1024 * 1024)
    if md5:
        h = hashlib.md5()
        with open(dest, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != md5:
            raise ValueError(f'Checksum mismatch for {dest}')


def extract(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    # bsdtar handles ZIP, gzip tar and ETH3D 7z. It rejects absolute/.. paths
    # by default (do not pass -P). Extraction occurs only inside task-owned temp dirs.
    subprocess.run(['bsdtar', '-xf', str(archive), '-C', str(target)], check=True)


def ensure_rt_checkpoint(path, checkpoints):
    if all((path / name).is_file() for name in ('encoder.pth', 'depth.pth')):
        return path
    import gdown
    checkpoints.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='rt-download-', dir=checkpoints) as temp:
        archive = Path(temp) / 'weights.zip'
        result = gdown.download(id='1Jf5K3m0DfAqVcVCE6y0cKufEKIHu86sz', output=str(archive), quiet=False)
        if not result or not zipfile.is_zipfile(archive):
            raise ValueError('RT release was not a ZIP archive; supply --rt-checkpoint with official extracted small monocular weights')
        with zipfile.ZipFile(archive) as z:
            selected = {}
            for name in ('encoder.pth', 'depth.pth'):
                candidates = [s for s in z.namelist() if s.endswith('/s/m_640_192/' + name)]
                if len(candidates) != 1:
                    raise ValueError(f'Cannot uniquely identify official small monocular {name}; supply --rt-checkpoint')
                selected[name] = z.read(candidates[0])
            path.mkdir(parents=True, exist_ok=True)
            for name, content in selected.items():
                (path / name).write_bytes(content)
    return path


def verify_complete(output, dataset, scene=None):
    """Require all three complete, comparable records before deleting owned data."""
    expected = ETH3D_SCENES[scene] if scene else {'da2k': 1033, 'diode': 771}[dataset]
    records = []
    for model in MODELS:
        base = output / 'eth3d-scenes' / scene / model if scene else output / dataset / model
        r = json.loads((base / 'result.json').read_text())
        manifest = json.loads((base / 'manifest.json').read_text())
        with open(base / 'per_sample.csv', newline='') as f:
            rows = list(csv.DictReader(f))
        if (r['model'] != model or r['dataset'] != dataset or r.get('scene') != scene or
                r['subset'] or r['backend'] != 'cpu' or r['counts']['images'] != expected or
                len(rows) != expected or len(manifest) != expected):
            raise ValueError(f'Incomplete CPU result: {dataset}/{scene}/{model}')
        if [row['filename'] for row in rows] != [e['name'] for e in manifest]:
            raise ValueError('CSV differs from input manifest')
        sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if sha != r['sample_manifest_sha256']:
            raise ValueError('Input manifest hash mismatch')
        if dataset == 'da2k' and (r['counts']['pairs'] != 2068 or sum(int(row['pairs']) for row in rows) != 2068):
            raise ValueError('Incomplete DA-2K pairs')
        records.append(r)
    for key in ('sample_manifest_sha256', 'counts', 'protocol', 'dataset_config',
                'sources', 'evaluator_sha256'):
        if any(r[key] != records[0][key] for r in records):
            raise ValueError(f'Incompatible {key}; data retained')
    if any(evaluation_signature(r) != evaluation_signature(records[0]) for r in records):
        raise ValueError('Incompatible evaluation code; data retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=ROOT / 'accuracy/data')
    parser.add_argument('--output', type=Path, default=ROOT / 'accuracy/results')
    parser.add_argument('--guide-checkpoint', type=Path, default=Path('/tmp/NYU_Half_GuideDepth-S.pth'))
    parser.add_argument('--rt-checkpoint', type=Path, default=ROOT / 'accuracy/checkpoints/RTMonoDepth/s/m_640_192')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if not shutil.which('bsdtar'):
        raise SystemExit('bsdtar is required for ETH3D 7z extraction')
    verify_upstreams(ROOT / 'upstream')
    # Probe before creating scratch files or downloading hundreds of MB.
    for host in ('huggingface.co', 'diode-dataset.s3.amazonaws.com', 'www.eth3d.net', 'drive.google.com'):
        try:
            with socket.create_connection((host, 443), timeout=15):
                pass
        except OSError as e:
            raise SystemExit(f'Network preflight failed for {host}: {e}. No dataset downloaded.') from e
    import tqdm  # fail early on missing dependencies
    checkpoints = ROOT / 'accuracy/checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    guide = checkpoints / 'NYU_Half_GuideDepth-S.pth'
    if not guide.exists():
        shutil.copyfile(args.guide_checkpoint, guide)
    rt = ensure_rt_checkpoint(args.rt_checkpoint, checkpoints)
    weights = dict(zip(MODELS, (guide, ROOT / 'upstream/ZipDepth/checkpoints/zipdepth_base.pth', rt)))
    # Confirm all released weights strictly load BEFORE consuming dataset bandwidth.
    for model, checkpoint in weights.items():
        subprocess.run([sys.executable, '-c',
                        'from pathlib import Path; from accuracy.models import Predictor; '
                        f'Predictor({model!r}, Path({str(checkpoint)!r}), Path("upstream"), "cpu")'],
                       cwd=ROOT, check=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    if args.output.resolve().is_relative_to(args.scratch.resolve()):
        raise SystemExit('--output must be outside scratch storage')
    batches = [('da2k', None), ('diode', None)] + [('eth3d', s) for s in ETH3D_SCENES]
    for dataset, scene in batches:
        result_base = args.output / 'eth3d-scenes' / scene if scene else args.output / dataset
        if all((result_base / m / 'result.json').exists() for m in MODELS):
            verify_complete(args.output, dataset, scene)
            continue
        if shutil.disk_usage(args.scratch).free < 16 * 1024**3:
            raise SystemExit('Need at least 16 GiB free for the next dataset/scene; completed results retained')
        # Only this newly-created, owned directory may be deleted. On failure retain
        # its files for diagnosis; never delete user-provided dataset directories.
        work = Path(tempfile.mkdtemp(prefix=f'{scene or dataset}-', dir=args.scratch)).resolve()
        try:
            if dataset == 'da2k':
                archive = work / 'DA-2K.zip'
                download(DA2K, archive)
                extract(archive, work / 'unpacked')
                roots = list((work / 'unpacked').rglob('annotations.json'))
                if len(roots) != 1:
                    raise ValueError('Cannot uniquely locate DA-2K annotations.json')
                shutil.move(str(roots[0].parent), str(work / 'DA-2K'))
                archive.unlink()
            elif dataset == 'diode':
                archive = work / 'val.tar.gz'
                download(DIODE, archive, md5='5c895d09201b88973c8fe4552a67dd85')
                extract(archive, work / 'DIODE')
                archive.unlink()
            else:
                for kind, target in (('undistorted', 'images'), ('depth', 'depth')):
                    archive = work / f'{scene}_dslr_{kind}.7z'
                    download(f'https://www.eth3d.net/data/{archive.name}', archive)
                    extract(archive, work / 'ETH3D' / target)
                    archive.unlink()
            for model, checkpoint in weights.items():
                if (result_base / model / 'result.json').exists():
                    continue
                command = [sys.executable, '-m', 'accuracy.benchmark', '--device', 'cpu',
                           '--threads', str(args.threads), '--dataset', dataset, '--model', model,
                           '--checkpoint', str(checkpoint), '--data', str(work), '--output', str(args.output)]
                if scene:
                    command += ['--scene', scene]
                subprocess.run(command, cwd=ROOT, check=True)
            verify_complete(args.output, dataset, scene)
        except BaseException:
            print(f'Run failed; downloaded files retained at {work}', file=sys.stderr)
            raise
        else:
            shutil.rmtree(work)
            print(f'Completed {scene or dataset}: results verified, owned raw data deleted', flush=True)
    for model in MODELS:
        if not (args.output / 'eth3d' / model / 'result.json').exists():
            merge(args.output, model)
    (ROOT / 'accuracy/RESULTS.md').write_text(render(args.output))


if __name__ == '__main__':
    main()
