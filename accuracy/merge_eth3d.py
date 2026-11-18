"""Merge per-image results from all ETH3D scenes, without averaging scene means."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .benchmark import ETH3D_SCENES, MODELS, digest
from .report import evaluation_signature


def mean_rows(rows):
    keys = [k for k in rows[0] if k != 'filename']
    if any(not math.isfinite(float(row[k])) for row in rows for k in keys):
        raise ValueError('Non-finite per-image metric')
    return {k: math.fsum(float(row[k]) for row in rows) / len(rows) for k in keys}


def merge(root, model):
    all_rows, manifest, records = [], [], []
    for scene, count in ETH3D_SCENES.items():
        src = root / 'eth3d-scenes' / scene / model
        record = json.loads((src / 'result.json').read_text())
        entries = json.loads((src / 'manifest.json').read_text())
        with open(src / 'per_sample.csv', newline='') as f:
            rows = list(csv.DictReader(f))
        if (record['model'] != model or record['dataset'] != 'eth3d' or
                record['scene'] != scene or record['subset'] or
                record['counts']['images'] != count or len(entries) != count or len(rows) != count):
            raise ValueError(f'Incomplete/wrong scene: {scene}/{model}')
        if [r['filename'] for r in rows] != [e['name'] for e in entries]:
            raise ValueError(f'Manifest/CSV mismatch: {scene}')
        if any(e['name'].split('/')[0] != scene for e in entries):
            raise ValueError(f'Wrong scene names: {scene}')
        sha = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        if sha != record['sample_manifest_sha256']:
            raise ValueError(f'Manifest hash mismatch: {scene}')
        if records:
            for key in ('backend', 'dtype', 'protocol', 'checkpoint_sha256', 'input_size',
                        'sources', 'dataset_config', 'evaluator_sha256', 'threads'):
                if record[key] != records[0][key]:
                    raise ValueError(f'Incompatible {key} in {scene}')
            if evaluation_signature(record) != evaluation_signature(records[0]):
                raise ValueError(f'Incompatible evaluation code in {scene}')
        records.append(record)
        all_rows.extend(rows)
        manifest.extend(entries)
    all_rows.sort(key=lambda r: r['filename'])
    manifest.sort(key=lambda e: e['name'])
    if len({e['name'] for e in manifest}) != sum(ETH3D_SCENES.values()):
        raise ValueError('Duplicate/missing ETH3D images')
    out = root / 'eth3d' / model
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    record = dict(records[-1], metrics=mean_rows(all_rows), counts={'images': len(all_rows)},
                  scene=None, sample_manifest_sha256=hashlib.sha256(
                      json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  aggregation='mean of all per-image metrics, not mean of scene means',
                  scene_result_sha256={s: digest(root / 'eth3d-scenes' / s / model / 'result.json')
                                       for s in ETH3D_SCENES})
    with open(out / 'per_sample.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('accuracy/results'))
    args = parser.parse_args()
    for model in MODELS:
        merge(args.results, model)


if __name__ == '__main__':
    main()
