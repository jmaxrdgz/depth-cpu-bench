"""Generate a table only from complete, comparable measured runs."""
import argparse
import json
import math
from pathlib import Path
from .benchmark import MODELS, DATASETS
MODEL_LABELS = {'guidedepth-s': 'GuideDepth-S', 'zipdepth': 'ZipDepth',
                'rt-monodepth-s': 'RT-MonoDepth-S'}


def evaluation_signature(record):
    # Downloader/table changes do not alter predictions or metric computation.
    # Keep all file hashes in provenance, but compare evaluation code itself.
    return {k: v for k, v in record['harness_sha256'].items()
            if k in ('benchmark.py', 'models.py', '__init__.py')}


def highlight(value, values, lower):
    ranked = sorted(set(values), reverse=not lower)
    text = f'{value:.4f}'
    if value == ranked[0]:
        return f'**{text}**'
    if len(ranked) > 1 and value == ranked[1]:
        return f'<ins>{text}</ins>'
    return text


def render(root):
    records = {}
    for dataset in DATASETS:
        for model in MODELS:
            r = json.loads((root / dataset / model / 'result.json').read_text())
            if r['subset'] or r['backend'] not in ('cpu', 'mps') or r.get('scene'):
                raise ValueError('Comparison requires full dataset runs, not subsets or individual scenes')
            if r['model'] != model or r['dataset'] != dataset:
                raise ValueError('Misplaced result file')
            if any(not math.isfinite(v) for v in r['metrics'].values()):
                raise ValueError('Non-finite metric in result file')
            records[dataset, model] = r
        group = [records[dataset, m] for m in MODELS]
        for field in ('sample_manifest_sha256', 'protocol', 'counts', 'dataset_config',
                      'sources', 'evaluator_sha256', 'dtype', 'backend'):
            if any(r[field] != group[0][field] for r in group):
                raise ValueError(f'{dataset}: incompatible {field}')
        if any(evaluation_signature(r) != evaluation_signature(group[0]) for r in group):
            raise ValueError(f'{dataset}: incompatible evaluation code')
        expected = {'da2k': 1033, 'diode': 771, 'eth3d': 454}[dataset]
        if group[0]['counts']['images'] != expected:
            raise ValueError(f'{dataset}: expected {expected} images')
        if dataset == 'da2k' and group[0]['counts']['pairs'] != 2068:
            raise ValueError('DA-2K requires 2068 pairs')
    for model in MODELS:
        for dataset in DATASETS:
            for field in ('checkpoint_sha256', 'input_size', 'backend'):
                if records[dataset, model][field] != records['da2k', model][field]:
                    raise ValueError(f'{model}: changed {field} across datasets')
    lines = ['# Pretrained relative-depth accuracy', '',
             f'FP32 {records["da2k", MODELS[0]]["backend"].upper()}; released weights; no training or test-time augmentation.',
             'Best: **bold**. Second distinct best: <ins>underlined</ins>. Ties share rank.', '',
             'Dense metrics use per-image least-squares scale/shift in inverse depth,',
             'then clipped depth metrics from the pinned ZipDepth evaluator.',
             'These scores do not establish metric scale accuracy.', '']
    lines += ['[Protocol, checkpoints and input shapes](README.md) · '
              '[Recorded environment](results/environment.json)', '']
    for dataset in DATASETS:
        sections = [(dataset.upper(), '')]
        if dataset == 'diode':
            sections += [('DIODE indoors', 'indoors/'), ('DIODE outdoor', 'outdoor/')]
        for title, prefix in sections:
            columns = [('pair_accuracy', 'Pair accuracy ↑', False)] if dataset == 'da2k' else [
                ('abs_rel', 'AbsRel ↓', True), ('sq_rel', 'SqRel ↓', True),
                ('rmse', 'Aligned RMSE (m) ↓', True), ('rmse_log', 'RMSElog ↓', True),
                ('log10', 'log10 ↓', True), ('delta1', 'δ1 ↑', False),
                ('delta2', 'δ2 ↑', False), ('delta3', 'δ3 ↑', False)]
            lines += [f'## {title}', '', '| Model | ' + ' | '.join(c[1] for c in columns) + ' |',
                      '| --- | ' + ' | '.join('---:' for c in columns) + ' |']
            for model in MODELS:
                cells = []
                for key, _, lower in columns:
                    key = prefix + key
                    values = [records[dataset, m]['metrics'][key] for m in MODELS]
                    cells.append(highlight(records[dataset, model]['metrics'][key], values, lower))
                lines.append(f'| {MODEL_LABELS[model]} | ' + ' | '.join(cells) + ' |')
            lines += ['']
    lines += ['## Recorded runs', '']
    for (dataset, model), r in records.items():
        lines.append(f'- [{dataset}/{model}](results/{dataset}/{model}/result.json): '
                     f'{r["counts"]["images"]} images; {r["timestamp"]}')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('accuracy/results'))
    parser.add_argument('--output', type=Path, default=Path('accuracy/RESULTS.md'))
    args = parser.parse_args()
    args.output.write_text(render(args.results))


if __name__ == '__main__':
    main()
