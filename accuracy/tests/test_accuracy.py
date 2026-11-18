import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from accuracy.benchmark import ROOT, MODELS, DATASETS, ordinal_metrics, da2k_samples, validate_dense
from accuracy.report import render, highlight

sys.path.insert(0, str(ROOT / 'upstream/ZipDepth'))
from zipdepth.evaluation.datasets import DATASET_CONFIGS, discover_samples, load_gt
from zipdepth.evaluation.evaluator import evaluate


class OrdinalTests(unittest.TestCase):
    def test_coordinates_polarity_and_ties(self):
        pred = np.array([[1., 2., 3.], [4., 5., 6.]])
        pair = dict(point1=[1, 2], point2=[0, 1], closer_point='point1')
        self.assertEqual(ordinal_metrics(pred, [pair]), (1, 1))
        self.assertEqual(ordinal_metrics(-pred, [pair]), (0, 1))
        self.assertEqual(ordinal_metrics(np.ones_like(pred), [pair]), (0, 1))
        pair['closer_point'] = 'point2'
        self.assertEqual(ordinal_metrics(pred, [pair]), (0, 1))
        pair['point1'] = [-1, 0]
        with self.assertRaises(ValueError):
            ordinal_metrics(pred, [pair])

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'annotations.json').write_text(json.dumps({'../escape.png': [1]}))
            with self.assertRaises(ValueError):
                da2k_samples(root)

    def test_released_zip_filename_encoding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            name = 'images/副本.jpg'
            actual = root / name.encode('utf-8').decode('cp437')
            actual.parent.mkdir()
            actual.write_bytes(b'test')
            (root / 'annotations.json').write_text(json.dumps({name: [dict(
                point1=[0, 0], point2=[0, 1], closer_point='point1')]}))
            sample = da2k_samples(root)[0]
            self.assertEqual(Path(sample['image_path']), actual.resolve())
            self.assertEqual(sample['name'], name)


class DenseTests(unittest.TestCase):
    def test_upstream_diode_loader_mask_and_affine_alignment(self):
        import cv2
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scan = root / 'indoors/scene_00001/scan_00001'
            scan.mkdir(parents=True)
            cv2.imwrite(str(scan / 'sample.png'), np.zeros((8, 8, 3), np.uint8))
            gt = np.linspace(1, 10, 64).reshape(8, 8).astype(np.float32)
            mask = np.ones((8, 8), bool)
            mask[0] = False
            np.save(scan / 'sample_depth.npy', gt[..., None])
            np.save(scan / 'sample_depth_mask.npy', mask)
            samples = discover_samples('diode', str(root), domain='indoors')
            self.assertEqual(len(samples), 1)
            cfg = DATASET_CONFIGS['diode']
            validate_dense(samples, cfg, load_gt)

            class Predictor:
                def infer_image(self, image):
                    # Known affine inverse-depth prediction should align perfectly.
                    return 3.0 / gt + 2.0

            metrics = evaluate(Predictor(), samples, cfg, output_dir=str(root / 'out'))
            self.assertLess(metrics['abs_rel'], 1e-5)
            self.assertEqual(metrics['delta1'], 1.0)
            (scan / 'sample_depth_mask.npy').unlink()
            with self.assertRaises(FileNotFoundError):
                validate_dense(samples, cfg, load_gt)


class CheckpointTests(unittest.TestCase):
    def test_rt_full_checkpoint_contract_and_missing_key_rejection(self):
        import torch
        from adapters import load_file
        from accuracy.models import Predictor
        module = load_file('test_rt_checkpoint', ROOT / 'upstream/RT-MonoDepth/networks/RTMonoDepth/RTMonoDepth_s.py')
        encoder = module.DepthEncoder()
        decoder = module.DepthDecoder(encoder.num_ch_enc)
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d)
            state = dict(encoder.state_dict(), height=64, width=96, use_stereo=False)
            torch.save(state, checkpoint / 'encoder.pth')
            torch.save(decoder.state_dict(), checkpoint / 'depth.pth')
            pred = Predictor('rt-monodepth-s', checkpoint, ROOT / 'upstream', 'cpu')
            result = pred.infer_image(np.zeros((64, 96, 3), np.uint8))
            self.assertEqual(result.shape, (64, 96))
            self.assertTrue(np.isfinite(result).all())
            del state[next(k for k in state if k.endswith('weight'))]
            torch.save(state, checkpoint / 'encoder.pth')
            with self.assertRaises(RuntimeError):
                Predictor('rt-monodepth-s', checkpoint, ROOT / 'upstream', 'cpu')


class ReportTests(unittest.TestCase):
    def test_ranking_directions_ties(self):
        self.assertEqual(highlight(0.1, [0.1, 0.2, 0.3], True), '**0.1000**')
        self.assertEqual(highlight(0.2, [0.1, 0.2, 0.3], True), '<ins>0.2000</ins>')
        self.assertEqual(highlight(0.9, [0.9, 0.8, 0.8], False), '**0.9000**')
        self.assertEqual(highlight(0.8, [0.9, 0.8, 0.8], False), '<ins>0.8000</ins>')

    def test_full_comparison_and_reject_different_datasets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for dataset in DATASETS:
                for i, model in enumerate(MODELS):
                    metrics = {'pair_accuracy': 0.9 - i * 0.1}
                    for prefix in ('', 'indoors/', 'outdoor/'):
                        for key in ('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'delta1', 'delta2', 'delta3'):
                            metrics[prefix + key] = 0.1 + i * 0.1
                    r = dict(model=model, dataset=dataset, subset=False, backend='cpu',
                             sample_manifest_sha256=dataset, protocol=dataset, dataset_config={},
                             sources={}, evaluator_sha256='x', harness_sha256={}, dtype='float32',
                             counts=dict(images={'da2k': 1033, 'diode': 771, 'eth3d': 454}[dataset], pairs=2068),
                             input_size=[192, 640], checkpoint_sha256={model: 'x'},
                             metrics=metrics, timestamp='synthetic-test-only')
                    dest = root / dataset / model
                    dest.mkdir(parents=True)
                    (dest / 'result.json').write_text(json.dumps(r))
            self.assertIn('<ins>', render(root))
            file = root / 'diode/zipdepth/result.json'
            r = json.loads(file.read_text())
            r['sample_manifest_sha256'] = 'different'
            file.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError, 'sample_manifest'):
                render(root)
            r['subset'] = True
            file.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError, 'full dataset'):
                render(root)


if __name__ == '__main__':
    unittest.main()
