import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from accuracy.benchmark import MODELS, ETH3D_SCENES
from accuracy.merge_eth3d import mean_rows, merge
from accuracy.sequential import verify_complete


class SequentialTests(unittest.TestCase):
    def test_scene_counts(self):
        self.assertEqual(len(ETH3D_SCENES), 13)
        self.assertEqual(sum(ETH3D_SCENES.values()), 454)

    def test_image_weighting(self):
        rows = [dict(filename='small/a', abs_rel='0')] + [
            dict(filename=f'large/{i}', abs_rel='1') for i in range(3)]
        self.assertEqual(mean_rows(rows)['abs_rel'], 0.75)
        rows[0]['abs_rel'] = 'nan'
        with self.assertRaises(ValueError):
            mean_rows(rows)

    def test_merge_and_cleanup_gate(self):
        # Reduced synthetic scene set exercises the actual file/provenance path.
        scenes = {'first': 1, 'second': 3}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for scene, count in scenes.items():
                for model in MODELS:
                    out = root / 'eth3d-scenes' / scene / model
                    out.mkdir(parents=True)
                    manifest = [dict(name=f'{scene}/{i}', image_sha256=str(i)) for i in range(count)]
                    sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
                    r = dict(model=model, dataset='eth3d', scene=scene, subset=False,
                             backend='cpu', dtype='float32', protocol='test', checkpoint_sha256={},
                             input_size=[64, 96], sources={}, dataset_config={}, evaluator_sha256='e',
                             harness_sha256={}, threads=1, counts={'images': count},
                             metrics={'abs_rel': 0 if scene == 'first' else 1},
                             sample_manifest_sha256=sha)
                    (out / 'result.json').write_text(json.dumps(r))
                    (out / 'manifest.json').write_text(json.dumps(manifest))
                    with open(out / 'per_sample.csv', 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['filename', 'abs_rel'])
                        writer.writeheader()
                        writer.writerows(dict(filename=e['name'], abs_rel=r['metrics']['abs_rel']) for e in manifest)
            with patch('accuracy.sequential.ETH3D_SCENES', scenes):
                verify_complete(root, 'eth3d', 'first')
                rfile = root / 'eth3d-scenes/first/zipdepth/result.json'
                record = json.loads(rfile.read_text())
                record['subset'] = True
                rfile.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    verify_complete(root, 'eth3d', 'first')
                record['subset'] = False
                rfile.write_text(json.dumps(record))
            with patch('accuracy.merge_eth3d.ETH3D_SCENES', scenes):
                merge(root, 'zipdepth')
                record = json.loads((root / 'eth3d/zipdepth/result.json').read_text())
                self.assertEqual(record['counts']['images'], 4)
                self.assertEqual(record['metrics']['abs_rel'], 0.75)
                self.assertIsNone(record['scene'])
                self.assertEqual(len(record['scene_result_sha256']), 2)


if __name__ == '__main__':
    unittest.main()
