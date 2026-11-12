"""Check architecture contracts that matter for interpreting benchmark timings."""
from pathlib import Path
import unittest

import torch

from adapters import build_model
from lnca import LNCADepth

ROOT = Path(__file__).resolve().parents[1]


class NewModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_lnca_parameter_budget_and_shared_steps(self):
        model = LNCADepth().eval()
        counts = {name: sum(p.numel() for p in module.parameters())
                  for name, module in model.named_children()}
        self.assertEqual(counts, dict(encoder=19392, perception=288, update=2656, decoder=18496))
        self.assertEqual(sum(counts.values()), 40832)
        calls = []
        handle = model.update.register_forward_hook(lambda *unused: calls.append(1))
        with torch.inference_mode():
            output = model(torch.rand(1, 3, 64, 96))
        handle.remove()
        self.assertEqual(len(calls), 4)
        self.assertEqual(tuple(output.shape), (1, 1, 64, 96))
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue((output > 0).all().item())

    def test_lnca_needs_no_upstream_checkout(self):
        model = build_model('lnca-reimplementation', ROOT / 'nonexistent')
        self.assertIsInstance(model, LNCADepth)

    def test_zipdepth_paths_and_rectangular_output(self):
        if not (ROOT / 'upstream/ZipDepth/zipdepth').is_dir():
            self.skipTest('Run setup_upstreams.py to test ZipDepth integration')
        for name, unfold in [('zipdepth', True), ('zipdepth-npu', False)]:
            with self.subTest(model=name):
                model = build_model(name, ROOT / 'upstream').eval()
                self.assertEqual(model.decoder.convex_up.use_unfold, unfold)
                self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))
                with torch.inference_mode():
                    output = model(torch.rand(1, 3, 64, 96))
                self.assertEqual(tuple(output.shape), (1, 1, 64, 96))
                self.assertTrue(torch.isfinite(output).all().item())

    def test_unknown_model_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown model'):
            build_model('unknown', ROOT / 'upstream')


if __name__ == '__main__':
    unittest.main()
