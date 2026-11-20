import unittest

import numpy as np

from inference.zipdepth_ncnn import colorize, prepare_frame


class PreprocessingTests(unittest.TestCase):
    def test_color_order_range_and_letterbox_bounds(self):
        image = np.zeros((101, 203, 3), dtype=np.uint8)
        image[:, :, 2] = 255
        tensor, (top, left, height, width) = prepare_frame(image, 384, 384)
        self.assertEqual(tensor.shape, (3, 384, 384))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(tensor[0], 1)
        np.testing.assert_array_equal(tensor[1:], 0)
        self.assertEqual((top, left, height, width), (96, 0, 191, 384))
        self.assertLess(abs(width / height - 203 / 101), .02)

    def test_portrait_and_stretch(self):
        image = np.zeros((203, 101, 3), dtype=np.uint8)
        _, rectangle = prepare_frame(image, 384, 384)
        self.assertEqual(rectangle, (0, 96, 384, 191))
        _, rectangle = prepare_frame(image, 384, 512, 'stretch')
        self.assertEqual(rectangle, (0, 0, 384, 512))

    def test_invalid_input_and_constant_visualization(self):
        with self.assertRaises(ValueError):
            prepare_frame(np.zeros((0, 10, 3), dtype=np.uint8), 384, 384)
        with self.assertRaises(ValueError):
            prepare_frame(np.zeros((10, 10, 3), dtype=np.uint8), 384, 384, 'unknown')
        image = colorize(np.ones((10, 20), dtype=np.float32))
        self.assertEqual(image.shape, (10, 20, 3))
        self.assertEqual(image.dtype, np.uint8)


if __name__ == '__main__':
    unittest.main()
