import io
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.image import decode_image, render_preview_png


class ImageDecodeTestCase(unittest.TestCase):
    def test_decode_png_image(self):
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        image[:, :, 1] = 200
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        decoded = decode_image(io.BytesIO(encoded.tobytes()))
        self.assertEqual(decoded.shape, image.shape)
        self.assertEqual(decoded.dtype, np.uint8)

    def test_reject_empty_file(self):
        with self.assertRaises(ValueError):
            decode_image(io.BytesIO(b""))

    def test_render_preview_png(self):
        image = np.zeros((20, 10, 3), dtype=np.uint8)
        image[:, :, 2] = 180
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        preview = render_preview_png(io.BytesIO(encoded.tobytes()))
        decoded = cv2.imdecode(np.frombuffer(preview, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
