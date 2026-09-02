import unittest

import cv2
import numpy as np

from segmentation import coverage_percent, prepare_multisegment, segment_image


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        image = np.full((180, 260, 3), 220, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (129, 159), (35, 35, 35), -1)
        gradient = np.linspace(0, 35, image.shape[1], dtype=np.uint8)
        self.image = np.clip(image.astype(np.int16) - gradient[None, :, None], 0, 255).astype(np.uint8)

    def test_otsu_detects_dark_foreground(self):
        result = segment_image(self.image, "otsu", foreground_dark=True)
        self.assertGreater(coverage_percent(result.mask), 25)
        self.assertLess(coverage_percent(result.mask), 70)

    def test_multisegment_has_ordered_regions_and_thresholds(self):
        prep = prepare_multisegment(self.image, max_regions=4)
        region_count = len(np.unique(prep.region_map))
        self.assertGreaterEqual(region_count, 2)
        self.assertEqual(region_count, len(prep.automatic_thresholds))

    def test_fixed_profile_is_reused(self):
        values = {0: 90, 1: 110, 2: 130, 3: 150}
        result = segment_image(
            self.image,
            "multisegment",
            max_regions=4,
            region_values=values,
            application_mode="fixed",
        )
        self.assertTrue(set(result.applied_thresholds).issubset(set(values.values())))
        self.assertEqual(result.mask.dtype, np.uint8)

    def test_relative_profile_offsets_automatic_thresholds(self):
        base = segment_image(self.image, "multisegment", max_regions=3)
        shifted = segment_image(
            self.image,
            "multisegment",
            max_regions=3,
            region_values={0: 5, 1: 5, 2: 5},
            application_mode="relative",
        )
        for original, adjusted in zip(base.applied_thresholds, shifted.applied_thresholds):
            self.assertAlmostEqual(min(255, original + 5), adjusted)

    def test_missing_fixed_region_falls_back_to_automatic(self):
        base = segment_image(self.image, "multisegment", max_regions=4)
        partial = segment_image(
            self.image,
            "multisegment",
            max_regions=4,
            region_values={0: 90},
            application_mode="fixed",
        )
        self.assertEqual(partial.applied_thresholds[0], 90)
        for original, applied in zip(base.applied_thresholds[1:], partial.applied_thresholds[1:]):
            self.assertAlmostEqual(original, applied)


if __name__ == "__main__":
    unittest.main()
