"""
Unit tests for ABCD metric calculations in MoleAnalyzer.

These tests use synthetic images (circles, ellipses, irregular shapes) so they
can run without model weights or real clinical images.
"""

import os
import sys
import pytest
import numpy as np
import cv2
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics.merged_improved_metrics import MoleAnalyzer


def _make_temp_images(image_rgb, mask_binary):
    """Save an RGB image and binary mask to temp files, return (img_path, mask_path)."""
    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_mask = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_img.close()
    tmp_mask.close()
    cv2.imwrite(tmp_img.name, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(tmp_mask.name, (mask_binary * 255).astype(np.uint8))
    return tmp_img.name, tmp_mask.name


def _cleanup(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Asymmetry tests
# ---------------------------------------------------------------------------

class TestAsymmetry:
    def test_perfect_circle_has_low_asymmetry(self):
        """A perfect circle should have near-zero asymmetry."""
        img = np.zeros((256, 256, 3), dtype=np.uint8) + 128
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 80, 1, -1)
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            mask_uint8 = analyzer.mask.astype(np.uint8) * 255
            asym = analyzer.compute_asymmetry(mask_uint8)
            assert asym is not None
            assert asym < 0.05, f"Circle asymmetry should be ~0, got {asym}"
        finally:
            _cleanup(img_path, mask_path)

    def test_half_circle_has_high_asymmetry(self):
        """A half-circle (semicircle) should have significant asymmetry."""
        img = np.zeros((256, 256, 3), dtype=np.uint8) + 128
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 80, 1, -1)
        mask[128:, :] = 0  # Remove bottom half
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            mask_uint8 = analyzer.mask.astype(np.uint8) * 255
            asym = analyzer.compute_asymmetry(mask_uint8)
            assert asym is not None
            assert asym > 0.3, f"Half-circle asymmetry should be high, got {asym}"
        finally:
            _cleanup(img_path, mask_path)

    def test_empty_mask_returns_none(self):
        """Empty mask should return None (undefined), not 0."""
        img = np.zeros((256, 256, 3), dtype=np.uint8) + 128
        mask = np.zeros((256, 256), dtype=np.uint8)
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            mask_uint8 = np.zeros((256, 256), dtype=np.uint8)
            result = analyzer.compute_asymmetry(mask_uint8)
            assert result is None, f"Empty mask should return None, got {result}"
        finally:
            _cleanup(img_path, mask_path)


# ---------------------------------------------------------------------------
# Border irregularity tests
# ---------------------------------------------------------------------------

class TestBorderIrregularity:
    def test_circle_has_border_near_one(self):
        """A circle's border irregularity should be close to 1.0."""
        img = np.zeros((256, 256, 3), dtype=np.uint8) + 128
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 80, 1, -1)
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            mask_uint8 = analyzer.mask.astype(np.uint8) * 255
            border = analyzer.border_irregularity_index(mask_uint8)
            # Discrete circle perimeter is slightly > ideal, so allow up to 1.15
            assert 0.9 < border < 1.15, f"Circle border should be ~1.0, got {border}"
        finally:
            _cleanup(img_path, mask_path)

    def test_star_shape_has_high_irregularity(self):
        """A star-shaped mask should have border irregularity significantly > 1.0."""
        img = np.zeros((256, 256, 3), dtype=np.uint8) + 128
        mask = np.zeros((256, 256), dtype=np.uint8)
        # Draw a star polygon
        center = (128, 128)
        outer_r, inner_r = 90, 40
        pts = []
        for i in range(10):
            angle = np.pi / 2 + i * np.pi / 5
            r = outer_r if i % 2 == 0 else inner_r
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] - r * np.sin(angle))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            mask_uint8 = analyzer.mask.astype(np.uint8) * 255
            border = analyzer.border_irregularity_index(mask_uint8)
            assert border > 1.3, f"Star border should be irregular (>1.3), got {border}"
        finally:
            _cleanup(img_path, mask_path)

    def test_empty_mask_returns_zero(self):
        """Empty mask should return 0."""
        img = np.zeros((256, 256, 3), dtype=np.uint8) + 128
        mask = np.zeros((256, 256), dtype=np.uint8)
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            mask_uint8 = np.zeros((256, 256), dtype=np.uint8)
            border = analyzer.border_irregularity_index(mask_uint8)
            assert border == 0, f"Empty mask border should be 0, got {border}"
        finally:
            _cleanup(img_path, mask_path)


# ---------------------------------------------------------------------------
# Diameter tests
# ---------------------------------------------------------------------------

class TestDiameter:
    def test_circle_diameter_matches_drawn_radius(self):
        """Feret diameter of a drawn circle should approximate 2 * radius."""
        img = np.zeros((256, 256, 3), dtype=np.uint8) + 128
        mask = np.zeros((256, 256), dtype=np.uint8)
        radius = 60
        cv2.circle(mask, (128, 128), radius, 1, -1)
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            mask_uint8 = analyzer.mask.astype(np.uint8) * 255
            diameter = analyzer.calculate_diameter(mask_uint8)
            expected = 2 * radius
            # Allow 5% tolerance due to discretization
            assert abs(diameter - expected) / expected < 0.05, \
                f"Expected diameter ~{expected}, got {diameter}"
        finally:
            _cleanup(img_path, mask_path)

    def test_horizontal_rectangle_diameter(self):
        """Feret diameter of a rectangle should be its diagonal."""
        img = np.zeros((256, 256, 3), dtype=np.uint8) + 128
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:140, 60:200] = 1  # 40 x 140 rectangle
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            mask_uint8 = analyzer.mask.astype(np.uint8) * 255
            diameter = analyzer.calculate_diameter(mask_uint8)
            expected_diag = np.sqrt(40**2 + 140**2)
            assert abs(diameter - expected_diag) / expected_diag < 0.05, \
                f"Expected diagonal ~{expected_diag:.1f}, got {diameter:.1f}"
        finally:
            _cleanup(img_path, mask_path)

    def test_empty_mask_returns_zero(self):
        """Empty mask should return 0."""
        img = np.zeros((256, 256, 3), dtype=np.uint8) + 128
        mask = np.zeros((256, 256), dtype=np.uint8)
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            mask_uint8 = np.zeros((256, 256), dtype=np.uint8)
            diameter = analyzer.calculate_diameter(mask_uint8)
            assert diameter == 0
        finally:
            _cleanup(img_path, mask_path)


# ---------------------------------------------------------------------------
# Color analysis tests
# ---------------------------------------------------------------------------

class TestColourAnalysis:
    def test_uniform_color_has_low_variance(self):
        """A uniform-colored lesion should have near-zero color variance."""
        img = np.full((256, 256, 3), fill_value=150, dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 80, 1, -1)
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            variance = analyzer.color_space_analysis(analyzer.original_img, analyzer.mask)
            assert variance < 10, f"Uniform color variance should be ~0, got {variance}"
        finally:
            _cleanup(img_path, mask_path)

    def test_multicolor_has_high_variance(self):
        """A lesion with multiple distinct colors should have high variance."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 80, 1, -1)
        # Paint top half red, bottom half blue
        img[:128, :] = [0, 0, 255]   # Red in BGR
        img[128:, :] = [255, 0, 0]   # Blue in BGR
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            variance = analyzer.color_space_analysis(analyzer.original_img, analyzer.mask)
            assert variance > 100, f"Multi-color variance should be high, got {variance}"
        finally:
            _cleanup(img_path, mask_path)


# ---------------------------------------------------------------------------
# Full analyze() integration test
# ---------------------------------------------------------------------------

class TestAnalyzeIntegration:
    def test_analyze_returns_all_keys(self):
        """analyze() should return all expected ABCD keys and Raw_Metrics."""
        img = np.zeros((256, 256, 3), dtype=np.uint8) + 128
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 60, 1, -1)
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            results = analyzer.analyze(show=False)
            for key in ["Asymmetry", "Border", "Diameter", "Colour"]:
                assert key in results, f"Missing key: {key}"
            assert "Raw_Metrics" in results
            raw = results["Raw_Metrics"]
            for rk in ["Area_pixels", "Asymmetry_0_1", "Border_CircularityIndex", "Diameter_Feret_pixels"]:
                assert rk in raw, f"Missing raw metric: {rk}"
        finally:
            _cleanup(img_path, mask_path)

    def test_scaled_scores_are_consistent_with_raw(self):
        """Scaled scores should match raw values * scaling constants."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :128] = [100, 50, 200]
        img[:, 128:] = [200, 100, 50]
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 60, 1, -1)
        img_path, mask_path = _make_temp_images(img, mask)
        try:
            analyzer = MoleAnalyzer(img_path, mask_path)
            results = analyzer.analyze(show=False)
            raw = results["Raw_Metrics"]

            # Asymmetry: scaled = raw * 10
            if raw["Asymmetry_0_1"] is not None:
                expected_asym = raw["Asymmetry_0_1"] * MoleAnalyzer.ASYMMETRY_SCALE
                assert abs(results["Asymmetry"] - expected_asym) < 0.01

            # Border: scaled = raw / 10
            expected_border = raw["Border_CircularityIndex"] / MoleAnalyzer.BORDER_SCALE
            assert abs(results["Border"] - expected_border) < 0.01

            # Diameter: scaled = raw / 10
            expected_diam = raw["Diameter_Feret_pixels"] / MoleAnalyzer.DIAMETER_SCALE
            assert abs(results["Diameter"] - expected_diam) < 0.01
        finally:
            _cleanup(img_path, mask_path)


# ---------------------------------------------------------------------------
# Shared helpers tests
# ---------------------------------------------------------------------------

class TestSharedHelpers:
    def test_percent_change_basic(self):
        from constants import percent_change
        assert percent_change(10, 15) == pytest.approx(50.0)
        assert percent_change(100, 50) == pytest.approx(-50.0)

    def test_percent_change_from_zero(self):
        from constants import percent_change
        assert percent_change(0, 10) is None

    def test_percent_change_none_values(self):
        from constants import percent_change
        assert percent_change(None, 10) is None
        assert percent_change(10, None) is None

    def test_safe_get_metric(self):
        from constants import safe_get_metric
        assert safe_get_metric({"Asymmetry": 5.2}, "Asymmetry") == pytest.approx(5.2)
        assert safe_get_metric({"Asymmetry": 5.2}, "Missing") is None
        assert safe_get_metric({"val": "not_a_number"}, "val") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
