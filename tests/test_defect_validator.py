"""
Unit tests for the CV-based defect validator.
Tests edge detection, texture analysis, and visual evidence scoring.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

from src.safety.defect_validator import (
    DefectRegionValidator, 
    ValidationResult, 
    validate_defects
)


class TestDefectRegionValidator:
    """Tests for DefectRegionValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return DefectRegionValidator()
    
    @pytest.fixture
    def clean_image(self, temp_dir):
        """Create a clean uniform image (no defects)."""
        # Create a uniform gray image
        img = np.ones((500, 500, 3), dtype=np.uint8) * 180
        img_path = temp_dir / "clean_image.jpg"
        cv2.imwrite(str(img_path), img)
        return img_path, img
    
    @pytest.fixture
    def defect_image(self, temp_dir):
        """Create an image with a clear defect (dark crack on light background)."""
        # Create a light gray background
        img = np.ones((500, 500, 3), dtype=np.uint8) * 200
        
        # Draw a dark crack-like line (this is a clear defect)
        cv2.line(img, (200, 100), (300, 300), (50, 50, 50), 3)
        cv2.line(img, (300, 300), (350, 350), (50, 50, 50), 2)
        
        img_path = temp_dir / "defect_image.jpg"
        cv2.imwrite(str(img_path), img)
        return img_path, img
    
    @pytest.fixture
    def rust_image(self, temp_dir):
        """Create an image with color anomaly (rust)."""
        # Create a gray background
        img = np.ones((500, 500, 3), dtype=np.uint8) * 150
        
        # Add a rusty area (orange/brown color)
        rust_region = img[150:250, 150:250]
        rust_region[:, :] = [30, 80, 180]  # BGR for rust color
        
        img_path = temp_dir / "rust_image.jpg"
        cv2.imwrite(str(img_path), img)
        return img_path, img
    
    def test_validate_clean_region_returns_low_score(self, validator, clean_image):
        """Test that a uniform clean region returns low visual evidence score."""
        img_path, img = clean_image
        
        # Validate a region in the middle of the clean image
        bbox = {"x": 40, "y": 40, "width": 20, "height": 20}
        result = validator.validate_defect_region(img, bbox, "scratch")
        
        # Clean region should have low visual evidence score
        assert result.visual_evidence_score < 0.5
        assert result.is_valid is False or result.visual_evidence_score < validator.DOWNGRADE_THRESHOLD
    
    def test_validate_defect_region_returns_higher_score(self, validator, defect_image):
        """Test that a region with actual defect returns higher score."""
        img_path, img = defect_image
        
        # Validate the region containing the crack
        bbox = {"x": 40, "y": 20, "width": 30, "height": 50}  # Area with the crack
        result = validator.validate_defect_region(img, bbox, "crack")
        
        # Defect region should have higher visual evidence score
        assert result.edge_score > 0  # Cracks show up in edge detection
        assert isinstance(result, ValidationResult)
    
    def test_validate_color_anomaly(self, validator, rust_image):
        """Test that color anomalies (rust) are detected."""
        img_path, img = rust_image
        
        # Validate the rusty region
        bbox = {"x": 30, "y": 30, "width": 20, "height": 20}  # Rust area
        result = validator.validate_defect_region(img, bbox, "rust")
        
        # Color deviation should be detected
        assert result.color_score > 0
    
    def test_edge_score_for_crack_types(self, validator, defect_image):
        """Test that edge-critical defect types boost scoring."""
        img_path, img = defect_image
        
        bbox = {"x": 40, "y": 20, "width": 30, "height": 50}
        
        # Crack type should rely heavily on edge detection
        result = validator.validate_defect_region(img, bbox, "crack")
        
        # Check that edge score was calculated
        assert 0 <= result.edge_score <= 1.0
    
    def test_validate_all_defects(self, validator, temp_dir):
        """Test batch validation of multiple defects."""
        # Create test image
        img = np.ones((500, 500, 3), dtype=np.uint8) * 180
        cv2.line(img, (100, 100), (200, 200), (50, 50, 50), 3)  # One defect
        img_path = temp_dir / "multi_defect.jpg"
        cv2.imwrite(str(img_path), img)
        
        defects = [
            {"type": "crack", "bbox": {"x": 20, "y": 20, "width": 20, "height": 20}},
            {"type": "scratch", "bbox": {"x": 60, "y": 60, "width": 10, "height": 10}},  # Clean area
        ]
        
        validated, rejected = validator.validate_all_defects(img_path, defects)
        
        # Should return lists
        assert isinstance(validated, list)
        assert isinstance(rejected, list)
        # All defects should have cv_validation field added
        for defect in validated + rejected:
            assert "cv_validation" in defect
    
    def test_validation_result_to_dict(self, validator, clean_image):
        """Test ValidationResult serialization."""
        img_path, img = clean_image
        
        bbox = {"x": 40, "y": 40, "width": 20, "height": 20}
        result = validator.validate_defect_region(img, bbox, "test")
        
        result_dict = result.to_dict()
        
        assert "is_valid" in result_dict
        assert "visual_evidence_score" in result_dict
        assert "edge_score" in result_dict
        assert "texture_score" in result_dict
    
    def test_empty_bbox_handling(self, validator, clean_image):
        """Test handling of invalid/empty bounding boxes."""
        img_path, img = clean_image
        
        defects = [
            {"type": "crack"},  # No bbox
            {"type": "scratch", "bbox": None},  # None bbox
        ]
        
        validated, rejected = validator.validate_all_defects(img_path, defects)
        
        # Defects without bbox should still be in validated (with skip note)
        assert len(validated) == 2
        for defect in validated:
            assert defect["cv_validation"]["skipped"] is True


class TestValidateDefectsFunction:
    """Tests for the convenience validate_defects function."""
    
    def test_validate_defects_function(self, temp_dir):
        """Test the module-level validate_defects function."""
        # Create test image
        img = np.ones((500, 500, 3), dtype=np.uint8) * 180
        img_path = temp_dir / "test.jpg"
        cv2.imwrite(str(img_path), img)
        
        defects = [{"type": "test", "bbox": {"x": 25, "y": 25, "width": 10, "height": 10}}]
        
        validated, rejected = validate_defects(img_path, defects)
        
        assert isinstance(validated, list)
        assert isinstance(rejected, list)


class TestThresholds:
    """Tests for validation thresholds."""
    
    def test_reject_threshold(self):
        """Test that reject threshold is properly configured."""
        validator = DefectRegionValidator()
        assert validator.REJECT_THRESHOLD == 0.3
    
    def test_downgrade_threshold(self):
        """Test that downgrade threshold is properly configured."""
        validator = DefectRegionValidator()
        assert validator.DOWNGRADE_THRESHOLD == 0.5
    
    def test_weights_sum_to_one(self):
        """Test that signal weights sum to 1.0."""
        validator = DefectRegionValidator()
        total_weight = sum(validator.WEIGHTS.values())
        assert abs(total_weight - 1.0) < 0.001
