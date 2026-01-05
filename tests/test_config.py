"""
Tests for config module.
"""

import pytest
from pathlib import Path


class TestConfig:
    """Test cases for config module."""

    def test_base_url_exists(self):
        """Test BASE_URL is configured."""
        from digicol_scraper.config import BASE_URL

        assert BASE_URL is not None
        assert "digicol.dpm.org.cn" in BASE_URL

    def test_output_dir_is_path(self):
        """Test OUTPUT_DIR is a Path object."""
        from digicol_scraper.config import OUTPUT_DIR

        assert isinstance(OUTPUT_DIR, Path)

    def test_tile_size_positive(self):
        """Test TILE_SIZE is positive integer."""
        from digicol_scraper.config import TILE_SIZE

        assert isinstance(TILE_SIZE, int)
        assert TILE_SIZE > 0

    def test_headers_contains_required_keys(self):
        """Test HEADERS contains required keys."""
        from digicol_scraper.config import HEADERS

        assert "User-Agent" in HEADERS
        assert "Accept" in HEADERS
        assert "Content-Type" in HEADERS
