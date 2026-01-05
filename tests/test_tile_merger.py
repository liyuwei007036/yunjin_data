"""
Tests for tile merger module.
"""

import pytest
import os
import tempfile
from PIL import Image
from digicol_scraper.tile_merger import TileMerger


class TestTileMerger:
    """Test cases for TileMerger class."""

    def test_init(self, tmp_path):
        """Test tile merger initialization."""
        merger = TileMerger(str(tmp_path))
        assert merger.tile_dir == str(tmp_path)
        assert merger.tile_size == 510

    def test_init_custom_tile_size(self, tmp_path):
        """Test initialization with custom tile size."""
        merger = TileMerger(str(tmp_path), tile_size=256)
        assert merger.tile_size == 256

    def test_get_level_dir(self, tmp_path):
        """Test level directory path generation."""
        merger = TileMerger(str(tmp_path))
        level_dir = merger._get_level_dir(5)
        assert level_dir == str(tmp_path / "level_5")

    def test_find_highest_level_no_levels(self, tmp_path):
        """Test finding highest level when no levels exist."""
        merger = TileMerger(str(tmp_path))
        assert merger.find_highest_level() == -1

    def test_find_highest_level_with_levels(self, tmp_path):
        """Test finding highest level with existing levels."""
        # Create level directories
        (tmp_path / "level_3").mkdir()
        (tmp_path / "level_5").mkdir()
        (tmp_path / "level_7").mkdir()

        merger = TileMerger(str(tmp_path))
        assert merger.find_highest_level() == 7

    def test_get_tiles_for_level_empty(self, tmp_path):
        """Test getting tiles for non-existent level."""
        merger = TileMerger(str(tmp_path))
        tiles = merger.get_tiles_for_level(5)
        assert tiles == {}

    def test_get_tiles_for_level_with_files(self, tmp_path):
        """Test getting tiles with existing tile files."""
        level_dir = tmp_path / "level_5"
        level_dir.mkdir()

        # Create test tile files
        (level_dir / "0_0.png").touch()
        (level_dir / "1_0.png").touch()
        (level_dir / "0_1.png").touch()

        merger = TileMerger(str(tmp_path))
        tiles = merger.get_tiles_for_level(5)

        assert len(tiles) == 3
        assert (0, 0) in tiles
        assert (1, 0) in tiles
        assert (0, 1) in tiles
