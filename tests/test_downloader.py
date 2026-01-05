"""
Tests for tile downloader module.
"""

import pytest
import os
import tempfile
from digicol_scraper.downloader import TileDownloader


class TestTileDownloader:
    """Test cases for TileDownloader class."""

    def test_init(self, tmp_path):
        """Test downloader initialization."""
        downloader = TileDownloader(str(tmp_path))
        assert downloader.output_dir == str(tmp_path)
        assert downloader.session is not None

    def test_get_tile_path(self, tmp_path):
        """Test tile path generation."""
        downloader = TileDownloader(str(tmp_path))
        path = downloader._get_tile_path(
            "https://example.com/tiles/5/0_1.png", 5
        )
        assert "level_5" in path
        assert path.endswith("0_1.png")

    def test_is_downloaded_false_for_nonexistent(self, tmp_path):
        """Test is_downloaded returns False for non-existent file."""
        downloader = TileDownloader(str(tmp_path))
        assert downloader._is_downloaded(str(tmp_path / "nonexistent.png")) is False

    def test_is_downloaded_true_for_existing_file(self, tmp_path):
        """Test is_downloaded returns True for existing file."""
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"\x00" * 100)

        downloader = TileDownloader(str(tmp_path))
        assert downloader._is_downloaded(str(test_file)) is True

    def test_is_downloaded_false_for_empty_file(self, tmp_path):
        """Test is_downloaded returns False for empty file."""
        test_file = tmp_path / "empty.png"
        test_file.write_bytes(b"")

        downloader = TileDownloader(str(tmp_path))
        assert downloader._is_downloaded(str(test_file)) is False

    def test_verify_download_empty(self, tmp_path):
        """Test verify download with no level directories."""
        downloader = TileDownloader(str(tmp_path))
        assert downloader.verify_download(0) is True

    def test_verify_download_mismatch(self, tmp_path):
        """Test verify download when count doesn't match."""
        level_dir = tmp_path / "level_5"
        level_dir.mkdir()
        (level_dir / "0_0.png").touch()
        (level_dir / "0_1.png").touch()

        downloader = TileDownloader(str(tmp_path))
        assert downloader.verify_download(5) is False
