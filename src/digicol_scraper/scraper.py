#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scraper Main Entry for digicol-scraper.

功能：爬取故宫数字文物库织绣类文物的 DeepZoom 瓦片图，并拼接为完整图片。

用法：
    python -m digicol_scraper.scraper --mode test --limit 10
    python -m digicol_scraper.scraper --mode full
    python -m digicol_scraper.scraper --mode download-only --uuid <uuid>
    python -m digicol_scraper.scraper --mode merge-only --uuid <uuid>
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

from .api_client import ApiClient
from .tile_fetcher import TileFetcher
from .downloader import TileDownloader
from .tile_merger import TileMerger
from .config import OUTPUT_DIR, DOWNLOAD_ONLY_HIGHEST_LEVEL

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Scraper:
    """Scraper Scheduler class - coordinates modules to complete artifact scraping tasks."""

    def __init__(self, output_dir: str = str(OUTPUT_DIR)) -> None:
        """
        Initialize scraper scheduler.

        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        self.api_client = ApiClient()
        self.tile_fetcher = TileFetcher()
        os.makedirs(output_dir, exist_ok=True)

    def _get_artifact_dir(self, artifact: Dict) -> str:
        """
        Get artifact output directory path.

        Args:
            artifact: Artifact info dictionary

        Returns:
            Full path to artifact output directory
        """
        name = artifact.get("name", "unknown")
        relic_no = artifact.get("culturalRelicNo", "unknown")
        safe_name = "".join(
            c for c in name if c.isalnum() or c in (" ", "-", "_")
        )[:50]
        return os.path.join(self.output_dir, f"{safe_name}_{relic_no}")

    def _load_artifacts(self) -> List[Dict]:
        """
        Load artifact list from local cache.

        Returns:
            Artifact list dictionary
        """
        path = os.path.join(self.output_dir, "artifacts.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_artifacts(self, artifacts: List[Dict]) -> None:
        """
        Save artifact list to local cache.

        Args:
            artifacts: Artifact list
        """
        with open(
            os.path.join(self.output_dir, "artifacts.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(artifacts, f, ensure_ascii=False, indent=2)

    def _process_artifact(self, artifact: Dict) -> bool:
        """
        Process single artifact: download and merge tiles.

        Args:
            artifact: Artifact info dictionary

        Returns:
            True on success, False on failure
        """
        uuid = artifact.get("uuid")
        name = artifact.get("name", "Unknown")

        artifact_dir = self._get_artifact_dir(artifact)
        tiles_dir = os.path.join(artifact_dir, "tiles")
        merged_dir = os.path.join(artifact_dir, "merged")

        # Skip if already processed
        if os.path.exists(os.path.join(merged_dir, "full_image.png")):
            return True

        print(f"  Processing: {name}")
        config = self.tile_fetcher.get_tile_config(uuid)
        if not config or not config.get("levels"):
            return False

        # Download tiles
        downloader = TileDownloader(tiles_dir)
        max_level = config.get("maxLevel", 13)
        for level in config["levels"]:
            if DOWNLOAD_ONLY_HIGHEST_LEVEL and level["level"] != max_level:
                continue
            urls = self.tile_fetcher.generate_tile_urls(
                config["tilesUrl"], level["level"], level["rows"], level["cols"]
            )
            if urls:
                downloader.download_tiles(urls, level["level"])

        # Merge tiles
        merger = TileMerger(tiles_dir, tile_size=config.get("tileSize", 510))
        if merger.find_highest_level() >= 0:
            merger.merge_highest_level(os.path.join(merged_dir, "full_image.png"))
            return True
        return False

    def run_test(self, limit: int = 10) -> None:
        """
        Test mode: scrape specified number of artifacts.

        Args:
            limit: Number of artifacts to process (default 10)
        """
        print(f"Running test on {limit} artifacts...")
        artifacts = self.api_client.get_all_artifacts(limit=limit)
        if not artifacts:
            print("Failed to fetch artifacts")
            return

        self._save_artifacts(artifacts)
        for i, artifact in enumerate(artifacts, 1):
            print(f"\n[{i}/{len(artifacts)}]", end=" ")
            self._process_artifact(artifact)
            time.sleep(1)
        print(f"\nDone. Output: {self.output_dir}")

    def run_full(self) -> None:
        """Full mode: process all artifacts."""
        print("Running full crawl...")
        artifacts = self._load_artifacts()
        if not artifacts:
            artifacts = self.api_client.get_all_artifacts()
            if not artifacts:
                print("Failed to fetch artifacts")
                return
            self._save_artifacts(artifacts)

        processed = sum(1 for a in artifacts if self._process_artifact(a))
        print(f"\nProcessed: {processed}/{len(artifacts)}")

    def run_download_only(self, uuid: str) -> None:
        """
        Download only mode: download tiles for specified artifact.

        Args:
            uuid: Artifact UUID
        """
        artifacts = self._load_artifacts()
        artifact = next((a for a in artifacts if a.get("uuid") == uuid), None)
        if artifact:
            self._process_artifact(artifact)
            print("Download complete")
        else:
            print(f"Artifact {uuid} not found")


def main() -> None:
    """Main function - parse CLI arguments and start scraper."""
    parser = argparse.ArgumentParser(description="故宫数字文物库爬虫")
    parser.add_argument(
        "--mode",
        choices=["test", "full", "download-only", "merge-only"],
        default="test",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--uuid")
    args = parser.parse_args()

    scraper = Scraper()
    if args.mode == "test":
        scraper.run_test(args.limit)
    elif args.mode == "full":
        scraper.run_full()
    elif args.mode == "download-only" and args.uuid:
        scraper.run_download_only(args.uuid)


if __name__ == "__main__":
    main()
