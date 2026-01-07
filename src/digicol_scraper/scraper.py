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
import time
from typing import Dict, List

import tempfile
import shutil

from .api_client import ApiClient
from .tile_fetcher import TileFetcher
from .downloader import TileDownloader
from .tile_merger import TileMerger
from .config import OUTPUT_DIR, IMAGES_DIR, DOWNLOAD_ONLY_HIGHEST_LEVEL


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
        os.makedirs(str(IMAGES_DIR), exist_ok=True)

    def _get_image_path(self, artifact: Dict) -> str:
        """
        Get artifact output image path.

        Args:
            artifact: Artifact info dictionary

        Returns:
            Full path to artifact output image
        """
        name = artifact.get("name", "unknown")
        safe_name = "".join(
            c for c in name if c.isalnum() or c in (" ", "-", "_")
        )[:50]
        return os.path.join(str(IMAGES_DIR), f"{safe_name}.png")

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

        output_path = self._get_image_path(artifact)

        # Skip if already processed
        if os.path.exists(output_path):
            return True

        print(f"  Processing: {name}")
        print(f"  [DEBUG] UUID: {uuid}")

        config = self.tile_fetcher.get_tile_config(uuid)
        if not config:
            print(f"  [ERROR] Failed to get tile config for {name}")
            return False

        if not config.get("levels"):
            print(f"  [ERROR] No levels in config for {name}")
            return False

        print(f"  [DEBUG] Config: {config['width']}x{config['height']}, maxLevel={config['maxLevel']}, tilesUrl={config['tilesUrl'][:50]}..." if config.get('tilesUrl') else f"  [DEBUG] Config: {config['width']}x{config['height']}, maxLevel={config['maxLevel']}, tilesUrl=EMPTY")

        # Create temp directory for tiles
        temp_tile_dir = tempfile.mkdtemp()
        try:
            # Download tiles
            downloader = TileDownloader(temp_tile_dir)
            max_level = config.get("maxLevel", 13)
            tiles_downloaded = 0
            total_tiles = 0
            for level in config["levels"]:
                if DOWNLOAD_ONLY_HIGHEST_LEVEL and level["level"] != max_level:
                    continue
                urls = self.tile_fetcher.generate_tile_urls(
                    config["tilesUrl"], level["level"], level["rows"], level["cols"]
                )
                if urls:
                    total_tiles = len(urls)
                    print(f"  [DEBUG] Downloading {total_tiles} tiles for level {level['level']}")
                    successful = downloader.download_tiles(urls, level["level"])
                    tiles_downloaded = len(successful)
                    print(f"  [DEBUG] Downloaded {tiles_downloaded}/{total_tiles} tiles")

            # Merge tiles
            merger = TileMerger(temp_tile_dir, tile_size=config.get("tileSize", 510))
            highest_level = merger.find_highest_level()
            print(f"  [DEBUG] Highest available level: {highest_level}")
            if highest_level >= 0:
                success = merger.merge_highest_level(output_path)
                if success:
                    print(f"  [SUCCESS] Saved to: {output_path}")
                return success
            else:
                print(f"  [ERROR] No tiles found to merge")
        finally:
            # Clean up temp tile directory
            shutil.rmtree(temp_tile_dir, ignore_errors=True)
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
