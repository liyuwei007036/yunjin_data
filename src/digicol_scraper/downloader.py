"""
Tile Downloader module for digicol-scraper.

功能：多线程并行下载 DeepZoom 瓦片图，支持断点续传。
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set
import requests
from tqdm import tqdm
from .config import (
    HEADERS,
    MAX_WORKERS,
    TIMEOUT,
    PROXY_ENABLED,
    PROXY_TYPE,
    PROXY_HOST,
    PROXY_PORT,
)


class TileDownloader:
    """
    Tile Downloader class.

    使用多线程并行下载瓦片图，支持断点续传（已下载的瓦片不会重复下载）。
    """

    def __init__(self, output_dir: str, max_workers: int = MAX_WORKERS) -> None:
        """
        Initialize tile downloader.

        Args:
            output_dir: Tile output directory
            max_workers: Maximum concurrent threads
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._setup_proxy()
        os.makedirs(output_dir, exist_ok=True)

    def _get_proxy(self) -> dict:
        """Get proxy configuration."""
        if not PROXY_ENABLED:
            return {}
        return {
            "http": f"{PROXY_TYPE}://{PROXY_HOST}:{PROXY_PORT}",
            "https": f"{PROXY_TYPE}://{PROXY_HOST}:{PROXY_PORT}",
        }

    def _setup_proxy(self) -> None:
        """Configure session proxy settings."""
        self.session.proxies = self._get_proxy()

    def _get_tile_path(self, url: str, level: int) -> str:
        """
        Get tile save path.

        Args:
            url: Tile download URL
            level: Tile level

        Returns:
            Tile local save path
        """
        return os.path.join(self.output_dir, f"level_{level}", url.split("/")[-1])

    def _is_downloaded(self, path: str) -> bool:
        """
        Check if tile is already downloaded.

        Args:
            path: Tile file path

        Returns:
            True if downloaded and file is non-empty, False otherwise
        """
        return os.path.exists(path) and os.path.getsize(path) > 0

    def download_tile(self, url: str, level: int) -> bool:
        """
        Download single tile.

        Args:
            url: Tile download URL
            level: Tile level

        Returns:
            True on success, False on failure
        """
        path = self._get_tile_path(url, level)
        if self._is_downloaded(path):
            return True

        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            response = self.session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            return True
        except requests.exceptions.RequestException:
            return False

    def download_tiles(
        self, urls: List[str], level: int, desc: str = "Downloading"
    ) -> Set[str]:
        """
        Batch download tiles (multi-threaded).

        Args:
            urls: List of tile URLs
            level: Tile level
            desc: Progress bar description

        Returns:
            Set of successfully downloaded URLs
        """
        successful = set()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_tile, url, level): url for url in urls
            }
            for future in tqdm(as_completed(futures), total=len(urls), desc=desc):
                if future.result():
                    successful.add(futures[future])
        return successful

    def verify_download(self, expected_count: int) -> bool:
        """
        Verify download completeness.

        Args:
            expected_count: Expected tile count

        Returns:
            True if count matches, False otherwise
        """
        count = sum(
            len(os.listdir(os.path.join(self.output_dir, d)))
            for d in os.listdir(self.output_dir)
            if d.startswith("level_")
        )
        return count == expected_count


def main() -> None:
    """Main function (shows message when no test data)."""
    print("No test URLs provided")


if __name__ == "__main__":
    main()
