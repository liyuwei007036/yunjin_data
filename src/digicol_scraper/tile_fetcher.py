"""
Tile Fetcher module for digicol-scraper.

功能：使用 Playwright 浏览器自动化获取文物的 DeepZoom 瓦片配置信息。
"""

import re
from typing import Dict, List, Optional
import requests
from playwright.sync_api import sync_playwright
from .config import (
    DETAIL_URL,
    DETAILS_URL,
    HEADERS,
    PROXY_ENABLED,
    PROXY_TYPE,
    PROXY_HOST,
    PROXY_PORT,
    PROXY_USER,
    PROXY_PASSWORD,
)


class TileFetcher:
    """
    Tile Configuration Fetcher class.

    负责使用 Playwright 动态解析文物详情页，提取 DeepZoom 瓦片配置信息。
    """

    def __init__(self) -> None:
        """Initialize tile fetcher."""
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._setup_proxy()

    def _get_proxy(self) -> Optional[dict]:
        """Get requests proxy configuration."""
        if not PROXY_ENABLED:
            return None
        proxy_url = f"{PROXY_TYPE}://{PROXY_HOST}:{PROXY_PORT}"
        return {"http": proxy_url, "https": proxy_url}

    def _get_playwright_proxy(self) -> Optional[dict]:
        """Get Playwright browser proxy configuration."""
        if not PROXY_ENABLED:
            return None
        return {"server": f"{PROXY_TYPE}://{PROXY_HOST}:{PROXY_PORT}"}

    def _setup_proxy(self) -> None:
        """Configure session proxy settings."""
        proxy = self._get_proxy()
        if proxy:
            self.session.proxies = proxy

    def get_tile_config(self, uuid: str) -> Optional[Dict]:
        """
        Get tile configuration for specified artifact.

        Args:
            uuid: Artifact UUID

        Returns:
            Dictionary containing tile configuration:
            {
                'artifact_uuid': uuid,
                'width': image width,
                'height': image height,
                'minLevel': min level,
                'maxLevel': max level,
                'tileSize': tile size,
                'tilesUrl': tile base URL,
                'levels': [...]  # Level details list
            }
            Returns None on failure
        """
        url = f"{DETAIL_URL}?id={uuid}"
        captured_url = [None]

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, proxy=self._get_playwright_proxy())
            page = browser.new_page()

            def handle_response(response):
                if "/cultural/details" in response.url:
                    captured_url[0] = response.url

            page.on("response", handle_response)

            try:
                # Step 1: Visit detail page to get image_id
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(1000)

                # Click button to open large image
                page.evaluate(
                    """
                    () => {
                        const elements = document.querySelectorAll('a');
                        for (const el of elements) {
                            if (el.onclick && el.onclick.toString().includes('showBigImage')) {
                                el.click();
                                break;
                            }
                        }
                    }
                """
                )
                page.wait_for_timeout(2000)

                # Extract image_id from captured URL
                image_id = None
                if captured_url[0]:
                    print(f"  [DEBUG] Captured URL: {captured_url[0]}")
                    match = re.search(r"\?id=(\d+)", captured_url[0])
                    if match:
                        image_id = match.group(1)
                        print(f"  [DEBUG] Extracted image_id from response: {image_id}")

                # Fallback: extract from page scripts
                if not image_id:
                    print(f"  [DEBUG] Trying to extract image_id from page scripts...")
                    image_id = page.evaluate(
                        r"""
                        () => {
                            const scripts = document.querySelectorAll('script');
                            for (const script of scripts) {
                                const text = script.textContent;
                                const match = text.match(/details\?id=(\d+)/);
                                if (match) return match[1];
                            }
                            return null;
                        }
                    """
                    )
                    if image_id:
                        print(f"  [DEBUG] Extracted image_id from scripts: {image_id}")

                if not image_id:
                    print(f"  [ERROR] Could not find image_id for uuid={uuid}")
                    return None

                # Step 2: Visit details config page to get tile configuration
                details_url = f"{DETAILS_URL}?id={image_id}"
                print(f"  [DEBUG] Fetching details config from: {details_url}")
                page.goto(details_url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(1500)

                # Check if page loaded correctly
                page_content = page.content()
                print(f"  [DEBUG] Page content length: {len(page_content)} chars")

                config = page.evaluate(
                    """() => {
                    try {
                        const viewer = window.viewer;
                        if (!viewer) {
                            console.log('window.viewer is undefined');
                            return null;
                        }
                        if (!viewer.source) {
                            console.log('viewer.source is undefined');
                            return null;
                        }
                        const s = viewer.source;
                        return {
                            width: s.width,
                            height: s.height,
                            minLevel: s.minLevel || 0,
                            maxLevel: s.maxLevel || 13,
                            tilesUrl: s.tilesUrl || "",
                            tileSize: s._tileWidth || s.tileSize || 510
                        };
                    } catch(e) {
                        console.log('Error:', e.message);
                        return null;
                    }
                }"""
                )

                if not config:
                    print(f"  [ERROR] Failed to extract config from details page")
                    return None

                print(f"  [DEBUG] Config extracted: width={config['width']}, height={config['height']}, maxLevel={config['maxLevel']}")
                print(f"  [DEBUG] tilesUrl: {config['tilesUrl'][:80] if config['tilesUrl'] else 'EMPTY'}...")

                # Build level information
                width = config["width"]
                height = config["height"]
                min_level = config["minLevel"]
                max_level = config["maxLevel"]
                tile_size = config["tileSize"]
                tiles_url = config["tilesUrl"]

                levels = []
                for level in range(min_level, max_level + 1):
                    level_width = width >> (max_level - level)
                    level_height = height >> (max_level - level)
                    num_cols = (level_width + tile_size - 1) // tile_size
                    num_rows = (level_height + tile_size - 1) // tile_size
                    levels.append(
                        {
                            "level": level,
                            "width": level_width,
                            "height": level_height,
                            "rows": list(range(num_rows)),
                            "cols": list(range(num_cols)),
                            "total_tiles": num_cols * num_rows,
                        }
                    )

                return {
                    "artifact_uuid": uuid,
                    "width": width,
                    "height": height,
                    "minLevel": min_level,
                    "maxLevel": max_level,
                    "tileSize": tile_size,
                    "tilesUrl": tiles_url,
                    "levels": levels,
                }

            except Exception as e:
                print(f"  [ERROR] Playwright exception: {type(e).__name__}: {e}")
                import traceback
                print(f"  [ERROR] {traceback.format_exc()}")
                return None

    def generate_tile_urls(
        self, tiles_url: str, level: int, rows: List[int], cols: List[int]
    ) -> List[str]:
        """
        Generate tile download URL list for specified level.

        Args:
            tiles_url: Tile base URL
            level: Tile level
            rows: Row number list
            cols: Column number list

        Returns:
            List of tile download URLs
        """
        return [f"{tiles_url}{level}/{col}_{row}.png" for col in cols for row in rows]


def main() -> None:
    """Main function for testing tile fetcher functionality."""
    fetcher = TileFetcher()
    test_uuid = "6ee002c4f2fa4adfbd66af2bfdd60788"
    config = fetcher.get_tile_config(test_uuid)
    if config:
        print(
            f"Config: {config['width']}x{config['height']}, "
            f"{len(config['levels'])} levels"
        )
    else:
        print("Failed to get tile config")


if __name__ == "__main__":
    main()
