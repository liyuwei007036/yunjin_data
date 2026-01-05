"""
API Client module for digicol-scraper.

功能：负责与故宫数字文物库 API 交互，获取文物列表数据。
"""

import json
import time
from typing import Dict, List, Optional
import requests
from tqdm import tqdm
from .config import (
    QUERY_LIST_URL,
    HEADERS,
    CATEGORY_ID,
    PAGE_SIZE,
    REQUEST_DELAY,
    TIMEOUT,
    PROXY_ENABLED,
    PROXY_TYPE,
    PROXY_HOST,
    PROXY_PORT,
)


class ApiClient:
    """
    API Client class.

    负责发送 HTTP 请求获取文物列表数据，支持分页查询和代理设置。
    """

    def __init__(self) -> None:
        """Initialize API client with session and proxy configuration."""
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._setup_proxy()

    def _get_proxy(self) -> Optional[dict]:
        """Get proxy configuration dictionary."""
        if not PROXY_ENABLED:
            return None
        return {
            "http": f"{PROXY_TYPE}://{PROXY_HOST}:{PROXY_PORT}",
            "https": f"{PROXY_TYPE}://{PROXY_HOST}:{PROXY_PORT}",
        }

    def _setup_proxy(self) -> None:
        """Configure session proxy settings."""
        proxy = self._get_proxy()
        if proxy:
            self.session.proxies = proxy

    def _make_request(
        self, url: str, data: dict, max_retries: int = 3
    ) -> Optional[dict]:
        """
        Send HTTP POST request with retry mechanism.

        Args:
            url: Request URL
            data: POST request data
            max_retries: Maximum retry attempts

        Returns:
            Server response JSON data, or None on failure
        """
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, json=data, timeout=TIMEOUT)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
        return None

    def get_artifacts_page(self, page: int) -> Optional[Dict]:
        """
        Get artifact list for specified page.

        Args:
            page: Page number (1-indexed)

        Returns:
            Response data containing artifact list, or None on failure
        """
        return self._make_request(
            QUERY_LIST_URL, {"page": page, "categoryList": [CATEGORY_ID]}
        )

    def get_all_artifacts(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all artifacts list.

        Args:
            limit: Optional limit on number of artifacts to return

        Returns:
            List of artifacts, each as a dictionary
        """
        first = self.get_artifacts_page(1)
        if not first:
            return []

        total_pages = first.get("pagecount", 0)
        artifacts = first.get("rows", [])

        if limit:
            if len(artifacts) >= limit:
                artifacts = artifacts[:limit]
            else:
                for page in tqdm(range(2, total_pages + 1), desc="Fetching pages"):
                    if len(artifacts) >= limit:
                        break
                    data = self.get_artifacts_page(page)
                    if data and data.get("rows"):
                        artifacts.extend(data["rows"])
                    time.sleep(REQUEST_DELAY)
                artifacts = artifacts[:limit]
        else:
            for page in tqdm(range(2, total_pages + 1), desc="Fetching pages"):
                data = self.get_artifacts_page(page)
                if data and data.get("rows"):
                    artifacts.extend(data["rows"])
                time.sleep(REQUEST_DELAY)

        print(f"Fetched {len(artifacts)} artifacts")
        return artifacts

    def save_artifacts(self, artifacts: List[Dict], filepath: str) -> None:
        """
        Save artifacts list to JSON file.

        Args:
            artifacts: List of artifacts
            filepath: Output file path
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, ensure_ascii=False, indent=2)


def main() -> None:
    """Main function for testing API client functionality."""
    client = ApiClient()
    artifacts = client.get_all_artifacts(limit=10)
    client.save_artifacts(artifacts, "test_artifacts.json")


if __name__ == "__main__":
    main()
