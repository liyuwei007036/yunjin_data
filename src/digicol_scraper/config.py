"""
Configuration module for digicol-scraper.

集中管理所有可配置的参数，包括 API 地址、代理设置、下载参数等。
"""

import os
from pathlib import Path

# ==================== Base Configuration ====================

BASE_URL = "https://digicol.dpm.org.cn"
QUERY_LIST_URL = f"{BASE_URL}/cultural/queryList"
DETAIL_URL = f"{BASE_URL}/cultural/detail"
DETAILS_URL = f"{BASE_URL}/cultural/details"
IMAGE_CDN = "https://shuziwenwu-1259446244.cos.ap-beijing.myqcloud.com"

# ==================== HTTP Headers ====================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Content-Type": "application/json",
    "Referer": BASE_URL,
    "Origin": BASE_URL,
}

# ==================== Category Configuration ====================

CATEGORY_ID = "4"
PAGE_SIZE = 32

# ==================== Download Configuration ====================

MAX_WORKERS = 5
REQUEST_DELAY = 1.0
TIMEOUT = 30

# ==================== Output Configuration ====================

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"

# ==================== Tile Configuration ====================

TILE_SIZE = 510
TILE_EXTENSION = "png"
DOWNLOAD_ONLY_HIGHEST_LEVEL = True

# ==================== Proxy Configuration ====================

PROXY_ENABLED = False
PROXY_TYPE = "socks5"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 9567
PROXY_USER = None
PROXY_PASSWORD = None

# ==================== Progress Configuration ====================

PROGRESS_UPDATE_INTERVAL = 10
