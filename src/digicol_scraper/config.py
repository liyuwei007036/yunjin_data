"""
Configuration module for digicol-scraper.

集中管理所有可配置的参数，包括 API 地址、代理设置、下载参数等。
配置从 config.yaml 文件加载。
"""

from pathlib import Path
import yaml

# ==================== Load Configuration from YAML ====================

_CONFIG_FILE = Path(__file__).parent / "config.yaml"

def _load_config():
    """Load configuration from YAML file."""
    if not _CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {_CONFIG_FILE}. "
            "Please ensure config.yaml exists in the digicol_scraper directory."
        )
    
    with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config

_config = _load_config()

# ==================== Base Configuration ====================

BASE_URL = _config["base_url"]
QUERY_LIST_URL = _config["query_list_url"]
DETAIL_URL = _config["detail_url"]
DETAILS_URL = _config["details_url"]
IMAGE_CDN = _config["image_cdn"]

# ==================== HTTP Headers ====================

HEADERS = _config["headers"].copy()
# Ensure Referer and Origin use BASE_URL dynamically
HEADERS["Referer"] = BASE_URL
HEADERS["Origin"] = BASE_URL

# ==================== Category Configuration ====================

CATEGORY_ID = _config["category_id"]
PAGE_SIZE = _config["page_size"]

# ==================== Download Configuration ====================

MAX_WORKERS = _config["max_workers"]
REQUEST_DELAY = _config["request_delay"]
TIMEOUT = _config["timeout"]

# ==================== Output Configuration ====================

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"

# ==================== Tile Configuration ====================

TILE_SIZE = _config["tile_size"]
TILE_EXTENSION = _config["tile_extension"]
DOWNLOAD_ONLY_HIGHEST_LEVEL = _config["download_only_highest_level"]

# ==================== Proxy Configuration ====================

PROXY_ENABLED = _config["proxy_enabled"]
PROXY_TYPE = _config["proxy_type"]
PROXY_HOST = _config["proxy_host"]
PROXY_PORT = _config["proxy_port"]
PROXY_USER = _config["proxy_user"]
PROXY_PASSWORD = _config["proxy_password"]

# ==================== Progress Configuration ====================

PROGRESS_UPDATE_INTERVAL = _config["progress_update_interval"]
