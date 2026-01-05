"""
Tests for API client module.
"""

import pytest
from digicol_scraper.api_client import ApiClient


class TestApiClient:
    """Test cases for ApiClient class."""

    def test_init(self):
        """Test API client initialization."""
        client = ApiClient()
        assert client.session is not None

    def test_get_proxy_disabled(self, monkeypatch):
        """Test proxy is None when disabled."""
        import digicol_scraper.config as config

        monkeypatch.setattr(config, "PROXY_ENABLED", False)
        client = ApiClient()
        assert client._get_proxy() is None
