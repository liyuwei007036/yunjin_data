#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for digicol-scraper
"""

from setuptools import setup, find_packages

setup(
    name="digicol-scraper",
    version="0.1.0",
    description="故宫数字文物库织绣类文物 DeepZoom 瓦片图下载工具",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.2",
        "pillow>=10.0.1",
        "playwright>=1.40.0",
        "tqdm>=4.66.1",
    ],
    entry_points={
        "console_scripts": [
            "digicol-scraper=digicol_scraper.scraper:main",
        ],
    },
)

