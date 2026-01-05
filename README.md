# 故宫数字文物库爬虫

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

故宫数字文物库织绣类文物 DeepZoom 瓦片图下载工具。

## 功能特性

- API 客户端：获取文物列表数据
- 瓦片配置获取：使用 Playwright 动态解析 DeepZoom 配置
- 多线程下载：并行下载瓦片图，支持断点续传
- 瓦片拼接：使用 Pillow 将瓦片拼接为完整图片

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 Playwright 浏览器
playwright install chromium
```

## 使用方法

```bash
# 测试模式：抓取 10 个文物
python -m digicol_scraper.scraper --mode test --limit 10

# 完整模式：处理所有文物
python -m digicol_scraper.scraper --mode full

# 仅下载模式：下载指定文物
python -m digicol_scraper.scraper --mode download-only --uuid <uuid>

# 仅拼接模式：拼接已下载的瓦片
python -m digicol_scraper.scraper --mode merge-only --uuid <uuid>
```

## 项目结构

```
digicol_scraper/
├── __init__.py         # 包初始化
├── api_client.py       # API 客户端
├── config.py           # 配置文件
├── scraper.py          # 主入口
├── downloader.py       # 瓦片下载器
├── tile_fetcher.py     # 瓦片配置获取
├── tile_merger.py      # 瓦片拼接器
├── requirements.txt    # 依赖列表
└── output/             # 输出目录
```

## 配置说明

所有配置项都在 `config.py` 中修改：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| PROXY_ENABLED | 是否启用代理 | True |
| PROXY_TYPE | 代理类型 | socks5 |
| PROXY_HOST | 代理服务器地址 | 127.0.0.1 |
| PROXY_PORT | 代理服务器端口 | 9567 |
| DOWNLOAD_ONLY_HIGHEST_LEVEL | 仅下载最高分辨率 | True |
| MAX_WORKERS | 最大并发线程数 | 5 |

## 依赖

- requests >= 2.31.0
- beautifulsoup4 >= 4.12.2
- pillow >= 10.0.1
- playwright >= 1.40.0
- tqdm >= 4.66.1
