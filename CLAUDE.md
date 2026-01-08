# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

故宫数字文物数据处理项目，包含两个主要模块：

1. **digicol-scraper**: 故宫数字文物库爬虫，下载 DeepZoom 瓦片图并拼接为完整图片
2. **data-cleaner**: 基于 VLM + Grounded SAM 的图案提取工具，用于从图片中提取图案用于模型微调

## Common Commands

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 Playwright 浏览器
playwright install chromium

# 进入 src 目录（必须）
cd src

# 爬虫 - 测试模式（抓取 10 个文物）
python -m digicol_scraper.scraper --mode test --limit 10

# 爬虫 - 完整模式（处理所有文物）
python -m digicol_scraper.scraper --mode full

# 爬虫 - 仅下载模式
python -m digicol_scraper.scraper --mode download-only --uuid <uuid>

# 数据清洗 - 默认使用 config.yaml
python -m data_cleaner.run

# 数据清洗 - 指定配置文件
python -m data_cleaner.run --config my_config.yaml

# 数据清洗 - 覆盖输入输出路径
python -m data_cleaner.run --input output/images --output output/patterns
```

**注意**: 必须从 `src` 目录运行，或设置 `PYTHONPATH=src` 环境变量。

## Architecture

### digicol-scraper 模块

```
scraper.py (主调度器)
    ├── ApiClient (API 客户端，获取文物列表)
    ├── TileFetcher (使用 Playwright 解析 DeepZoom 配置)
    ├── TileDownloader (多线程瓦片下载器，支持断点续传)
    └── TileMerger (瓦片拼接器)
配置: config.yaml → config.py 常量
```

### data-cleaner 模块

```
run.py (CLI 入口)
    └── PatternExtractor (图案提取主类)
        ├── ImageAnalyzer (图片类型分析)
        │   └── VLMRecognizer (Qwen2-VL 视觉大模型识别物体类别)
        └── strategies/GroundedSAMStrategy (Grounding DINO + SAM 分割)
            └── TrainingDataExporter (COCO 格式标注导出)
配置: config.yaml → Config 数据类
```

### 数据流程

```
digicol-scraper: API → 瓦片配置 → 并行下载 → 拼接 → output/images/

data-cleaner: input_dir → VLM识别类别 → Grounding DINO检测 → SAM分割 → 去重/质量筛选 → output/patterns/
```

## Key Configuration Files

- `src/digicol_scraper/config.yaml`: 爬虫配置（代理、并发、API 地址）
- `src/data_cleaner/config.yaml`: 图案提取配置（模型路径、阈值、尺寸）

## Dependencies Notes

- **Grounded SAM**: 需要先安装 CUDA 版本的 PyTorch
- **VLM 模型**: 默认使用 Qwen/Qwen2-VL-7B-Instruct
- **模型文件**: 需手动下载至 `models/` 目录
