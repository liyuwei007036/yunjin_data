#!/usr/bin/env python3
"""
云锦图片数据清洗工具 - 启动脚本

使用方法:
    # 方式1: 使用 python -m（推荐）
    python -m src.data_cleaner.run_cleaner

    # 方式2: 安装为开发模式
    pip install -e .
    run-cleaner
"""

import argparse
import sys
from pathlib import Path

from src.data_cleaner.config import load_config, get_output_subdirs
from src.data_cleaner.cleaner import DataCleaner


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="云锦图片数据清洗工具 - 基于 YAML 配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 方式1: 使用 python -m
    python -m src.data_cleaner.run_cleaner

    # 方式2: 使用自定义配置
    python -m src.data_cleaner.run_cleaner --config my_config.yaml

    # 方式3: 覆盖输入输出目录
    python -m src.data_cleaner.run_cleaner --input ./images --output ./cleaned

    # 方式4: 禁用GPU加速
    python -m src.data_cleaner.run_cleaner --no-gpu
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="配置文件路径（默认: data_cleaner/config.yaml）"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入目录（覆盖配置文件）"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出目录（覆盖配置文件）"
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["quality", "style", "full"],
        help="处理模式（覆盖配置文件）"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="禁用GPU加速（覆盖配置文件）"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="安静模式，不显示进度条"
    )

    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # 覆盖配置
    if args.input:
        config.input_dir = Path(args.input)
    if args.output:
        config.output_dir = Path(args.output)
    if args.mode:
        config.mode = args.mode
    if args.no_gpu:
        config.style.use_gpu = False

    # 打印配置信息
    print("=" * 60)
    print("           云锦图片数据清洗工具")
    print("=" * 60)
    print(f"  配置文件: {config_path or 'data_cleaner/config.yaml'}")
    print(f"  输入目录: {config.input_dir}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  处理模式: {config.mode}")
    print(f"  GPU加速: {'是' if config.style.use_gpu else '否'}")
    print(f"  生成报告: {'是' if config.generate_report else '否'}")
    print("-" * 60)

    # 显示子目录配置
    subdirs = get_output_subdirs(config_path)
    print("  输出子目录:")
    for name, path in subdirs.items():
        print(f"    {name}: {config.output_dir / path}")
    print("-" * 60)

    # 创建并运行清洗器
    cleaner = DataCleaner(config)

    statistics = cleaner.clean(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        show_progress=not args.quiet,
    )

    # 打印总结报告
    print("\n" + "=" * 60)
    print("                    云锦图片数据清洗报告")
    print("=" * 60)
    print(f"  处理时间: {statistics['timestamp']}")
    print("-" * 60)
    print("  【图片统计】")
    print(f"    总图片数:     {statistics['total']:>8}")
    print(f"    通过清洗:     {statistics['approved']:>8} ({statistics['approved_rate']:.2%})")
    print(f"    质量不达标:   {statistics['rejected_quality']:>8}")
    print(f"    非云锦风格:   {statistics['rejected_style']:>8}")
    print(f"    需人工复核:   {statistics['needs_review']:>8}")
    print("-" * 60)
    print("  【质量评估】")
    print(f"    平均清晰度:   {statistics['avg_quality_score']:>8.2f}")
    print(f"    平均风格分:   {statistics['avg_style_score']:>8.2f}")
    print("-" * 60)
    print("  【输出目录】")
    for name, path in subdirs.items():
        print(f"    {name:12}: {config.output_dir / path}/")
    print("-" * 60)
    print(f"  清洗完成！共处理 {statistics['total']} 张图片，")
    print(f"  其中 {statistics['approved']} 张可用于 SD1.5 微调训练。")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
