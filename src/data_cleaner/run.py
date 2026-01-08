"""命令行入口 - 使用 YAML 配置文件."""

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .extractor import PatternExtractor

logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        description="全自动图案提取 - 从文物图片中智能提取图案用于模型微调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认使用 config.yaml 配置文件
  python -m data_cleaner.run

  # 指定配置文件
  python -m data_cleaner.run --config my_config.yaml

  # 覆盖配置文件中的输入输出路径
  python -m data_cleaner.run --input output/images --output output/patterns
        """,
    )

    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )

    parser.add_argument(
        "-i", "--input",
        help="输入目录或单张图片路径 (覆盖配置文件)"
    )

    parser.add_argument(
        "-o", "--output",
        help="输出目录 (覆盖配置文件)"
    )

    parser.add_argument(
        "-t", "--target-size",
        type=int,
        help="目标输出尺寸，正方形 (覆盖配置文件)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="启用详细日志输出"
    )

    return parser.parse_args()


def generate_report(stats: dict, output_file: str):
    """生成 Markdown 报告."""
    lines = [
        "# 图片提取处理报告",
        "",
        "## 统计",
        f"- 总计: {stats['total']} 张",
        f"- 成功: {stats['success']} 张",
        f"- 失败: {stats['failed']} 张",
        f"- 总图案数: {stats['total_patterns']}",
        "",
    ]

    if stats["failed_files"]:
        lines.extend([
            "## 失败列表",
            "",
            "| 文件名 | 错误 |",
            "|--------|------|",
        ])
        for name, error in stats["failed_files"]:
            lines.append(f"| {name} | {error} |")

    report = "\n".join(lines) + "\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"报告已保存: {output_file}")


def main():
    """主函数."""
    args = parse_args()

    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 1. 加载 YAML 配置
    config = Config.from_yaml(args.config)

    # 2. 覆盖配置 (命令行参数优先)
    if args.input:
        config.input_dir = args.input
    if args.output:
        config.output_dir = args.output
    if args.target_size:
        config.target_size = args.target_size

    logger.info(f"加载配置文件: {args.config}")
    logger.info(f"输入目录: {config.input_dir}")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info(f"使用 VLM: {config.use_vlm}")

    # 3. 创建提取器
    try:
        extractor = PatternExtractor(config)
    except RuntimeError as e:
        logger.error(f"初始化失败: {e}")
        sys.exit(1)

    input_path = Path(args.input) if args.input else Path(config.input_dir)

    if input_path.is_file():
        # 单张图片处理
        logger.info(f"处理单张图片: {input_path.name}")

        try:
            _, _, patterns = extractor.extract(str(input_path))
            logger.info(f"提取到 {len(patterns)} 个图案")

            # 保存图案
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            base_name = input_path.stem
            for i, pattern in enumerate(patterns):
                output_file = output_dir / f"{base_name}_{i+1:03d}.png"
                pattern.save(output_file, "PNG")
                logger.info(f"  保存: {output_file.name}")

        except Exception as e:
            logger.exception(f"处理失败: {e}")
            sys.exit(1)

    elif input_path.is_dir():
        # 批量处理
        logger.info(f"处理目录: {input_path}")

        stats = extractor.process_directory(
            str(input_path),
            config.output_dir,
            pattern_prefix=""
        )

        # 打印统计
        logger.info(f"\n处理完成!")
        logger.info(f"  总计: {stats['total']} 张")
        logger.info(f"  成功: {stats['success']} 张")
        logger.info(f"  失败: {stats['failed']} 张")
        logger.info(f"  总图案数: {stats['total_patterns']}")

        # 生成报告
        report_file = Path(config.output_dir) / "processing_report.md"
        generate_report(stats, str(report_file))

    else:
        logger.error(f"错误: 路径不存在 - {config.input_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
