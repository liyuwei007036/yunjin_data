"""命令行入口 - 使用 YAML 配置文件."""

import argparse
import sys
from pathlib import Path

from .config import Config
from .extractor import PatternExtractor


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

    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> dict:
    """从 YAML 文件加载配置."""
    import yaml

    if not Path(config_path).exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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

    print(f"报告已保存: {output_file}")


def main():
    """主函数."""
    args = parse_args()

    # 1. 加载 YAML 配置
    yaml_config = load_config_from_yaml(args.config)

    # 2. 构建配置 (命令行参数覆盖 YAML)
    config = Config(
        input_dir=args.input or yaml_config.get("input_dir", "output/images"),
        output_dir=args.output or yaml_config.get("output_dir", "output/patterns"),
        target_size=args.target_size or yaml_config.get("target_size", 1024),
        min_crop_size=yaml_config.get("min_crop_size", 512),
        max_patterns_per_image=yaml_config.get("max_patterns_per_image", 20),
        use_vlm=yaml_config.get("use_vlm", True),
        vlm_model=yaml_config.get("vlm_model", "Qwen/Qwen2-VL-7B-Instruct"),
        grounding_dino_path=yaml_config.get("grounding_dino_path", "models/grounding_dino/grounding_dino_swin-t_ogc.pth"),
        sam_model_path=yaml_config.get("sam_model_path", "models/sam/sam_vit_l_0b3195.pth"),
        box_threshold=yaml_config.get("box_threshold", 0.25),
        text_threshold=yaml_config.get("text_threshold", 0.25),
        sam_mask_threshold=yaml_config.get("sam_mask_threshold", 0.5),
        min_quality_score=yaml_config.get("min_quality_score", 10.0),
        min_content_ratio=yaml_config.get("min_content_ratio", 0.1),
    )

    print(f"加载配置文件: {args.config}")
    print(f"输入目录: {config.input_dir}")
    print(f"输出目录: {config.output_dir}")
    print(f"使用 VLM: {config.use_vlm}")

    # 3. 创建提取器
    extractor = PatternExtractor(config)

    input_path = Path(args.input) if args.input else Path(config.input_dir)

    if input_path.is_file():
        # 单张图片处理
        print(f"处理单张图片: {input_path.name}")

        try:
            patterns = extractor.extract(str(input_path))
            print(f"提取到 {len(patterns)} 个图案")

            # 保存图案
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            base_name = input_path.stem
            for i, pattern in enumerate(patterns):
                output_file = output_dir / f"{base_name}_{i+1:03d}.png"
                pattern.save(output_file, "PNG")
                print(f"  保存: {output_file.name}")

        except Exception as e:
            print(f"处理失败: {e}")
            sys.exit(1)

    elif input_path.is_dir():
        # 批量处理
        print(f"处理目录: {input_path}")

        stats = extractor.process_directory(
            str(input_path),
            config.output_dir,
            pattern_prefix=""
        )

        # 打印统计
        print(f"\n处理完成!")
        print(f"  总计: {stats['total']} 张")
        print(f"  成功: {stats['success']} 张")
        print(f"  失败: {stats['failed']} 张")
        print(f"  总图案数: {stats['total_patterns']}")

        # 生成报告
        report_file = Path(config.output_dir) / "processing_report.md"
        generate_report(stats, str(report_file))

    else:
        print(f"错误: 路径不存在 - {config.input_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
