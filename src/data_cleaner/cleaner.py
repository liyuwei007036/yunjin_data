"""
Main Data Cleaner Module.

云锦图片数据清洗主模块，整合质量检查和风格分类。
支持与 digicol-scraper 输出目录结构保持一致。
"""

import json
import shutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# 确保 src 目录在 Python 路径中（支持直接运行脚本）
_src_path = Path(__file__).parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from data_cleaner.config import (
    CleanerConfig,
    QualityConfig,
    StyleConfig,
    APPROVED_DIR,
    REJECTED_QUALITY_DIR,
    REJECTED_STYLE_DIR,
    REVIEW_DIR,
    REPORTS_DIR,
    PROJECT_ROOT,
)
from data_cleaner.quality_checker import QualityChecker, QualityResult
from data_cleaner.style_classifier import StyleClassifier, StyleResult


@dataclass
class CleaningResult:
    """完整清洗结果"""

    image_path: Path
    quality_result: Optional[QualityResult]
    style_result: Optional[StyleResult]
    final_status: str  # approved, rejected_quality, rejected_style, review
    quality_score: float = 0.0
    style_score: float = 0.0

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "image_path": str(self.image_path),
            "final_status": self.final_status,
            "quality_score": self.quality_score,
            "style_score": self.style_score,
        }


@dataclass
class ImageMetadata:
    """图片元数据（标注文件内容）"""

    # 文件信息
    image_file: str = "full_image.png"
    metadata_file: str = "metadata.json"

    # 清洗结果
    status: str = ""  # approved, rejected_quality, rejected_style, review

    # 质量检查数据
    width: int = 0
    height: int = 0
    sharpness: float = 0.0
    is_corrupted: bool = False
    has_alpha_channel: bool = False

    # 风格分类数据
    style_score: float = 0.0  # 云锦置信度
    is_brocade: bool = False
    needs_review: bool = False

    # 处理信息
    processed_at: str = ""
    quality_rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典格式（排除None值）"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_results(
        cls,
        image_path: Path,
        quality_result: Optional[QualityResult],
        style_result: Optional[StyleResult],
        final_status: str,
    ) -> "ImageMetadata":
        """从清洗结果创建元数据"""
        metadata = cls()

        # 从文物名目录获取信息
        metadata.artifact_name = image_path.parent.name

        metadata.status = final_status
        metadata.processed_at = datetime.now().isoformat()

        # 质量检查数据
        if quality_result:
            metadata.width = quality_result.width
            metadata.height = quality_result.height
            metadata.sharpness = quality_result.sharpness
            metadata.is_corrupted = quality_result.is_corrupted
            metadata.has_alpha_channel = quality_result.has_alpha_channel
            if not quality_result.is_passed:
                metadata.quality_rejection_reason = quality_result.rejection_reason

        # 风格分类数据
        if style_result:
            metadata.style_score = style_result.brocade_score
            metadata.is_brocade = style_result.is_brocade
            metadata.needs_review = style_result.needs_review

        return metadata


class DataCleaner:
    """图片数据清洗主模块"""

    def __init__(self, config: Optional[CleanerConfig] = None):
        """
        初始化数据清洗器

        Args:
            config: 清洗配置，默认为标准配置
        """
        self.config = config or CleanerConfig()
        self.quality_checker = QualityChecker(self.config.quality)
        self.style_checker = StyleClassifier(self.config.style)

        # 创建输出目录
        self._create_output_dirs()

    def _create_output_dirs(self):
        """创建输出目录"""
        APPROVED_DIR.mkdir(parents=True, exist_ok=True)
        REJECTED_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
        REJECTED_STYLE_DIR.mkdir(parents=True, exist_ok=True)
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def find_images(self, input_dir: Path) -> list[Path]:
        """
        查找目录下所有文物图片（精确匹配爬虫输出结构）

        爬虫输出结构：
            output/{文物名}_{编号}/merged/full_image.png

        Args:
            input_dir: 输入目录

        Returns:
            list[Path]: 图片路径列表
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            return []

        image_paths = []

        # 查找所有 {dir}/merged/full_image.png
        for merged_dir in input_path.rglob("merged"):
            image_path = merged_dir / "full_image.png"
            if image_path.exists():
                image_paths.append(image_path)

        return sorted(set(image_paths))

    def clean_single(self, image_path: str | Path) -> CleaningResult:
        """
        清洗单张图片

        Args:
            image_path: 图片路径

        Returns:
            CleaningResult: 清洗结果
        """
        path = Path(image_path)

        # 1. 质量检查
        if self.config.mode in ("quality", "full"):
            quality_result = self.quality_checker.check(path)
            if not quality_result.is_passed:
                return CleaningResult(
                    image_path=path,
                    quality_result=quality_result,
                    style_result=None,
                    final_status="rejected_quality",
                    quality_score=quality_result.sharpness,
                )
        else:
            quality_result = None

        # 2. 风格分类
        if self.config.mode in ("style", "full"):
            style_result = self.style_checker.classify(path)
            if style_result.needs_review:
                return CleaningResult(
                    image_path=path,
                    quality_result=quality_result,
                    style_result=style_result,
                    final_status="review",
                    quality_score=quality_result.sharpness if quality_result else 1.0,
                    style_score=style_result.brocade_score,
                )
            elif not style_result.is_brocade:
                return CleaningResult(
                    image_path=path,
                    quality_result=quality_result,
                    style_result=style_result,
                    final_status="rejected_style",
                    quality_score=quality_result.sharpness if quality_result else 1.0,
                    style_score=style_result.brocade_score,
                )
        else:
            style_result = None

        # 3. 通过所有检查
        return CleaningResult(
            image_path=path,
            quality_result=quality_result,
            style_result=style_result,
            final_status="approved",
            quality_score=quality_result.sharpness if quality_result else 1.0,
            style_score=style_result.brocade_score if style_result else 1.0,
        )

    def clean(
        self,
        input_dir: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        show_progress: bool = True,
    ) -> dict:
        """
        批量清洗图片

        Args:
            input_dir: 输入目录，默认使用配置中的目录
            output_dir: 输出目录，默认使用配置中的目录
            show_progress: 是否显示进度条

        Returns:
            dict: 清洗统计信息
        """
        input_path = Path(input_dir) if input_dir else self.config.input_dir
        output_path = Path(output_dir) if output_dir else self.config.output_dir

        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_path}")

        # 查找图片
        print(f"Scanning for images in {input_path}...")
        image_paths = self.find_images(input_path)
        print(f"Found {len(image_paths)} images")

        if not image_paths:
            print("No images found")
            return {"total": 0, "approved": 0}

        # 加载CLIP模型（如果在风格模式下）
        if self.config.mode in ("style", "full"):
            self.style_checker.load_model()

        # 批量处理
        results: list[CleaningResult] = []
        iterator = tqdm(image_paths, desc="Cleaning") if show_progress else image_paths

        for path in iterator:
            result = self.clean_single(path)
            results.append(result)

            # 移动文件到对应目录
            self._move_result_file(path, result, output_path)

        # 生成报告
        statistics = self._generate_statistics(results)

        if self.config.generate_report:
            self._save_report(results, statistics)

        return statistics

    def _move_result_file(
        self, image_path: Path, result: CleaningResult, output_base: Path
    ):
        """
        根据清洗结果复制文件并生成标注文件

        Args:
            image_path: 原始图片路径
            result: 清洗结果
            output_base: 输出基准目录
        """
        # 选择目标目录
        if result.final_status == "approved":
            target_dir = APPROVED_DIR / image_path.parent.name
        elif result.final_status == "rejected_quality":
            target_dir = REJECTED_QUALITY_DIR / image_path.parent.name
        elif result.final_status == "rejected_style":
            target_dir = REJECTED_STYLE_DIR / image_path.parent.name
        else:  # review
            target_dir = REVIEW_DIR / image_path.parent.name

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / image_path.name

        try:
            # 复制图片文件
            shutil.copy2(image_path, target_path)

            # 生成标注文件 metadata.json
            metadata = ImageMetadata.from_results(
                image_path=image_path,
                quality_result=result.quality_result,
                style_result=result.style_result,
                final_status=result.final_status,
            )

            metadata_path = target_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error copying {image_path}: {e}")

    def _generate_statistics(self, results: list[CleaningResult]) -> dict:
        """
        生成清洗统计信息

        Args:
            results: 清洗结果列表

        Returns:
            dict: 统计信息
        """
        total = len(results)
        approved = sum(1 for r in results if r.final_status == "approved")
        rejected_quality = sum(1 for r in results if r.final_status == "rejected_quality")
        rejected_style = sum(1 for r in results if r.final_status == "rejected_style")
        review = sum(1 for r in results if r.final_status == "review")

        # 计算平均分数
        approved_results = [r for r in results if r.final_status == "approved"]
        avg_quality_score = (
            sum(r.quality_score for r in approved_results) / len(approved_results)
            if approved_results
            else 0
        )
        avg_style_score = (
            sum(r.style_score for r in approved_results) / len(approved_results)
            if approved_results
            else 0
        )

        return {
            "total": total,
            "approved": approved,
            "rejected_quality": rejected_quality,
            "rejected_style": rejected_style,
            "needs_review": review,
            "approved_rate": approved / total if total > 0 else 0,
            "avg_quality_score": avg_quality_score,
            "avg_style_score": avg_style_score,
            "timestamp": datetime.now().isoformat(),
        }

    def _save_report(self, results: list[CleaningResult], statistics: dict):
        """
        保存清洗报告

        Args:
            results: 清洗结果列表
            statistics: 统计信息
        """
        report = {
            "statistics": statistics,
            "details": [r.to_dict() for r in results],
        }

        report_path = REPORTS_DIR / f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\nReport saved to: {report_path}")

    def export_training_json(
        self, output_path: Path = REPORTS_DIR, caption_prefix: str = "cloud brocade, traditional Chinese textile"
    ):
        """
        导出训练用的JSON文件（可用于SD1.5微调）

        Args:
            output_path: 输出路径
            caption_prefix: 图片caption前缀
        """
        training_data = []

        # 收集所有approved目录下的图片
        for artifact_dir in APPROVED_DIR.iterdir():
            if artifact_dir.is_dir():
                image_path = artifact_dir / "full_image.png"
                metadata_path = artifact_dir / "metadata.json"

                if image_path.exists():
                    # 读取元数据获取风格分数
                    style_score = 0.0
                    if metadata_path.exists():
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                            style_score = metadata.get("style_score", 0.0)

                    # 生成caption（可根据风格分数调整）
                    confidence_suffix = ""
                    if style_score >= 0.8:
                        confidence_suffix = ", high quality cloud brocade"
                    elif style_score >= 0.7:
                        confidence_suffix = ", detailed brocade pattern"

                    caption = f"{caption_prefix}{confidence_suffix}"

                    training_data.append({
                        "image": str(image_path.relative_to(PROJECT_ROOT)),
                        "caption": caption,
                        "style_score": style_score,
                    })

        # 保存训练数据
        training_file = output_path / "training_data.json"
        with open(training_file, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        print(f"Training data saved to: {training_file}")
        print(f"Total training samples: {len(training_data)}")

        return training_data


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="云锦图片数据清洗工具")
    parser.add_argument(
        "--input", "-i", type=str, help="输入目录路径"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="输出目录路径"
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["quality", "style", "full"],
        default="full",
        help="清洗模式：quality(仅质量)、style(仅风格)、full(完整)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="禁用GPU"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="安静模式，不显示进度条"
    )

    args = parser.parse_args()

    # 创建配置
    style_config = StyleConfig(
        use_gpu=not args.no_gpu,
    )

    config = CleanerConfig(
        style=style_config,
        mode=args.mode,
    )

    # 创建清洗器并运行
    cleaner = DataCleaner(config)

    statistics = cleaner.clean(
        input_dir=args.input,
        output_dir=args.output,
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
    print(f"    通过图片:     approved/")
    print(f"    质量拒绝:     rejected_quality/")
    print(f"    风格拒绝:     rejected_style/")
    print(f"    待复核:       review/")
    print(f"    清洗报告:     reports/")
    print("-" * 60)
    print(f"  清洗完成！共处理 {statistics['total']} 张图片，")
    print(f"  其中 {statistics['approved']} 张可用于 SD1.5 微调训练。")
    print("=" * 60)


if __name__ == "__main__":
    main()
