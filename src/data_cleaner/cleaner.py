"""
Main Data Cleaner Module.

云锦图片数据清洗主模块，整合质量检查和风格分类。
支持与 digicol-scraper 输出目录结构保持一致。
"""

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from src.data_cleaner.config import (
    CleanerConfig,
    StyleConfig,
    PROJECT_ROOT,
    get_output_dir_path,
)
from src.data_cleaner.quality_checker import QualityChecker, QualityResult
from src.data_cleaner.style_classifier import StyleClassifier, StyleResult
from src.data_cleaner.auto_captioner import AutoCaptioner


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
        self.captioner = AutoCaptioner(self.config.caption) if self.config.caption.enabled else None

        # 初始化输出目录路径（基于配置的输出目录）
        output_dir = Path(self.config.output_dir)
        self.approved_dir = get_output_dir_path("approved", output_dir)
        self.rejected_quality_dir = get_output_dir_path("rejected_quality", output_dir)
        self.rejected_style_dir = get_output_dir_path("rejected_style", output_dir)
        self.review_dir = get_output_dir_path("review", output_dir)
        self.reports_dir = get_output_dir_path("reports", output_dir)

        # 创建输出目录
        self._create_output_dirs()

    def _resize_image(self, img: Image.Image, target_width: int, target_height: int, mode: str) -> Image.Image:
        """
        缩放图片到目标尺寸
        
        Args:
            img: PIL图片对象
            target_width: 目标宽度
            target_height: 目标高度
            mode: 缩放模式 ('stretch', 'fit', 'crop')
        
        Returns:
            Image.Image: 缩放后的图片
        """
        original_width, original_height = img.size
        
        # 如果尺寸已经匹配，直接返回
        if original_width == target_width and original_height == target_height:
            return img
        
        if mode == "stretch":
            # 强制拉伸到目标尺寸
            return img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        elif mode == "fit":
            # 等比例缩放+填充（letterbox/pillarbox）
            # 计算缩放比例，选择较小的比例以确保图片完全包含在目标尺寸内
            scale = min(target_width / original_width, target_height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # 等比例缩放
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 创建目标尺寸的黑色背景
            result = Image.new("RGB", (target_width, target_height), (0, 0, 0))
            
            # 计算居中位置
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # 将缩放后的图片粘贴到中心
            result.paste(resized, (x_offset, y_offset))
            return result
        
        elif mode == "crop":
            # 等比例缩放+中心裁剪
            # 计算缩放比例，选择较大的比例以确保填满目标尺寸
            scale = max(target_width / original_width, target_height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # 等比例缩放
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 计算裁剪位置（居中）
            x_offset = (new_width - target_width) // 2
            y_offset = (new_height - target_height) // 2
            
            # 裁剪到目标尺寸
            return resized.crop((x_offset, y_offset, x_offset + target_width, y_offset + target_height))
        
        else:
            # 默认使用强制拉伸
            return img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    def _create_output_dirs(self):
        """创建输出目录"""
        self.approved_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_quality_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_style_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def find_images(self, input_dir: Path) -> list[Path]:
        """
        查找目录下所有文物图片（精确匹配爬虫输出结构）

        爬虫输出结构：
            output/{文物名}_{编号}/merged/full_image.png
            output/{文物名}_{编号}/tiles/level_{level}/{col}_{row}.png

        Args:
            input_dir: 输入目录

        Returns:
            list[Path]: 图片路径列表
        """
        import re
        
        input_path = Path(input_dir)
        if not input_path.exists():
            return []

        image_paths = []

        # 1. 查找所有 {dir}/merged/full_image.png
        for merged_dir in input_path.rglob("merged"):
            image_path = merged_dir / "full_image.png"
            if image_path.exists():
                image_paths.append(image_path)

        # 2. 查找所有瓦片图 {dir}/tiles/level_{level}/{col}_{row}.png
        for tiles_dir in input_path.rglob("tiles"):
            # 查找所有 level_* 目录
            for level_dir in tiles_dir.iterdir():
                if level_dir.is_dir() and level_dir.name.startswith("level_"):
                    # 查找所有符合 {col}_{row}.png 格式的图片
                    for tile_file in level_dir.iterdir():
                        if tile_file.is_file() and tile_file.suffix.lower() == ".png":
                            # 验证文件名格式：数字_数字.png
                            match = re.match(r"^(\d+)_(\d+)\.png$", tile_file.name)
                            if match:
                                image_paths.append(tile_file)

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
            # 判断是否是瓦片图（路径包含 tiles/level_）
            is_tile = "tiles" in str(path) and "level_" in str(path)
            
            # 对瓦片图使用目标尺寸作为最小尺寸要求（因为所有图片都会统一缩放到目标尺寸）
            if is_tile:
                from src.data_cleaner.config import QualityConfig
                tile_config = QualityConfig(
                    min_width=self.config.target_width,  # 使用配置的目标尺寸
                    min_height=self.config.target_height,
                    blur_threshold=self.config.quality.blur_threshold,
                    check_alpha_channel=self.config.quality.check_alpha_channel,
                    check_corruption=self.config.quality.check_corruption,
                )
                # 临时使用瓦片图配置
                original_config = self.quality_checker.config
                self.quality_checker.config = tile_config
                quality_result = self.quality_checker.check(path)
                self.quality_checker.config = original_config
            else:
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

        # 生成自动标注（如果启用）
        annotation_stats = None
        if self.config.caption.enabled and self.captioner:
            print("\nGenerating annotations for approved images...")
            try:
                annotation_stats = self.generate_annotations(show_progress=show_progress)
            except Exception as e:
                print(f"Warning: Failed to generate annotations: {e}")

        # 将标注统计信息添加到总统计中
        if annotation_stats:
            statistics["annotation"] = annotation_stats
        else:
            statistics["annotation"] = None

        return statistics

    def _get_artifact_name(self, image_path: Path) -> str:
        """
        从图片路径中提取文物名（目录名）

        Args:
            image_path: 图片路径

        Returns:
            str: 文物名（目录名）
        """
        path = Path(image_path)
        
        # 如果是 merged/full_image.png，文物名在 parent.parent
        if path.parent.name == "merged":
            return path.parent.parent.name
        
        # 如果是 tiles/level_*/{col}_{row}.png，文物名在 parent.parent.parent
        if path.parent.parent.name == "tiles":
            return path.parent.parent.parent.name
        
        # 默认使用 parent.name
        return path.parent.name

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
        # 获取文物名
        artifact_name = self._get_artifact_name(image_path)
        
        # 选择目标目录
        if result.final_status == "approved":
            target_dir = self.approved_dir / artifact_name
        elif result.final_status == "rejected_quality":
            target_dir = self.rejected_quality_dir / artifact_name
        elif result.final_status == "rejected_style":
            target_dir = self.rejected_style_dir / artifact_name
        else:  # review
            target_dir = self.review_dir / artifact_name

        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 对于瓦片图，保持目录结构：tiles/level_{level}/{col}_{row}.png
        # 对于 full_image.png，直接使用文件名
        if image_path.parent.name == "merged":
            target_path = target_dir / image_path.name
        else:
            # 瓦片图：保持相对路径结构
            # 从 tiles 目录开始构建相对路径
            relative_path = image_path.relative_to(image_path.parent.parent.parent)
            target_path = target_dir / relative_path
        
        # 确保目标路径的父目录存在
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 所有图片统一缩放到配置的目标尺寸
            target_width = self.config.target_width
            target_height = self.config.target_height
            resize_mode = self.config.resize_mode
            
            # 打开图片
            with Image.open(image_path) as img:
                # 检查是否需要处理
                original_size = img.size
                needs_resize = original_size != (target_width, target_height)
                needs_convert = img.mode != "RGB"
                
                # 性能优化：如果尺寸和模式都匹配，直接复制
                if not needs_resize and not needs_convert:
                    # 直接复制文件（最快）
                    shutil.copy2(image_path, target_path)
                else:
                    # 需要处理：转换为RGB（如果需要）并缩放
                    if needs_convert:
                        img = img.convert("RGB")
                    
                    # 缩放图片（根据配置的模式）
                    if needs_resize:
                        img = self._resize_image(img, target_width, target_height, resize_mode)
                    
                    # 保存处理后的图片
                    img.save(target_path, "PNG", optimize=True)

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

        # 统计质量拒绝原因
        quality_rejection_reasons = {}
        for r in results:
            if r.final_status == "rejected_quality" and r.quality_result and r.quality_result.rejection_reason:
                reason = r.quality_result.rejection_reason
                # 提取主要拒绝原因（去掉具体数值）
                if reason.startswith("width_too_small"):
                    main_reason = "width_too_small"
                elif reason.startswith("height_too_small"):
                    main_reason = "height_too_small"
                elif reason.startswith("too_blurry"):
                    main_reason = "too_blurry"
                else:
                    main_reason = reason
                quality_rejection_reasons[main_reason] = quality_rejection_reasons.get(main_reason, 0) + 1

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
            "quality_rejection_reasons": quality_rejection_reasons,
            "timestamp": datetime.now().isoformat(),
        }

    def _save_report(self, results: list[CleaningResult], statistics: dict):
        """
        保存清洗报告

        Args:
            results: 清洗结果列表
            statistics: 统计信息
        """
        report_path = self.reports_dir / f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("                    云锦图片数据清洗报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"  处理时间: {statistics['timestamp']}\n")
            f.write("-" * 60 + "\n")
            f.write("  【图片统计】\n")
            f.write(f"    总图片数:     {statistics['total']:>8}\n")
            f.write(f"    通过清洗:     {statistics['approved']:>8} ({statistics['approved_rate']:.2%})\n")
            f.write(f"    质量不达标:   {statistics['rejected_quality']:>8}\n")
            f.write(f"    非云锦风格:   {statistics['rejected_style']:>8}\n")
            f.write(f"    需人工复核:   {statistics['needs_review']:>8}\n")
            f.write("-" * 60 + "\n")
            f.write("  【质量评估】\n")
            f.write(f"    平均清晰度:   {statistics['avg_quality_score']:>8.2f}\n")
            f.write(f"    平均风格分:   {statistics['avg_style_score']:>8.2f}\n")
            
            # 质量拒绝原因统计
            if statistics.get('quality_rejection_reasons'):
                f.write("-" * 60 + "\n")
                f.write("  【质量拒绝原因统计】\n")
                reason_names = {
                    'width_too_small': '宽度不足',
                    'height_too_small': '高度不足',
                    'too_blurry': '清晰度不足',
                    'has_alpha_channel': '包含Alpha通道',
                    'corrupted_image': '图片损坏',
                    'file_not_exists': '文件不存在'
                }
                for reason, count in sorted(statistics['quality_rejection_reasons'].items(), key=lambda x: x[1], reverse=True):
                    reason_name = reason_names.get(reason, reason)
                    f.write(f"    {reason_name:12}: {count:>8}\n")
            
            # 自动标注信息
            if statistics.get('annotation'):
                f.write("-" * 60 + "\n")
                f.write("  【自动标注】\n")
                annotation_info = statistics['annotation']
                f.write(f"    标注图片数:   {annotation_info['images_annotated']:>8}\n")
                f.write(f"    生成标注数:   {annotation_info['annotation_count']:>8}\n")
                f.write(f"    标注文件:     {annotation_info['metadata_file']}\n")
            elif self.config.caption.enabled:
                f.write("-" * 60 + "\n")
                f.write("  【自动标注】\n")
                f.write("    状态:         未生成（标注功能已启用但生成失败）\n")
            else:
                f.write("-" * 60 + "\n")
                f.write("  【自动标注】\n")
                f.write("    状态:         未启用\n")
            
            f.write("-" * 60 + "\n")
            f.write("  【输出目录】\n")
            f.write(f"    approved    : {self.approved_dir}/\n")
            f.write(f"    rejected_quality: {self.rejected_quality_dir}/\n")
            f.write(f"    rejected_style: {self.rejected_style_dir}/\n")
            f.write(f"    review      : {self.review_dir}/\n")
            f.write(f"    reports     : {self.reports_dir}/\n")
            f.write("-" * 60 + "\n")
            f.write(f"  清洗完成！共处理 {statistics['total']} 张图片，\n")
            f.write(f"  其中 {statistics['approved']} 张可用于 SD1.5 微调训练。\n")
            f.write("=" * 60 + "\n")
            
            # 详细结果（可选）
            f.write("\n" + "=" * 60 + "\n")
            f.write("  【详细结果】\n")
            f.write("=" * 60 + "\n")
            for i, result in enumerate(results, 1):
                f.write(f"\n[{i}/{len(results)}] {result.image_path}\n")
                f.write(f"  状态: {result.final_status}\n")
                if result.quality_result:
                    f.write(f"  质量: 清晰度={result.quality_result.sharpness:.2f}, "
                           f"尺寸={result.quality_result.width}x{result.quality_result.height}\n")
                    if result.quality_result.rejection_reason:
                        f.write(f"  拒绝原因: {result.quality_result.rejection_reason}\n")
                if result.style_result:
                    f.write(f"  风格: 云锦分数={result.style_result.brocade_score:.3f}, "
                           f"是否云锦={result.style_result.is_brocade}\n")

        print(f"\nReport saved to: {report_path}")

    def export_training_json(
        self, output_path: Optional[Path] = None, caption_prefix: str = "cloud brocade, traditional Chinese textile"
    ):
        """
        导出训练用的JSON文件（可用于SD1.5微调）

        Args:
            output_path: 输出路径，默认为 reports_dir
            caption_prefix: 图片caption前缀
        """
        if output_path is None:
            output_path = self.reports_dir
        
        training_data = []

        # 收集所有approved目录下的图片（包括 full_image.png 和瓦片图）
        for artifact_dir in self.approved_dir.iterdir():
            if artifact_dir.is_dir():
                metadata_path = artifact_dir / "metadata.json"
                
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

                # 1. 查找 full_image.png
                image_path = artifact_dir / "full_image.png"
                if image_path.exists():
                    training_data.append({
                        "image": str(image_path.relative_to(PROJECT_ROOT)),
                        "caption": caption,
                        "style_score": style_score,
                    })
                
                # 2. 查找所有瓦片图 tiles/level_*/{col}_{row}.png
                tiles_dir = artifact_dir / "tiles"
                if tiles_dir.exists():
                    for level_dir in tiles_dir.iterdir():
                        if level_dir.is_dir() and level_dir.name.startswith("level_"):
                            for tile_file in level_dir.iterdir():
                                if tile_file.is_file() and tile_file.suffix.lower() == ".png":
                                    training_data.append({
                                        "image": str(tile_file.relative_to(PROJECT_ROOT)),
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

    def generate_annotations(
        self, approved_dir: Optional[Path] = None, show_progress: bool = True
    ) -> Optional[dict]:
        """
        为 approved 目录下的所有图片生成标注文件（metadata.jsonl）
        符合 Kohya_ss 格式要求

        Args:
            approved_dir: approved 目录路径，默认使用配置的目录
            show_progress: 是否显示进度条

        Returns:
            dict: 标注统计信息，包含 annotation_count 和 metadata_file
        """
        if not self.captioner:
            print("Auto captioner is not enabled or initialized")
            return None

        approved_path = approved_dir or self.approved_dir

        if not approved_path.exists():
            print(f"Approved directory does not exist: {approved_path}")
            return None

        # 收集所有图片文件（包括 full_image.png 和瓦片图）
        image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        image_paths = [
            p
            for p in approved_path.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            print("No images found in approved directory")
            return None

        print(f"Found {len(image_paths)} images to annotate")

        # 生成 caption
        captions = self.captioner.generate_captions_batch(
            image_paths, show_progress=show_progress
        )

        # 生成 metadata.jsonl
        metadata_file = approved_path / "metadata.jsonl"
        annotation_count = 0

        with open(metadata_file, "w", encoding="utf-8") as f:
            for image_path, caption in captions.items():
                # 计算相对于 approved 目录的相对路径
                relative_path = image_path.relative_to(approved_path)
                # 转换为使用正斜杠的路径（Kohya_ss 要求）
                file_name = str(relative_path).replace("\\", "/")

                # 写入 JSONL 格式
                annotation = {
                    "file_name": file_name,
                    "text": caption,
                }
                f.write(json.dumps(annotation, ensure_ascii=False) + "\n")
                annotation_count += 1

        print(f"\nAnnotations saved to: {metadata_file}")
        print(f"Total annotations: {annotation_count}")

        return {
            "annotation_count": annotation_count,
            "metadata_file": str(metadata_file),
            "images_annotated": len(image_paths),
        }


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
