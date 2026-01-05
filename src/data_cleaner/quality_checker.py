"""
Image Quality Checker Module.

图片质量检查模块，用于过滤低质量图片。
提供分辨率检查、模糊度检测、色彩检查等功能。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.data_cleaner.content_analyzer import ContentAnalyzer


@dataclass
class QualityResult:
    """图片质量检查结果"""

    image_path: Path
    is_passed: bool
    width: int
    height: int
    sharpness: float
    has_alpha_channel: bool
    is_corrupted: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "image_path": str(self.image_path),
            "is_passed": self.is_passed,
            "width": self.width,
            "height": self.height,
            "sharpness": self.sharpness,
            "has_alpha_channel": self.has_alpha_channel,
            "is_corrupted": self.is_corrupted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class QualityConfig:
    """质量检查配置"""

    min_width: int = 512
    min_height: int = 512
    blur_threshold: float = 50.0
    check_alpha_channel: bool = True
    check_corruption: bool = True


class QualityChecker:
    """图片质量检查器"""

    def __init__(self, config: Optional[QualityConfig] = None):
        """
        初始化质量检查器

        Args:
            config: 质量检查配置，默认为标准配置
        """
        self.config = config or QualityConfig()
        
        # 如果启用了内容检查，初始化内容分析器
        if self.config.check_content_ratio or self.config.check_border_region:
            self.content_analyzer = ContentAnalyzer(
                min_content_ratio=self.config.min_content_ratio,
                max_border_ratio=self.config.max_border_ratio,
                border_region_width=self.config.border_region_width,
            )
        else:
            self.content_analyzer = None

    def check(self, image_path: str | Path) -> QualityResult:
        """
        检查单张图片质量

        Args:
            image_path: 图片路径

        Returns:
            QualityResult: 质量检查结果
        """
        path = Path(image_path)

        # 初始化结果
        result = QualityResult(
            image_path=path,
            is_passed=True,
            width=0,
            height=0,
            sharpness=0.0,
            has_alpha_channel=False,
            is_corrupted=False,
            rejection_reason=None,
        )

        # 1. 检查文件是否存在
        if not path.exists():
            result.is_passed = False
            result.rejection_reason = "file_not_exists"
            return result

        # 2. 加载图片并检查损坏
        try:
            with Image.open(path) as img:
                result.width, result.height = img.size

                # 检查Alpha通道（只拒绝真正有透明像素的图片）
                if self.config.check_alpha_channel:
                    has_alpha = img.mode in ("RGBA", "LA", "P")
                    result.has_alpha_channel = has_alpha
                    
                    if has_alpha:
                        # 检查Alpha通道是否真的有用（是否有透明或半透明像素）
                        if img.mode == "RGBA":
                            # 提取Alpha通道
                            alpha_channel = img.split()[3]
                            # 检查是否有非完全不透明的像素（alpha < 255）
                            alpha_array = np.array(alpha_channel)
                            has_transparency = np.any(alpha_array < 255)
                        elif img.mode == "LA":
                            # LA模式：L是亮度，A是Alpha
                            alpha_channel = img.split()[1]
                            alpha_array = np.array(alpha_channel)
                            has_transparency = np.any(alpha_array < 255)
                        elif img.mode == "P":
                            # P模式（调色板模式）：检查是否有透明色
                            has_transparency = "transparency" in img.info
                        else:
                            has_transparency = False
                        
                        # 只有当Alpha通道真正有用时才拒绝
                        if has_transparency:
                            result.is_passed = False
                            result.rejection_reason = "has_alpha_channel"
                            return result
                        # 如果Alpha通道都是不透明的，可以继续处理

        except Exception:
            result.is_corrupted = True
            result.is_passed = False
            result.rejection_reason = "corrupted_image"
            return result

        # 3. 检查分辨率
        if result.width < self.config.min_width:
            result.is_passed = False
            result.rejection_reason = f"width_too_small({result.width}<{self.config.min_width})"
            return result

        if result.height < self.config.min_height:
            result.is_passed = False
            result.rejection_reason = f"height_too_small({result.height}<{self.config.min_height})"
            return result

        # 4. 计算清晰度（使用Laplacian方差）
        result.sharpness = self._calculate_sharpness(path)

        if result.sharpness < self.config.blur_threshold:
            result.is_passed = False
            result.rejection_reason = f"too_blurry({result.sharpness:.2f}<{self.config.blur_threshold})"
            return result

        # ========== 5. 内容占比检查和边界区域检查（如果启用） ==========
        # 目的：检测图片内容完整性和边界区域，避免空白/边界区域过多的图片通过检查
        # 注意：只进行一次内容分析，避免重复计算
        if (self.config.check_content_ratio or self.config.check_border_region) and self.content_analyzer:
            # 执行内容分析（包括内容占比和边界占比）
            content_result = self.content_analyzer.analyze_content(path)
            
            # 5.1 内容占比检查
            # 目的：确保图片中有足够的有效内容（图案区域），而不是主要是空白
            # 阈值：如果有效内容占比 < min_content_ratio，拒绝该图片
            if self.config.check_content_ratio:
                if content_result.content_ratio < self.config.min_content_ratio:
                    result.is_passed = False
                    result.rejection_reason = f"content_ratio_too_low({content_result.content_ratio:.2f}<{self.config.min_content_ratio})"
                    return result
            
            # 5.2 边界区域检查（适用于所有图片：full_image.png 和瓦片图）
            # 目的：检测图片边缘是否主要是空白/背景，避免边界区域、拼接过渡区域通过检查
            # 阈值：如果边界空白占比 > max_border_ratio，拒绝该图片
            if self.config.check_border_region:
                if content_result.border_ratio > self.config.max_border_ratio:
                    result.is_passed = False
                    result.rejection_reason = f"border_ratio_too_high({content_result.border_ratio:.2f}>{self.config.max_border_ratio})"
                    return result

        return result

    def _calculate_sharpness(self, image_path: Path) -> float:
        """
        计算图片清晰度（使用Laplacian方差）

        Args:
            image_path: 图片路径

        Returns:
            float: 清晰度分数，越高表示越清晰
        """
        try:
            # 使用PIL读取图片（更可靠，支持中文路径）
            with Image.open(image_path) as pil_img:
                # 转换为RGB模式（如果不是的话）
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # 转换为numpy数组
                img_array = np.array(pil_img)
                
                # 转换为BGR格式（OpenCV使用BGR）
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            if img is None or img.size == 0:
                return 0.0

            # 转换为灰度图
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # 检查图片尺寸是否有效
            if gray.shape[0] < 3 or gray.shape[1] < 3:
                return 0.0

            # 计算Laplacian方差
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()

            # 检查是否为NaN或无效值
            if np.isnan(sharpness) or np.isinf(sharpness):
                return 0.0

            return float(sharpness)
        except Exception as e:
            # 记录错误但不中断处理
            # print(f"Warning: Failed to calculate sharpness for {image_path}: {e}")
            return 0.0

    def check_batch(
        self, image_paths: list[str | Path], show_progress: bool = True
    ) -> list[QualityResult]:
        """
        批量检查图片质量

        Args:
            image_paths: 图片路径列表
            show_progress: 是否显示进度条

        Returns:
            list[QualityResult]: 质量检查结果列表
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(image_paths, desc="Quality Checking") if show_progress else image_paths

        for path in iterator:
            result = self.check(path)
            results.append(result)

        return results

    def get_statistics(self, results: list[QualityResult]) -> dict:
        """
        获取批量检查统计信息

        Args:
            results: 质量检查结果列表

        Returns:
            dict: 统计信息
        """
        total = len(results)
        passed = sum(1 for r in results if r.is_passed)

        # 统计拒绝原因
        rejection_counts = {}
        for r in results:
            if not r.is_passed and r.rejection_reason:
                reason = r.rejection_reason
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        return {
            "total": total,
            "passed": passed,
            "rejected": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "rejection_reasons": rejection_counts,
        }


def check_image_quality(
    image_path: str | Path,
    min_width: int = 512,
    min_height: int = 512,
    blur_threshold: float = 50.0,
) -> QualityResult:
    """
    便捷函数：检查单张图片质量

    Args:
        image_path: 图片路径
        min_width: 最小宽度
        min_height: 最小高度
        blur_threshold: 模糊阈值

    Returns:
        QualityResult: 质量检查结果
    """
    config = QualityConfig(
        min_width=min_width,
        min_height=min_height,
        blur_threshold=blur_threshold,
    )
    checker = QualityChecker(config)
    return checker.check(image_path)
