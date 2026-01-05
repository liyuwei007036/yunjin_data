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

                # 检查Alpha通道
                if self.config.check_alpha_channel:
                    result.has_alpha_channel = img.mode in ("RGBA", "LA", "P")
                    if result.has_alpha_channel:
                        result.is_passed = False
                        result.rejection_reason = "has_alpha_channel"
                        return result

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
            # 使用OpenCV加载图片
            img = cv2.imread(str(image_path))
            if img is None:
                return 0.0

            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 计算Laplacian方差
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()

            return float(sharpness)
        except Exception:
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
