"""
Content Analyzer Module.

内容分析器模块，用于分析图片内容完整性、边界区域等。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class ContentAnalysisResult:
    """内容分析结果"""

    content_ratio: float  # 有效内容占比（0.0-1.0）
    border_ratio: float    # 边界空白占比（0.0-1.0）
    is_valid: bool         # 是否有效（基于阈值判断）

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "content_ratio": self.content_ratio,
            "border_ratio": self.border_ratio,
            "is_valid": self.is_valid,
        }


class ContentAnalyzer:
    """内容分析器"""

    def __init__(
        self,
        min_content_ratio: float = 0.3,
        max_border_ratio: float = 0.5,
        border_region_width: float = 0.15,
    ):
        """
        初始化内容分析器

        Args:
            min_content_ratio: 最小内容占比阈值
            max_border_ratio: 最大边界空白占比阈值
            border_region_width: 边界区域宽度比例（默认15%）
        """
        self.min_content_ratio = min_content_ratio
        self.max_border_ratio = max_border_ratio
        self.border_region_width = border_region_width

    def calculate_content_ratio(self, image_path: Path) -> float:
        """
        计算有效内容占比

        使用颜色分析和边缘检测的组合方法。

        Args:
            image_path: 图片路径

        Returns:
            float: 有效内容占比（0.0-1.0）
        """
        try:
            # 使用PIL读取图片（更可靠，支持中文路径）
            with Image.open(image_path) as pil_img:
                # 转换为RGB模式
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # 转换为numpy数组
                img_array = np.array(pil_img)
            
            if img_array.size == 0:
                return 0.0

            h, w = img_array.shape[:2]
            total_pixels = h * w

            # 方法1: 颜色分析（HSV色彩空间）
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # 检测非中性色区域
            # 饱和度阈值：低饱和度（< 30）认为是中性色/灰白色
            # 亮度范围：排除过暗或过亮的区域
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            
            # 非中性色像素：饱和度 > 30 且 亮度在合理范围内
            non_neutral_mask = (saturation > 30) & (value > 20) & (value < 240)
            color_content_ratio = np.sum(non_neutral_mask) / total_pixels

            # 方法2: 边缘检测
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 使用Canny边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 计算边缘密度（边缘像素占比）
            edge_pixels = np.sum(edges > 0)
            edge_ratio = edge_pixels / total_pixels
            
            # 使用形态学操作识别图案区域
            # 对边缘进行膨胀，连接附近的边缘
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # 计算有边缘的区域占比
            edge_content_ratio = np.sum(dilated > 0) / total_pixels

            # 组合两种方法：取最大值（更宽松）或平均值（更严格）
            # 使用最大值，因为有些图案可能颜色单一但边缘明显，或颜色丰富但边缘较少
            content_ratio = max(color_content_ratio, edge_content_ratio)

            return float(np.clip(content_ratio, 0.0, 1.0))

        except Exception as e:
            # 如果分析失败，返回0（表示无有效内容）
            return 0.0

    def detect_border_region(self, image_path: Path) -> float:
        """
        检测图片边界空白区域占比（适用于所有图片）

        Args:
            image_path: 图片路径

        Returns:
            float: 边界空白占比（0.0-1.0）
        """
        try:
            # 使用PIL读取图片（更可靠，支持中文路径）
            with Image.open(image_path) as pil_img:
                # 转换为RGB模式
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # 转换为numpy数组
                img_array = np.array(pil_img)
            
            if img_array.size == 0:
                return 1.0  # 如果图片无效，认为全是边界

            h, w = img_array.shape[:2]
            
            # 计算边界区域尺寸
            border_h = max(1, int(h * self.border_region_width))
            border_w = max(1, int(w * self.border_region_width))

            # 提取四边区域
            top = img_array[0:border_h, :]
            bottom = img_array[h-border_h:h, :]
            left = img_array[:, 0:border_w]
            right = img_array[:, w-border_w:w]

            # 计算边界区域总像素数（注意重叠部分）
            top_pixels = border_h * w
            bottom_pixels = border_h * w
            left_pixels = border_w * h
            right_pixels = border_w * h
            # 减去四个角的重复计算
            corner_pixels = 2 * border_h * border_w
            total_border_pixels = top_pixels + bottom_pixels + left_pixels + right_pixels - corner_pixels

            # 合并所有边界区域进行分析
            border_regions = []
            if top.size > 0:
                border_regions.append(top)
            if bottom.size > 0:
                border_regions.append(bottom)
            if left.size > 0:
                border_regions.append(left)
            if right.size > 0:
                border_regions.append(right)

            if not border_regions:
                return 0.0

            # 将所有边界区域合并
            border_pixels_list = []
            for region in border_regions:
                if region.ndim == 3:
                    # RGB图像，展平为像素列表
                    pixels = region.reshape(-1, 3)
                else:
                    # 灰度图
                    pixels = region.reshape(-1, 1)
                border_pixels_list.append(pixels)
            
            if not border_pixels_list:
                return 0.0

            all_border_pixels = np.vstack(border_pixels_list)

            # 转换为HSV色彩空间进行分析
            if all_border_pixels.shape[1] == 3:
                # RGB图像
                border_rgb = all_border_pixels.reshape(-1, 1, 3).astype(np.uint8)
                border_hsv = cv2.cvtColor(border_rgb, cv2.COLOR_RGB2HSV)
                border_hsv = border_hsv.reshape(-1, 3)
                
                saturation = border_hsv[:, 1]
                value = border_hsv[:, 2]
                
                # 检测空白/背景像素：
                # 1. 低饱和度（中性色/灰白色）
                # 2. 或高亮度（接近白色）
                # 3. 或低亮度（接近黑色，但排除真正的图案黑色）
                is_blank = (
                    (saturation < 30) |  # 低饱和度（灰白色）
                    (value > 240) |      # 高亮度（接近白色）
                    (value < 30)         # 低亮度（接近黑色，可能是背景）
                )
            else:
                # 灰度图
                gray_values = all_border_pixels.flatten()
                # 空白像素：接近白色或接近黑色
                is_blank = (gray_values > 240) | (gray_values < 30)

            # 计算空白像素占比
            blank_count = np.sum(is_blank)
            border_ratio = blank_count / len(is_blank) if len(is_blank) > 0 else 0.0

            return float(np.clip(border_ratio, 0.0, 1.0))

        except Exception as e:
            # 如果分析失败，返回1.0（表示全是边界，最保守）
            return 1.0

    def analyze_content(self, image_path: Path) -> ContentAnalysisResult:
        """
        综合分析图片内容

        Args:
            image_path: 图片路径

        Returns:
            ContentAnalysisResult: 内容分析结果
        """
        content_ratio = self.calculate_content_ratio(image_path)
        border_ratio = self.detect_border_region(image_path)

        # 判断是否有效
        # 有效条件：内容占比 >= 阈值 且 边界占比 <= 阈值
        is_valid = (
            content_ratio >= self.min_content_ratio and
            border_ratio <= self.max_border_ratio
        )

        return ContentAnalysisResult(
            content_ratio=content_ratio,
            border_ratio=border_ratio,
            is_valid=is_valid,
        )

