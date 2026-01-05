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
        
        使用颜色分析和边缘检测的组合方法，识别图片中的有效图案区域。
        有效内容指非空白、非背景的图案区域（如云锦图案）。

        Args:
            image_path: 图片路径

        Returns:
            float: 有效内容占比（0.0-1.0），0.0表示全是空白，1.0表示全是有效内容
        """
        try:
            # 使用PIL读取图片（更可靠，支持中文路径）
            with Image.open(image_path) as pil_img:
                # 转换为RGB模式（统一格式，便于后续处理）
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # 转换为numpy数组（用于OpenCV处理）
                img_array = np.array(pil_img)
            
            if img_array.size == 0:
                return 0.0

            h, w = img_array.shape[:2]
            total_pixels = h * w

            # ========== 方法1: 颜色分析（HSV色彩空间） ==========
            # 目的：检测非中性色（有颜色的）区域，排除灰白色背景
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # 提取饱和度和亮度通道
            saturation = hsv[:, :, 1]  # 饱和度：0-255，越高颜色越鲜艳
            value = hsv[:, :, 2]        # 亮度：0-255，越高越亮
            
            # 检测非中性色像素（有效内容）
            # 条件1：饱和度 > 30（排除低饱和度的灰白色）
            # 条件2：亮度在合理范围内（20-240，排除过暗或过亮的区域）
            # 满足条件的像素被认为是有效内容（有颜色的图案）
            non_neutral_mask = (saturation > 30) & (value > 20) & (value < 240)
            color_content_ratio = np.sum(non_neutral_mask) / total_pixels

            # ========== 方法2: 边缘检测 ==========
            # 目的：检测有图案纹理的区域，即使颜色单一也能识别
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 使用Canny边缘检测（阈值50-150）
            # 检测图片中的边缘（图案边界、纹理等）
            edges = cv2.Canny(gray, 50, 150)
            
            # 计算边缘密度（边缘像素占比）
            edge_pixels = np.sum(edges > 0)
            edge_ratio = edge_pixels / total_pixels
            
            # 使用形态学操作（膨胀）识别图案区域
            # 目的：连接附近的边缘，形成连续的图案区域
            # 方法：对边缘进行膨胀操作（3x3核，迭代2次）
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # 计算有边缘的区域占比（膨胀后的边缘区域）
            edge_content_ratio = np.sum(dilated > 0) / total_pixels

            # ========== 组合两种方法 ==========
            # 取两种方法的最大值（更宽松的策略）
            # 原因：有些图案可能颜色单一但边缘明显，或颜色丰富但边缘较少
            # 使用最大值可以确保只要有一种方法检测到内容，就认为有有效内容
            # 例如：color_ratio=0.2, edge_ratio=0.5 -> content_ratio=0.5
            content_ratio = max(color_content_ratio, edge_content_ratio)

            # 确保结果在0.0-1.0范围内
            return float(np.clip(content_ratio, 0.0, 1.0))

        except Exception as e:
            # 如果分析失败（图片损坏、格式不支持等），返回0（表示无有效内容）
            return 0.0

    def detect_border_region(self, image_path: Path) -> float:
        """
        检测图片边界空白区域占比（适用于所有图片）
        
        分析图片四边边缘区域（默认各15%宽度），检测这些区域是否主要是空白/背景。
        用于识别边界区域、拼接过渡区域等不适合训练的图片。

        Args:
            image_path: 图片路径

        Returns:
            float: 边界空白占比（0.0-1.0），0.0表示边界全是有效内容，1.0表示边界全是空白
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
            
            # ========== 步骤1: 计算边界区域尺寸 ==========
            # 根据配置的边界区域宽度比例（默认15%）计算边界区域的实际像素尺寸
            # 例如：512x512图片，15% = 约77像素
            border_h = max(1, int(h * self.border_region_width))  # 上下边界高度
            border_w = max(1, int(w * self.border_region_width))    # 左右边界宽度

            # ========== 步骤2: 提取四边区域 ==========
            # 从图片中提取四个边缘区域（上、下、左、右）
            top = img_array[0:border_h, :]                    # 上边界：从顶部开始的 border_h 行
            bottom = img_array[h-border_h:h, :]               # 下边界：从底部开始的 border_h 行
            left = img_array[:, 0:border_w]                   # 左边界：从左侧开始的 border_w 列
            right = img_array[:, w-border_w:w]                # 右边界：从右侧开始的 border_w 列

            # ========== 步骤3: 合并所有边界区域 ==========
            # 收集所有有效的边界区域（非空）
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
                return 0.0  # 如果没有边界区域，认为边界全是有效内容

            # ========== 步骤4: 将边界区域转换为像素列表 ==========
            # 将所有边界区域的像素展平为一维数组，便于统一分析
            border_pixels_list = []
            for region in border_regions:
                if region.ndim == 3:
                    # RGB图像：reshape为 (像素数, 3) 的形状
                    pixels = region.reshape(-1, 3)
                else:
                    # 灰度图：reshape为 (像素数, 1) 的形状
                    pixels = region.reshape(-1, 1)
                border_pixels_list.append(pixels)
            
            if not border_pixels_list:
                return 0.0

            # 合并所有边界像素
            all_border_pixels = np.vstack(border_pixels_list)

            # ========== 步骤5: 检测空白/背景像素 ==========
            # 使用HSV色彩空间分析，识别空白/背景像素
            if all_border_pixels.shape[1] == 3:
                # RGB图像：转换为HSV色彩空间
                border_rgb = all_border_pixels.reshape(-1, 1, 3).astype(np.uint8)
                border_hsv = cv2.cvtColor(border_rgb, cv2.COLOR_RGB2HSV)
                border_hsv = border_hsv.reshape(-1, 3)
                
                saturation = border_hsv[:, 1]  # 饱和度通道
                value = border_hsv[:, 2]        # 亮度通道
                
                # 检测空白/背景像素的条件（满足任一条件即认为是空白）：
                # 1. 低饱和度（< 30）：中性色/灰白色，通常是背景
                # 2. 高亮度（> 240）：接近白色，通常是空白背景
                # 3. 低亮度（< 30）：接近黑色，可能是背景（注意：可能误判真正的黑色图案）
                is_blank = (
                    (saturation < 30) |  # 条件1：低饱和度（灰白色背景）
                    (value > 240) |      # 条件2：高亮度（白色背景）
                    (value < 30)         # 条件3：低亮度（黑色背景）
                )
            else:
                # 灰度图：直接使用灰度值判断
                gray_values = all_border_pixels.flatten()
                # 空白像素：接近白色（> 240）或接近黑色（< 30）
                is_blank = (gray_values > 240) | (gray_values < 30)

            # ========== 步骤6: 计算边界空白占比 ==========
            # 统计空白像素数量，计算占比
            blank_count = np.sum(is_blank)
            border_ratio = blank_count / len(is_blank) if len(is_blank) > 0 else 0.0

            return float(np.clip(border_ratio, 0.0, 1.0))

        except Exception as e:
            # 如果分析失败，返回1.0（表示全是边界，最保守）
            return 1.0

    def analyze_content(self, image_path: Path) -> ContentAnalysisResult:
        """
        综合分析图片内容
        
        同时计算内容占比和边界占比，并判断图片是否有效（满足质量要求）。
        这是内容分析的主入口方法，整合了所有内容检查功能。

        Args:
            image_path: 图片路径

        Returns:
            ContentAnalysisResult: 内容分析结果，包含：
                - content_ratio: 有效内容占比（0.0-1.0）
                - border_ratio: 边界空白占比（0.0-1.0）
                - is_valid: 是否有效（基于配置的阈值判断）
        """
        # 计算有效内容占比（使用颜色分析和边缘检测）
        content_ratio = self.calculate_content_ratio(image_path)
        # 检测边界空白区域占比（分析图片四边区域）
        border_ratio = self.detect_border_region(image_path)

        # 判断图片是否有效（满足质量要求）
        # 有效条件：
        #   1. 内容占比 >= min_content_ratio（有足够的有效内容）
        #   2. 边界占比 <= max_border_ratio（边界区域不能太多空白）
        # 只有同时满足两个条件，图片才被认为是有效的
        is_valid = (
            content_ratio >= self.min_content_ratio and
            border_ratio <= self.max_border_ratio
        )

        return ContentAnalysisResult(
            content_ratio=content_ratio,
            border_ratio=border_ratio,
            is_valid=is_valid,
        )

