"""
图片水印去除工具
支持多种去水印方法：颜色覆盖、Inpainting、深度学习修复
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


class WatermarkRemover:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def detect_dark_text_regions(
        self,
        image: np.ndarray,
        threshold: int = 80,
        min_area: int = 500,
        roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    ) -> np.ndarray:
        """
        检测深色文字区域（适合黑色水印）
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 如果指定了ROI，只处理该区域
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        # 使用自适应阈值处理
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 创建全尺寸mask
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 如果有ROI，将处理结果放回原位置
        if roi is not None:
            x, y, w, h = roi
            full_mask[y:y+h, x:x+w] = thresh
            thresh = full_mask

        # 查找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        mask = np.zeros_like(thresh)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                mask[labels == i] = 255

        return mask

    def detect_white_text_regions(
        self,
        image: np.ndarray,
        brightness_threshold: int = 200,
        min_area: int = 100
    ) -> np.ndarray:
        """
        检测高亮区域（适合白色水印）
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 创建高亮区域mask
        _, mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

        # 去除噪点
        mask = cv2.medianBlur(mask, 5)

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 查找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        # 创建最终mask
        final_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                final_mask[labels == i] = 255

        return final_mask

    def apply_inpainting(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        method: str = "telea",
        radius: int = 3
    ) -> np.ndarray:
        """
        应用图像修复算法

        Args:
            image: 输入图片
            mask: 修复区域mask
            method: "telea" 或 "ns" (Navier-Stokes)
            radius: 邻域半径
        """
        if method == "telea":
            return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
        else:
            return cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)

    def content_aware_fill(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        search_window: int = 50
    ) -> np.ndarray:
        """
        使用纹理合成填充（适合重复纹理）
        """
        result = image.copy()
        mask_indices = np.where(mask > 0)

        if len(mask_indices[0]) == 0:
            return result

        # 对mask区域进行逐像素填充
        for y, x in zip(mask_indices[0], mask_indices[1]):
            # 在周围寻找相似区域
            y_min = max(0, y - search_window)
            y_max = min(image.shape[0], y + search_window)
            x_min = max(0, x - search_window)
            x_max = min(image.shape[1], x + search_window)

            # 计算最佳匹配块（简化版）
            best_match = None
            best_score = float('inf')

            # 在周围区域采样
            for ry in range(y_min, y_max, 10):
                for rx in range(x_min, x_max, 10):
                    patch = image[ry:ry+15, rx:rx+15]
                    if patch.shape[:2] == (15, 15):
                        # 计算与当前像素的差异
                        center_color = image[y, x]
                        patch_center = patch[7, 7]
                        score = np.sum(np.abs(center_color.astype(int) - patch_center.astype(int)))

                        if score < best_score:
                            best_score = score
                            best_match = patch

            if best_match is not None:
                result[y-7:y+8, x-7:x+8] = best_match

        return result

    def remove_watermark_combined(
        self,
        image: np.ndarray,
        iterations: int = 2,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        组合方法去除水印（深色+高亮）
        """
        result = image.copy()

        for _ in range(iterations):
            # 检测深色文字
            dark_mask = self.detect_dark_text_regions(result, threshold=80, min_area=300, roi=roi)

            # 检测高亮文字
            white_mask = self.detect_white_text_regions(result, brightness_threshold=200, min_area=100)

            # 合并mask
            combined_mask = cv2.bitwise_or(dark_mask, white_mask)

            # 膨胀mask以覆盖完整水印
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

            # 应用修复 - 使用多次迭代
            for _ in range(3):
                result = cv2.inpaint(result, combined_mask, 10, cv2.INPAINT_NS)

        return result

    def remove_by_texture_copy(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        search_radius: int = 100
    ) -> np.ndarray:
        """
        使用周围纹理复制填充（比inpaint效果好）
        """
        result = image.copy()
        mask_indices = np.where(mask > 0)

        if len(mask_indices[0]) == 0:
            return result

        # 对mask区域进行逐块填充
        for y, x in zip(mask_indices[0], mask_indices[1]):
            # 跳过边界像素
            if y - 20 < 0 or y + 20 >= image.shape[0] or x - 20 < 0 or x + 20 >= image.shape[1]:
                continue

            # 在周围寻找相似纹理区域
            best_match = None
            best_score = float('inf')

            # 在多个方向搜索
            for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                rad = np.radians(angle)
                for dist in range(30, search_radius, 20):
                    sx = int(x + dist * np.cos(rad))
                    sy = int(y + dist * np.sin(rad))

                    if (sx - 15 < 0 or sx + 15 >= image.shape[1] or
                        sy - 15 < 0 or sy + 15 >= image.shape[0]):
                        continue

                    # 提取候选块
                    patch = image[sy-15:sy+15, sx-15:sx+15]
                    patch_mask = mask[sy-15:sy+15, sx-15:sx+15]

                    # 只使用没有水印的区域
                    if np.sum(patch_mask) > 0:
                        continue

                    # 计算与当前像素的颜色差异
                    current_block = image[max(0,y-15):y+15, max(0,x-15):x+15]
                    if current_block.shape != patch.shape:
                        continue

                    score = np.sum(np.abs(current_block.astype(float) - patch.astype(float)))

                    if score < best_score:
                        best_score = score
                        best_match = patch.copy()

            if best_match is not None:
                # 平滑边缘
                result[y-15:y+15, x-15:x+15] = best_match

        return result

    def remove_watermark_texture(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        使用纹理复制方法去除水印（效果更好）
        """
        result = image.copy()

        # 获取水印mask
        dark_mask = self.detect_dark_text_regions(result, threshold=80, min_area=300, roi=roi)
        white_mask = self.detect_white_text_regions(result, brightness_threshold=200, min_area=100)
        combined_mask = cv2.bitwise_or(dark_mask, white_mask)

        # 膨胀
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

        # 先用inpaint做基础修复
        result = cv2.inpaint(result, combined_mask, 15, cv2.INPAINT_NS)

        # 再用纹理复制精细修复
        result = self.remove_by_texture_copy(result, combined_mask, search_radius=150)

        return result

    def remove_right_bottom_watermark(
        self,
        image: np.ndarray,
        margin_right: int = 20,
        margin_bottom: int = 20,
        watermark_width: int = 150,
        watermark_height: int = 50
    ) -> np.ndarray:
        """
        去除右下角水印
        """
        h, w = image.shape[:2]

        # 计算ROI位置（右下角）
        roi_x = w - watermark_width - margin_right
        roi_y = h - watermark_height - margin_bottom
        roi_w = watermark_width
        roi_h = watermark_height

        print(f"水印区域: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

        return self.remove_watermark_combined(
            image,
            iterations=3,
            roi=(roi_x, roi_y, roi_w, roi_h)
        )

    def remove_watermark_aggressive(
        self,
        image: np.ndarray,
        dilate_iterations: int = 3
    ) -> np.ndarray:
        """
        激进去水印方法 - 多次迭代+大kernel膨胀
        """
        result = image.copy()

        # 检测所有非背景区域（水印区域）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用Canny边缘检测找边缘
        edges = cv2.Canny(gray, 50, 150)

        # 膨胀边缘
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=dilate_iterations)

        # 填充空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 应用修复
        result = self.apply_inpainting(result, mask, method="ns", radius=10)

        return result

    def detect_and_remove_watermark_2(
        self,
        image: np.ndarray,
        x: int, y: int, width: int, height: int
    ) -> np.ndarray:
        """
        手动指定区域检测并去除
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 在指定区域进行阈值处理
        roi = gray[y:y+height, x:x+width]
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 创建mask
        mask = np.zeros_like(gray)
        mask[y:y+height, x:x+width] = thresh

        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 修复
        result = self.apply_inpainting(image, mask, method="ns", radius=5)
        return result

    def process_image(
        self,
        image_path: str,
        method: str = "right_bottom"
    ) -> str:
        """
        处理单张图片
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            return ""

        h, w = image.shape[:2]
        print(f"图片尺寸: {w}x{h}")

        if method == "right_bottom":
            # 计算右下角ROI
            roi_x = w - 200 - 20
            roi_y = h - 60 - 30
            roi = (roi_x, roi_y, 200, 60)
            print(f"水印区域: x={roi_x}, y={roi_y}, w=200, h=60")
            result = self.remove_watermark_texture(image, roi=roi)
        elif method == "texture":
            result = self.remove_watermark_texture(image, roi=None)
        elif method == "combined":
            result = self.remove_watermark_combined(image, iterations=3)
        elif method == "aggressive":
            result = self.remove_watermark_aggressive(image, dilate_iterations=2)
        elif method == "ns":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.medianBlur(mask, 3)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            result = cv2.inpaint(image, mask, 5, cv2.INPAINT_NS)
        else:
            result = image

        # 保存结果
        output_path = self.output_dir / Path(image_path).name
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return str(output_path)

    def process_with_roi(
        self,
        image_path: str,
        roi_list: List[dict]
    ) -> str:
        """
        处理单张图片 - 手动指定ROI区域
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            return ""

        result = image.copy()

        for roi in roi_list:
            x = roi.get('x', 0)
            y = roi.get('y', 0)
            w = roi.get('width', 100)
            h = roi.get('height', 50)

            # 在ROI区域创建mask
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[y:y+h, x:x+w]

            # OTSU阈值
            _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 创建mask
            mask = np.zeros_like(gray)
            mask[y:y+h, x:x+w] = thresh

            # 形态学处理
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # 多次修复
            for _ in range(2):
                result = cv2.inpaint(result, mask, 5, cv2.INPAINT_NS)

        # 保存结果
        output_path = self.output_dir / Path(image_path).name
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return str(output_path)

    def batch_process(
        self,
        method: str = "combined"
    ) -> dict:
        """
        批量处理图片
        """
        results = {'success': [], 'failed': []}

        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = sorted([f for f in self.input_dir.iterdir()
                             if f.suffix.lower() in extensions])

        for image_path in image_files:
            print(f"处理: {image_path.name}")
            output = self.process_image(str(image_path), method)
            if output:
                results['success'].append(image_path.name)
            else:
                results['failed'].append(image_path.name)

        return results


def main():
    input_dir = r"C:\Users\booster\Desktop\images"
    output_dir = r"C:\Users\booster\Desktop\images\no_watermark"

    remover = WatermarkRemover(input_dir, output_dir)

    print("=" * 50)
    print("图片水印去除工具 v2.0 (纹理复制法)")
    print("=" * 50)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 50)

    # 方法选择：
    # "right_bottom" - 专门去除右下角水印（默认）
    # "texture" - 全图纹理复制填充（适合任意位置）
    # "combined" - 组合方法

    print("\n使用右下角水印去除方法（纹理复制）...")
    results = remover.batch_process(method="right_bottom")

    print(f"\n处理完成!")
    print(f"成功: {len(results['success'])} 张")
    print(f"失败: {len(results['failed'])} 张")

    if results['failed']:
        print(f"失败列表: {results['failed']}")


if __name__ == "__main__":
    main()
