"""Grounded SAM 分割策略 - 使用 Grounding DINO + SAM 精确分割物体/图案.

用于开放词汇分割，从图片中提取完整的图案区域用于训练.
注意: 不使用降级函数，必须有模型才能正常工作.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from ..image_analyzer import ImageAnalysis

logger = logging.getLogger(__name__)


class GroundedSAMStrategy:
    """Grounded SAM 分割策略.

    结合 Grounding DINO (开放词汇检测) 和 SAM (精确分割)，
    使用 VLM 识别的物体类别作为提示词，精确分割图案区域.
    """

    def __init__(
        self,
        grounding_dino_model: str,
        sam_model: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        device: str = "cuda",
    ):
        """初始化 Grounded SAM 策略.

        Args:
            grounding_dino_model: Grounding DINO 模型路径
            sam_model: SAM 模型路径
            box_threshold: 检测框置信度阈值
            text_threshold: 文本匹配阈值
            device: 运行设备

        Raises:
            RuntimeError: 当模型加载失败时
        """
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

        self._load_models(grounding_dino_model, sam_model)

    def _load_models(self, grounding_dino_model: str, sam_model: str):
        """加载 Grounding DINO 和 SAM 模型.

        Raises:
            RuntimeError: 当模型文件不存在或加载失败时
        """
        # 检查模型文件是否存在
        dino_path = Path(grounding_dino_model)
        sam_path = Path(sam_model)

        if not dino_path.exists():
            error_msg = f"Grounding DINO 模型文件不存在: {grounding_dino_model}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not sam_path.exists():
            error_msg = f"SAM 模型文件不存在: {sam_model}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"加载 Grounding DINO: {grounding_dino_model}")
        logger.info(f"加载 SAM: {sam_model}")

        try:
            from grounding_dino import GroundingDINO
            from segment_anything import sam_model_registry

            self.grounding_model = GroundingDINO(
                model_type="grounding_dino_s", device=self.device
            )
            self.grounding_model.load_checkpoint(grounding_dino_model)

            self.sam_model = sam_model_registry["vit_l"](
                checkpoint=sam_model, device=self.device
            )

            logger.info("模型加载成功")
        except Exception as e:
            error_msg = f"模型加载失败: {type(e).__name__}: {e}"
            logger.error(error_msg)
            raise RuntimeError(
                f"{error_msg}\n"
                f"请检查模型文件是否完整且兼容。"
            ) from e

    def extract(
        self,
        image: Image.Image,
        analysis: ImageAnalysis,
        target_size: int = 1024,
        min_crop_size: int = 256,
    ) -> List[Image.Image]:
        """分割图案区域.

        Args:
            image: 输入图片
            analysis: 图片分析结果 (包含 VLM 识别的物体类别)
            target_size: 目标输出尺寸
            min_crop_size: 最小裁剪尺寸

        Returns:
            分割后的图案列表
        """
        img_array = np.array(image)

        categories = analysis.categories
        if not categories:
            logger.warning("没有可识别的物体类别")
            return []

        prompt = self._build_prompt(categories)
        logger.debug(f"构建提示词: {prompt}")

        detections = self.grounding_model.predict_with_classes(
            image=img_array,
            classes=[prompt],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        logger.debug(f"检测到 {len(detections.xyxy)} 个目标")

        from segment_anything import SamPredictor

        predictor = SamPredictor(self.sam_model)
        predictor.set_image(img_array)

        all_masks = []
        for i, box in enumerate(detections.xyxy):
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=True,
            )

            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            logger.debug(f"目标 {i}: 最佳分数={scores[best_idx]:.3f}")

            mask = self._postprocess_mask(mask, min_crop_size)

            if mask is not None:
                all_masks.append(mask)

        logger.debug(f"分割完成，有效掩码: {len(all_masks)}")

        patterns = []
        for mask in all_masks:
            pattern = self._mask_to_image(img_array, mask)
            pattern = self._resize_pattern(pattern, target_size)
            patterns.append(pattern)

        return patterns

    def _build_prompt(self, categories: List[str]) -> str:
        """构建 Grounding DINO 提示词.

        格式: "category1 . category2 . category3 ."

        Args:
            categories: 物体类别列表

        Returns:
            Grounding DINO 格式的提示词
        """
        return " . ".join(categories) + " ."

    def _postprocess_mask(
        self, mask: np.ndarray, min_size: int
    ) -> Optional[np.ndarray]:
        """后处理掩码."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )

        valid_masks = []
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if bw >= min_size and bh >= min_size:
                valid_masks.append(mask)

        if not valid_masks:
            return None

        combined = np.zeros_like(mask, dtype=np.uint8)
        for m in valid_masks:
            combined = np.logical_or(combined, m)

        return combined.astype(np.uint8)

    def _mask_to_image(
        self, img_array: np.ndarray, mask: np.ndarray
    ) -> Image.Image:
        """将掩码转换为透明背景图片."""
        h, w = mask.shape

        if mask.shape[:2] != img_array.shape[:2]:
            mask = cv2.resize(mask, (w, h))

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = img_array[:h, :w]
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)

        return Image.fromarray(rgba, mode="RGBA")

    def _resize_pattern(self, pattern: Image.Image, target_size: int) -> Image.Image:
        """缩放到目标尺寸."""
        return pattern.resize(
            (target_size, target_size), Image.Resampling.LANCZOS
        )
