"""图片智能分析模块 - 自动识别图片类型并选择最佳提取策略."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class ImageType(Enum):
    """图片类型."""

    PATTERN_SEGMENT = "pattern_segment"  # 图案分割 (通用物体/图案)


@dataclass
class ImageAnalysis:
    """图片分析结果."""

    type: ImageType
    categories: List[str] = field(default_factory=list)  # VLM 识别的物体类别
    num_objects: int = 0
    texture_density: float = 0.0
    regularity: float = 0.0
    suggested_strategy: str = "grounded_sam"
    optimal_window_size: int = 512


class ImageAnalyzer:
    """图片智能分析器.

    使用 VLM 识别图片中的物体/图案，配合 Grounded SAM 精确分割.
    注意: 不使用降级函数，必须有本地模型才能正常工作.
    """

    def __init__(
        self,
        vlm_model: Optional[str] = None,
    ):
        """初始化分析器.

        Args:
            vlm_model: VLM 模型名称或路径

        Raises:
            RuntimeError: 当模型加载失败时
        """
        from .vlm_recognizer import VLMRecognizer
        from pathlib import Path

        model_name = vlm_model or "Qwen/Qwen2-VL-7B-Instruct"
        logger.info(f"初始化 VLM 识别器: {model_name}")

        try:
            self.vlm_recognizer = VLMRecognizer(model_name=model_name)
            logger.info("VLM 模型加载成功")
        except Exception as e:
            logger.error(f"VLM 模型加载失败: {type(e).__name__}: {e}")
            raise RuntimeError(
                f"无法加载 VLM 模型 '{model_name}': {e}\n"
                f"请确保模型已正确安装，或检查模型路径是否正确。"
            ) from e

    def analyze(self, image: Image.Image) -> ImageAnalysis:
        """分析图片，返回类型和建议.

        Args:
            image: 输入图片 (RGB模式)

        Returns:
            ImageAnalysis: 分析结果

        Raises:
            RuntimeError: 当 VLM 识别失败时
        """
        logger.debug("开始分析图片...")
        categories = self.vlm_recognizer.recognize(image)
        logger.debug(f"VLM 识别到 {len(categories)} 个类别: {categories}")

        w, h = image.size
        min_dim = min(w, h)

        return ImageAnalysis(
            type=ImageType.PATTERN_SEGMENT,
            categories=categories,
            num_objects=len(categories),
            texture_density=0.0,
            regularity=0.0,
            suggested_strategy="grounded_sam",
            optimal_window_size=min(min_dim, 1024),
        )
