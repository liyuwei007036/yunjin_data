"""图片智能分析模块 - 自动识别图片类型并选择最佳提取策略."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from PIL import Image


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
        """
        from .vlm_recognizer import VLMRecognizer

        self.vlm_recognizer = VLMRecognizer(
            model_name=vlm_model or "Qwen/Qwen2-VL-7B-Instruct"
        )

    def analyze(self, image: Image.Image) -> ImageAnalysis:
        """分析图片，返回类型和建议.

        Args:
            image: 输入图片 (RGB模式)

        Returns:
            ImageAnalysis: 分析结果

        Raises:
            RuntimeError: 当 VLM 识别失败时
        """
        categories = self.vlm_recognizer.recognize(image)

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
