"""配置管理."""

import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


def _setup_logging():
    """初始化日志配置."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


_setup_logging()


@dataclass
class Config:
    """图片清洗配置."""

    # 输入输出
    input_dir: str = "output/images"
    output_dir: str = "output/patterns"

    # 目标尺寸
    target_size: int = 1024
    min_crop_size: int = 512

    # 设备配置
    device: Optional[str] = None  # 自动检测

    # VLM 模型配置 (用于纹样识别)
    use_vlm: bool = True
    vlm_model: str = ""

    # Grounded SAM 模型配置 (用于纹样分割)
    grounding_dino_path: str = ""
    sam_model_path: str = ""
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    sam_mask_threshold: float = 0.5

    # 输出限制
    max_patterns_per_image: int = 20

    # 质量过滤
    min_quality_score: float = 10.0
    min_content_ratio: float = 0.1

    # 质量分数权重 (可配置化)
    quality_sharpness_weight: float = 0.5
    quality_content_weight: float = 0.5

    # 支持的图片格式
    supported_image_formats: List[str] = field(
        default_factory=lambda: ["png", "jpg", "jpeg", "webp", "bmp"]
    )

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """从 YAML 文件加载配置."""
        import yaml
        from pathlib import Path

        if not Path(config_path).exists():
            logger.warning(f"配置文件不存在: {config_path}")
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            input_dir=data.get("input_dir", "output/images"),
            output_dir=data.get("output_dir", "output/patterns"),
            target_size=data.get("target_size", 1024),
            min_crop_size=data.get("min_crop_size", 512),
            max_patterns_per_image=data.get("max_patterns_per_image", 20),
            use_vlm=data.get("use_vlm", True),
            vlm_model=data.get("vlm_model", "Qwen/Qwen2-VL-7B-Instruct"),
            grounding_dino_path=data.get("grounding_dino_path", "models/grounding_dino/grounding_dino_swin-t_ogc.pth"),
            sam_model_path=data.get("sam_model_path", "models/sam/sam_vit_l_0b3195.pth"),
            box_threshold=data.get("box_threshold", 0.25),
            text_threshold=data.get("text_threshold", 0.25),
            sam_mask_threshold=data.get("sam_mask_threshold", 0.5),
            min_quality_score=data.get("min_quality_score", 10.0),
            min_content_ratio=data.get("min_content_ratio", 0.1),
            quality_sharpness_weight=data.get("quality_sharpness_weight", 0.5),
            quality_content_weight=data.get("quality_content_weight", 0.5),
            supported_image_formats=data.get("supported_image_formats", ["png", "jpg", "jpeg", "webp", "bmp"]),
        )
