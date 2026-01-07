"""配置管理."""

from dataclasses import dataclass
from typing import Optional


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
    vlm_model: str = "Qwen/Qwen2-VL-7B-Instruct"

    # Grounded SAM 模型配置 (用于纹样分割)
    grounding_dino_path: str = "models/grounding_dino/grounding_dino_swin-t_ogc.pth"
    sam_model_path: str = "models/sam/sam_vit_l_0b3195.pth"
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    sam_mask_threshold: float = 0.5

    # 输出限制
    max_patterns_per_image: int = 20

    # 质量过滤
    min_quality_score: float = 10.0
    min_content_ratio: float = 0.1

    @classmethod
    def from_args(cls, args) -> "Config":
        """从命令行参数创建配置."""
        return cls(
            input_dir=args.input,
            output_dir=args.output,
            target_size=getattr(args, "target_size", 1024),
        )
