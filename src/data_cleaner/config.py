"""
Configuration module for cloud-brocade-cleaner.

云锦图片清洗模块配置。
支持 YAML 配置文件加载。
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ==================== Default Paths ====================

# 项目根目录（当前文件所在目录的父目录）
PROJECT_ROOT = Path(__file__).parent.parent

# 配置文件路径
CONFIG_FILE = PROJECT_ROOT / "data_cleaner" / "config.yaml"

# 清洗结果输出目录（默认）
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

# 清洗后图片输出子目录
CLEANED_OUTPUT_DIR = "cleaned"


# ==================== YAML 配置加载 ====================

def load_yaml_config(config_path: Optional[Path] = None) -> dict:
    """
    从 YAML 文件加载配置

    Args:
        config_path: 配置文件路径，默认使用 CONFIG_FILE

    Returns:
        dict: 配置字典
    """
    import yaml

    path = config_path or CONFIG_FILE
    if path is None or not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_value, base_path: Path = PROJECT_ROOT) -> Path:
    """
    解析路径配置值

    Args:
        path_value: 路径值（字符串或Path）
        base_path: 基准路径

    Returns:
        Path: 解析后的绝对路径
    """
    if path_value is None:
        return None

    if isinstance(path_value, Path):
        return path_value

    path = Path(path_value)

    # 如果是相对路径，基于项目根目录解析
    if not path.is_absolute():
        return base_path / path

    return path

# ==================== Quality Check Configuration ====================

@dataclass
class QualityConfig:
    """图片质量检查配置"""

    # 最小分辨率要求（SD1.5推荐最小尺寸）
    min_width: int = 512
    min_height: int = 512

    # 模糊度阈值（Laplacian方差，越高越清晰）
    blur_threshold: float = 50.0

    # 是否检查Alpha通道
    check_alpha_channel: bool = True

    # 是否检查图片损坏
    check_corruption: bool = True


# ==================== Style Classification Configuration ====================

@dataclass
class StyleConfig:
    """云锦风格分类配置"""

    # 云锦置信度阈值，大于此值认为是云锦
    style_threshold: float = 0.65

    # 人工复核阈值，小于此值直接拒绝
    review_threshold: float = 0.40

    # CLIP模型名称
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # 是否使用GPU
    use_gpu: bool = True

    # 批量推理大小
    batch_size: int = 32


# ==================== Cleaner Configuration ====================

@dataclass
class CleanerConfig:
    """数据清洗主配置"""

    # 质量检查配置
    quality: QualityConfig = field(default_factory=QualityConfig)

    # 风格分类配置
    style: StyleConfig = field(default_factory=StyleConfig)

    # 输入目录（爬虫输出的原始图片目录）
    input_dir: Path = None

    # 输出目录（清洗后图片存放目录）
    output_dir: Path = None

    # 处理模式：quality(仅质量)、style(仅风格)、full(完整)
    mode: str = "full"

    # 是否生成清洗报告
    generate_report: bool = True

    def __post_init__(self):
        if self.input_dir is None:
            self.input_dir = DEFAULT_OUTPUT_DIR
        if self.output_dir is None:
            self.output_dir = Path(CLEANED_OUTPUT_DIR)


# ==================== 配置加载函数 ====================

def load_config(config_path: Optional[Path] = None) -> CleanerConfig:
    """
    从 YAML 配置文件加载完整配置

    Args:
        config_path: 配置文件路径，默认使用 CONFIG_FILE

    Returns:
        CleanerConfig: 清洗配置对象
    """
    yaml_config = load_yaml_config(config_path)

    # 加载质量检查配置
    quality_data = yaml_config.get("quality", {})
    quality_config = QualityConfig(
        min_width=quality_data.get("min_width", 512),
        min_height=quality_data.get("min_height", 512),
        blur_threshold=quality_data.get("blur_threshold", 50.0),
        check_alpha_channel=quality_data.get("check_alpha_channel", True),
        check_corruption=quality_data.get("check_corruption", True),
    )

    # 加载风格分类配置
    style_data = yaml_config.get("style", {})
    style_config = StyleConfig(
        style_threshold=style_data.get("style_threshold", 0.65),
        review_threshold=style_data.get("review_threshold", 0.40),
        clip_model_name=style_data.get("clip_model_name", "openai/clip-vit-base-patch32"),
        use_gpu=style_data.get("use_gpu", True),
        batch_size=style_data.get("batch_size", 32),
    )

    # 加载基础配置
    input_dir = resolve_path(yaml_config.get("input_dir", "output"))
    output_dir = resolve_path(yaml_config.get("output_dir", "output/cleaned"))

    return CleanerConfig(
        quality=quality_config,
        style=style_config,
        input_dir=input_dir,
        output_dir=output_dir,
        mode=yaml_config.get("mode", "full"),
        generate_report=yaml_config.get("generate_report", True),
    )


# ==================== Output Subdirectories ====================

# 这些是相对于 output_dir 的子目录名（字符串）
DEFAULT_OUTPUT_SUBDIRS = {
    "approved": "approved",
    "rejected_quality": "rejected_quality",
    "rejected_style": "rejected_style",
    "review": "review",
    "reports": "reports",
}


def get_output_subdirs(config_path: Optional[Path] = None) -> dict:
    """
    获取输出子目录配置

    Args:
        config_path: 配置文件路径

    Returns:
        dict: 输出子目录配置
    """
    yaml_config = load_yaml_config(config_path)
    subdirs_config = yaml_config.get("output_subdirs", {})

    # 合并默认配置
    result = DEFAULT_OUTPUT_SUBDIRS.copy()
    result.update(subdirs_config)
    return result


def get_output_dir_path(subdir_name: str, base_output_dir: Path) -> Path:
    """
    获取完整的输出子目录路径

    Args:
        subdir_name: 子目录名
        base_output_dir: 基础输出目录

    Returns:
        Path: 完整的子目录路径
    """
    subdirs = get_output_subdirs()
    subdir = subdirs.get(subdir_name, subdir_name)
    return base_output_dir / subdir


# ==================== Prompt Templates for CLIP ====================

# 云锦风格正向提示词
CLOUC_BROCADE_PROMPT = (
    "a photo of Chinese cloud brocade (yunjin) traditional textile pattern, "
    "intricate silk brocade with gold thread, traditional Chinese embroidery, "
    "luxurious patterned fabric with cloud motifs"
)

# 负向提示词（非云锦）
NOT_BROCADE_PROMPT = (
    "a photo of modern object, not traditional textile, "
    "not silk brocade, not Chinese traditional pattern, "
    "computer screen, furniture, building, food, vehicle"
)
