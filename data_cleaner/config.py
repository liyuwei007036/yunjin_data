"""
Configuration module for cloud-brocade-cleaner.

云锦图片清洗模块配置。
"""

from pathlib import Path
from dataclasses import dataclass


# ==================== Default Paths ====================

# 项目根目录（当前文件所在目录的父目录）
PROJECT_ROOT = Path(__file__).parent.parent

# 输出目录（相对于项目根目录）
OUTPUT_DIR = PROJECT_ROOT / "output"

# 清洗结果输出目录
CLEANED_OUTPUT_DIR = OUTPUT_DIR / "cleaned"

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
    quality: QualityConfig = None

    # 风格分类配置
    style: StyleConfig = None

    # 输入目录（原始图片）
    input_dir: Path = None

    # 输出目录（清洗后图片）
    output_dir: Path = None

    # 处理模式：quality(仅质量)、style(仅风格)、full(完整)
    mode: str = "full"

    # 是否生成清洗报告
    generate_report: bool = True

    def __post_init__(self):
        if self.quality is None:
            self.quality = QualityConfig()
        if self.style is None:
            self.style = StyleConfig()
        if self.input_dir is None:
            self.input_dir = OUTPUT_DIR
        if self.output_dir is None:
            self.output_dir = CLEANED_OUTPUT_DIR


# ==================== Output Subdirectories ====================

APPROVED_DIR = CLEANED_OUTPUT_DIR / "approved"
REJECTED_QUALITY_DIR = CLEANED_OUTPUT_DIR / "rejected_quality"
REJECTED_STYLE_DIR = CLEANED_OUTPUT_DIR / "rejected_style"
REVIEW_DIR = CLEANED_OUTPUT_DIR / "review"
REPORTS_DIR = CLEANED_OUTPUT_DIR / "reports"


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
