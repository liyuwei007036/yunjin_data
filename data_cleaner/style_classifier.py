"""
Cloud Brocade Style Classifier Module.

云锦风格分类器模块，使用CLIP进行零样本分类。
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


@dataclass
class StyleResult:
    """风格分类结果"""

    image_path: Path
    is_brocade: bool
    needs_review: bool
    confidence: float
    brocade_score: float
    not_brocade_score: float

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "image_path": str(self.image_path),
            "is_brocade": self.is_brocade,
            "needs_review": self.needs_review,
            "confidence": self.confidence,
            "brocade_score": self.brocade_score,
            "not_brocade_score": self.not_brocade_score,
        }


@dataclass
class StyleConfig:
    """风格分类配置"""

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


class StyleClassifier:
    """云锦风格分类器（基于CLIP零样本分类）"""

    # 默认提示词
    PROMPT_POSITIVE = (
        "a photo of Chinese cloud brocade (yunjin) traditional textile pattern, "
        "intricate silk brocade with gold thread, traditional Chinese embroidery, "
        "luxurious patterned fabric with cloud motifs"
    )

    PROMPT_NEGATIVE = (
        "a photo of modern object, not traditional textile, "
        "not silk brocade, not Chinese traditional pattern, "
        "computer screen, furniture, building, food, vehicle, animal"
    )

    def __init__(self, config: Optional[StyleConfig] = None):
        """
        初始化风格分类器

        Args:
            config: 风格分类配置，默认为标准配置
        """
        self.config = config or StyleConfig()
        self.model = None
        self.processor = None

    def load_model(self):
        """加载CLIP模型"""
        if self.model is not None:
            return

        print(f"Loading CLIP model: {self.config.clip_model_name}...")

        self.model = CLIPModel.from_pretrained(self.config.clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)

        # 设置设备
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model.to(self.device)

        # 设置评估模式
        self.model.eval()

        print(f"Model loaded on {self.device}")

    def classify(self, image_path: str | Path) -> StyleResult:
        """
        对单张图片进行风格分类

        Args:
            image_path: 图片路径

        Returns:
            StyleResult: 分类结果
        """
        if self.model is None:
            self.load_model()

        path = Path(image_path)

        try:
            # 加载图片
            image = Image.open(path).convert("RGB")

            # 处理图片和文本
            inputs = self.processor(
                text=[self.PROMPT_POSITIVE, self.PROMPT_NEGATIVE],
                images=image,
                return_tensors="pt",
                padding=True,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # 获取分数
            brocade_score = probs[0][0].item()
            not_brocade_score = probs[0][1].item()

            # 判断结果
            is_brocade = brocade_score >= self.config.style_threshold
            needs_review = (
                self.config.review_threshold <= brocade_score < self.config.style_threshold
            )
            confidence = abs(brocade_score - not_brocade_score)

            return StyleResult(
                image_path=path,
                is_brocade=is_brocade,
                needs_review=needs_review,
                confidence=confidence,
                brocade_score=brocade_score,
                not_brocade_score=not_brocade_score,
            )

        except Exception as e:
            print(f"Error classifying {path}: {e}")
            return StyleResult(
                image_path=path,
                is_brocade=False,
                needs_review=False,
                confidence=0.0,
                brocade_score=0.0,
                not_brocade_score=1.0,
            )

    def classify_batch(
        self, image_paths: list[str | Path], show_progress: bool = True
    ) -> list[StyleResult]:
        """
        批量对图片进行风格分类

        Args:
            image_paths: 图片路径列表
            show_progress: 是否显示进度条

        Returns:
            list[StyleResult]: 分类结果列表
        """
        if self.model is None:
            self.load_model()

        results = []
        iterator = tqdm(image_paths, desc="Style Classification") if show_progress else image_paths

        for path in iterator:
            result = self.classify(path)
            results.append(result)

        return results

    def get_statistics(self, results: list[StyleResult]) -> dict:
        """
        获取批量分类统计信息

        Args:
            results: 分类结果列表

        Returns:
            dict: 统计信息
        """
        total = len(results)
        brocade = sum(1 for r in results if r.is_brocade)
        review = sum(1 for r in results if r.needs_review)
        not_brocade = total - brocade - review

        # 计算平均置信度
        avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0

        # 计算分数分布
        avg_brocade_score = sum(r.brocade_score for r in results) / total if total > 0 else 0

        return {
            "total": total,
            "brocade": brocade,
            "not_brocade": not_brocade,
            "needs_review": review,
            "brocade_rate": brocade / total if total > 0 else 0,
            "avg_confidence": avg_confidence,
            "avg_brocade_score": avg_brocade_score,
        }


def classify_brocade_style(
    image_path: str | Path,
    style_threshold: float = 0.65,
    review_threshold: float = 0.40,
    use_gpu: bool = True,
) -> StyleResult:
    """
    便捷函数：对单张图片进行云锦风格分类

    Args:
        image_path: 图片路径
        style_threshold: 云锦置信度阈值
        review_threshold: 人工复核阈值
        use_gpu: 是否使用GPU

    Returns:
        StyleResult: 分类结果
    """
    config = StyleConfig(
        style_threshold=style_threshold,
        review_threshold=review_threshold,
        use_gpu=use_gpu,
    )
    classifier = StyleClassifier(config)
    return classifier.classify(image_path)
