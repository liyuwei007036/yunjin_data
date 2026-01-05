"""
Cloud Brocade Style Classifier Module.

云锦风格分类器模块，使用CLIP进行零样本分类。
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List

import torch
from PIL import Image
from PIL import UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from src.data_cleaner.config import (
    StyleConfig,
    CLOUC_BROCADE_PROMPT,
    NOT_BROCADE_PROMPT,
)

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class StyleResult:
    """风格分类结果"""

    image_path: Path
    is_brocade: bool
    needs_review: bool
    confidence: float
    brocade_score: float
    not_brocade_score: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典格式"""
        result = {
            "image_path": str(self.image_path),
            "is_brocade": self.is_brocade,
            "needs_review": self.needs_review,
            "confidence": self.confidence,
            "brocade_score": self.brocade_score,
            "not_brocade_score": self.not_brocade_score,
        }
        if self.error:
            result["error"] = self.error
        return result


class StyleClassifier:
    """云锦风格分类器（基于CLIP零样本分类）"""

    def __init__(self, config: Optional[StyleConfig] = None):
        """
        初始化风格分类器

        Args:
            config: 风格分类配置，默认为标准配置

        Raises:
            ValueError: 如果配置参数无效
        """
        self.config = config or StyleConfig()
        self._validate_config()
        
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.device: str = "cpu"  # 在 load_model 中会更新

    def _validate_config(self):
        """验证配置参数"""
        if not 0.0 <= self.config.style_threshold <= 1.0:
            raise ValueError(
                f"style_threshold must be between 0.0 and 1.0, got {self.config.style_threshold}"
            )
        if not 0.0 <= self.config.review_threshold <= 1.0:
            raise ValueError(
                f"review_threshold must be between 0.0 and 1.0, got {self.config.review_threshold}"
            )
        if self.config.review_threshold >= self.config.style_threshold:
            raise ValueError(
                f"review_threshold ({self.config.review_threshold}) must be less than "
                f"style_threshold ({self.config.style_threshold})"
            )
        if self.config.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.config.batch_size}")

    def load_model(self):
        """加载CLIP模型"""
        if self.model is not None:
            return

        logger.info(f"Loading CLIP model: {self.config.clip_model_name}...")

        try:
            self.model = CLIPModel.from_pretrained(self.config.clip_model_name)
            self.processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)

            # 设置设备
            if self.config.use_gpu and torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                if self.config.use_gpu:
                    logger.warning("GPU requested but not available, using CPU")

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def _create_error_result(
        self, image_path: Path, error_message: str
    ) -> StyleResult:
        """创建错误结果"""
        return StyleResult(
            image_path=image_path,
            is_brocade=False,
            needs_review=False,
            confidence=0.0,
            brocade_score=0.0,
            not_brocade_score=1.0,
            error=error_message,
        )

    def classify(self, image_path: Union[str, Path]) -> StyleResult:
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

        # 验证文件存在
        if not path.exists():
            error_msg = f"Image file not found: {path}"
            logger.error(error_msg)
            return self._create_error_result(path, error_msg)

        try:
            # 加载图片
            image = Image.open(path).convert("RGB")
            
            # 验证图片尺寸
            if image.size[0] == 0 or image.size[1] == 0:
                error_msg = f"Invalid image dimensions: {image.size}"
                logger.error(f"{error_msg} for {path}")
                return self._create_error_result(path, error_msg)

        except UnidentifiedImageError:
            error_msg = f"Cannot identify image file: {path}"
            logger.error(error_msg)
            return self._create_error_result(path, error_msg)
        except Exception as e:
            error_msg = f"Failed to load image: {e}"
            logger.error(f"{error_msg} for {path}", exc_info=True)
            return self._create_error_result(path, error_msg)

        try:
            # 处理图片和文本
            inputs = self.processor(
                text=[CLOUC_BROCADE_PROMPT, NOT_BROCADE_PROMPT],
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
            brocade_score = float(probs[0][0].item())
            not_brocade_score = float(probs[0][1].item())

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
            error_msg = f"Classification error: {e}"
            logger.error(f"{error_msg} for {path}", exc_info=True)
            return self._create_error_result(path, error_msg)

    def _classify_batch_internal(
        self, images: List[Image.Image], image_paths: List[Path]
    ) -> List[StyleResult]:
        """
        内部批量分类方法，真正批量处理图片

        Args:
            images: PIL图片列表
            image_paths: 对应的图片路径列表

        Returns:
            分类结果列表
        """
        if not images:
            return []

        try:
            # 处理所有图片和文本
            inputs = self.processor(
                text=[CLOUC_BROCADE_PROMPT, NOT_BROCADE_PROMPT],
                images=images,
                return_tensors="pt",
                padding=True,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 批量推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # 处理结果
            results = []
            for i, path in enumerate(image_paths):
                brocade_score = float(probs[i][0].item())
                not_brocade_score = float(probs[i][1].item())

                is_brocade = brocade_score >= self.config.style_threshold
                needs_review = (
                    self.config.review_threshold <= brocade_score < self.config.style_threshold
                )
                confidence = abs(brocade_score - not_brocade_score)

                results.append(
                    StyleResult(
                        image_path=path,
                        is_brocade=is_brocade,
                        needs_review=needs_review,
                        confidence=confidence,
                        brocade_score=brocade_score,
                        not_brocade_score=not_brocade_score,
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Batch classification error: {e}", exc_info=True)
            # 如果批量处理失败，返回错误结果
            return [
                self._create_error_result(path, f"Batch processing error: {e}")
                for path in image_paths
            ]

    def classify_batch(
        self,
        image_paths: List[Union[str, Path]],
        show_progress: bool = True,
    ) -> List[StyleResult]:
        """
        批量对图片进行风格分类

        使用真正的批量处理以提高效率。

        Args:
            image_paths: 图片路径列表
            show_progress: 是否显示进度条

        Returns:
            list[StyleResult]: 分类结果列表
        """
        if self.model is None:
            self.load_model()

        if not image_paths:
            return []

        # 转换为Path对象
        paths = [Path(p) for p in image_paths]
        batch_size = self.config.batch_size
        results = []

        # 创建进度条
        iterator = (
            tqdm(
                range(0, len(paths), batch_size),
                desc="Style Classification",
                unit="batch",
            )
            if show_progress
            else range(0, len(paths), batch_size)
        )

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(paths))
            batch_paths = paths[batch_start:batch_end]

            # 加载批次图片
            batch_images = []
            valid_paths = []
            for path in batch_paths:
                if not path.exists():
                    logger.warning(f"Image file not found: {path}")
                    results.append(
                        self._create_error_result(path, "File not found")
                    )
                    continue

                try:
                    image = Image.open(path).convert("RGB")
                    if image.size[0] == 0 or image.size[1] == 0:
                        logger.warning(f"Invalid image dimensions: {image.size} for {path}")
                        results.append(
                            self._create_error_result(path, "Invalid image dimensions")
                        )
                        continue
                    batch_images.append(image)
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")
                    results.append(
                        self._create_error_result(path, f"Failed to load: {e}")
                    )
                    continue

            # 批量处理有效图片
            if batch_images:
                batch_results = self._classify_batch_internal(batch_images, valid_paths)
                results.extend(batch_results)

        return results

    def get_statistics(self, results: List[StyleResult]) -> dict:
        """
        获取批量分类统计信息

        Args:
            results: 分类结果列表

        Returns:
            dict: 统计信息，包含总数、各类别数量、比率和平均值
        """
        if not results:
            return {
                "total": 0,
                "brocade": 0,
                "not_brocade": 0,
                "needs_review": 0,
                "brocade_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_brocade_score": 0.0,
                "error_count": 0,
                "error_rate": 0.0,
            }

        total = len(results)
        brocade = sum(1 for r in results if r.is_brocade)
        review = sum(1 for r in results if r.needs_review)
        not_brocade = total - brocade - review
        error_count = sum(1 for r in results if r.error is not None)

        # 只计算无错误结果的统计
        valid_results = [r for r in results if r.error is None]
        valid_count = len(valid_results)

        avg_confidence = (
            sum(r.confidence for r in valid_results) / valid_count
            if valid_count > 0
            else 0.0
        )
        avg_brocade_score = (
            sum(r.brocade_score for r in valid_results) / valid_count
            if valid_count > 0
            else 0.0
        )

        return {
            "total": total,
            "brocade": brocade,
            "not_brocade": not_brocade,
            "needs_review": review,
            "brocade_rate": brocade / total if total > 0 else 0.0,
            "avg_confidence": avg_confidence,
            "avg_brocade_score": avg_brocade_score,
            "error_count": error_count,
            "error_rate": error_count / total if total > 0 else 0.0,
        }

    def cleanup(self):
        """清理资源，释放模型内存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model resources cleaned up")


def classify_brocade_style(
    image_path: Union[str, Path],
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
    try:
        return classifier.classify(image_path)
    finally:
        classifier.cleanup()
