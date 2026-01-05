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

from src.data_cleaner.content_analyzer import ContentAnalyzer


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
        
        # 如果启用了内容感知检查，初始化内容分析器
        if self.config.use_content_aware_check or self.config.multi_scale_check:
            # 使用默认配置初始化内容分析器
            # 注意：这里不检查内容占比阈值，只用于获取内容占比值
            self.content_analyzer = ContentAnalyzer(
                min_content_ratio=0.0,  # 不用于阈值判断
                max_border_ratio=1.0,   # 不用于阈值判断
                border_region_width=0.15,
            )
        else:
            self.content_analyzer = None

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

            # 获取基础分数（整体图片的云锦风格分数）
            brocade_score = probs[0][0].item()  # 云锦概率分数（0.0-1.0）
            not_brocade_score = probs[0][1].item()  # 非云锦概率分数（0.0-1.0）
            
            # ========== 多尺度检查（如果启用） ==========
            # 目的：检测图片各个区域是否都是云锦风格，避免只有部分区域是云锦却被整体误判
            # 方法：将图片分成4个象限（左上、右上、左下、右下），对每个局部区域进行风格分类
            # 效果：如果图片只有部分区域是云锦，多尺度检查会降低整体分数，提高分类准确性
            if self.config.multi_scale_check and self.content_analyzer:
                # 对局部区域进行风格分类，返回每个区域的云锦分数列表
                local_scores = self._classify_local_regions(image, path)
                if local_scores:
                    # 结合整体和局部分数：取平均值
                    # 公式：(整体分数 + 所有局部分数之和) / (1 + 局部区域数量)
                    # 例如：整体0.75，局部[0.8, 0.3, 0.2, 0.4] -> (0.75 + 1.7) / 5 = 0.49
                    brocade_score = (brocade_score + sum(local_scores)) / (1 + len(local_scores))
                    # 重新计算非云锦分数（确保两个分数之和为1.0）
                    not_brocade_score = 1.0 - brocade_score
            
            # ========== 内容感知检查（如果启用） ==========
            # 目的：根据图片中有效内容占比调整云锦分数，避免空白/背景区域较多的图片被误判
            # 方法：计算图片中有效图案区域占比，如果占比低则降低云锦分数
            # 效果：如果图片主要是空白或背景，即使包含云锦元素，分数也会被降低
            if self.config.use_content_aware_check and self.content_analyzer:
                # 获取内容分析结果（包括内容占比、边界占比等）
                content_result = self.content_analyzer.analyze_content(path)
                content_ratio = content_result.content_ratio  # 有效内容占比（0.0-1.0）
                
                # 根据内容占比调整分数
                # 调整因子计算公式：1.0 - weight * (1.0 - content_ratio)
                # - 当 content_ratio = 1.0（全是有效内容）时，adjustment_factor = 1.0（不调整）
                # - 当 content_ratio = 0.0（全是空白）时，adjustment_factor = 1.0 - weight（最大降低）
                # - 当 content_ratio = 0.5（一半内容）时，adjustment_factor = 1.0 - weight * 0.5（中等降低）
                # 例如：weight=0.3, content_ratio=0.3 -> adjustment_factor = 1.0 - 0.3*0.7 = 0.79
                adjustment_factor = 1.0 - self.config.content_ratio_weight * (1.0 - content_ratio)
                # 应用调整因子：降低云锦分数
                brocade_score = brocade_score * adjustment_factor
                # 重新计算非云锦分数（确保两个分数之和为1.0）
                not_brocade_score = 1.0 - brocade_score

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

    def _classify_local_regions(self, image: Image.Image, image_path: Path) -> list[float]:
        """
        对图片的局部区域进行分类（多尺度检查）
        
        将图片分成4个象限，对每个局部区域使用CLIP模型进行风格分类，
        返回每个区域的云锦分数。用于检测图片是否所有区域都是云锦风格。

        Args:
            image: PIL图片对象（完整图片）
            image_path: 图片路径（用于错误处理和日志）

        Returns:
            list[float]: 局部区域的云锦分数列表，按顺序为[左上, 右上, 左下, 右下]
                        如果某个区域处理失败，该区域的分数不会被添加到列表中
        """
        if self.model is None:
            return []

        try:
            w, h = image.size
            local_scores = []

            # 将图片分成4个象限（每个象限占图片的1/4）
            # 区域坐标格式：(left, top, right, bottom)
            regions = [
                (0, 0, w // 2, h // 2),           # 左上象限：从(0,0)到(w/2, h/2)
                (w // 2, 0, w, h // 2),          # 右上象限：从(w/2,0)到(w, h/2)
                (0, h // 2, w // 2, h),          # 左下象限：从(0,h/2)到(w/2, h)
                (w // 2, h // 2, w, h),          # 右下象限：从(w/2,h/2)到(w, h)
            ]

            # 对每个象限进行风格分类
            for region in regions:
                try:
                    # 裁剪出局部区域
                    local_image = image.crop(region)
                    
                    # 使用CLIP处理器准备输入（图片和文本提示词）
                    inputs = self.processor(
                        text=[self.PROMPT_POSITIVE, self.PROMPT_NEGATIVE],
                        images=local_image,
                        return_tensors="pt",
                        padding=True,
                    )

                    # 将输入数据移动到正确的设备（CPU或GPU）
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # 使用CLIP模型进行推理（不计算梯度，节省内存）
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        # 将logits转换为概率分布（softmax）
                        probs = logits_per_image.softmax(dim=1)

                    # 获取局部区域的云锦分数（第一个概率值）
                    local_brocade_score = probs[0][0].item()
                    local_scores.append(local_brocade_score)
                except Exception:
                    # 如果某个区域处理失败（例如图片太小、模型错误等），跳过该区域
                    # 继续处理其他区域，不影响整体流程
                    continue

            return local_scores

        except Exception:
            # 如果多尺度检查整体失败（例如模型未加载、图片读取错误等），返回空列表
            # 调用方会检查列表是否为空，如果为空则跳过多尺度检查，使用原始分数
            return []

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
