"""图案提取主模块 - 统一接口，智能选择策略，后处理.

支持:
- 透明背景 PNG 图片导出
- COCO 格式标注导出
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import Config
from .image_analyzer import ImageAnalysis, ImageAnalyzer, ImageType
from .strategies import GroundedSAMStrategy


class PatternExtractor:
    """图案提取主类.

    统一接口，智能选择提取策略，自动去重和质量筛选.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.analyzer = ImageAnalyzer(
            use_vlm=self.config.use_vlm,
            vlm_model=self.config.vlm_model
        )

        self.strategies = {
            ImageType.PATTERN_SEGMENT: GroundedSAMStrategy(
                grounding_dino_model=self.config.grounding_dino_path,
                sam_model=self.config.sam_model_path,
                device=self.config.device,
            ),
        }

    def extract(self, image_path: str) -> Tuple[Image.Image, ImageAnalysis, List[Image.Image]]:
        """从单张图片提取图案.

        Args:
            image_path: 图片路径

        Returns:
            (原始图片, 分析结果, 提取的图案列表)
        """
        image = Image.open(image_path).convert("RGB")
        analysis = self.analyzer.analyze(image)
        strategy = self.strategies[analysis.type]

        patterns = strategy.extract(
            image, analysis,
            target_size=self.config.target_size,
            min_crop_size=self.config.min_crop_size
        )
        patterns = self._post_process(patterns, analysis)

        return image, analysis, patterns

    def extract_from_image(self, image: Image.Image, analysis: ImageAnalysis) -> List[Image.Image]:
        """从已加载的图片提取图案."""
        strategy = self.strategies[analysis.type]

        patterns = strategy.extract(
            image, analysis,
            target_size=self.config.target_size,
            min_crop_size=self.config.min_crop_size
        )

        return self._post_process(patterns, analysis)

    def _post_process(
        self, patterns: List[Image.Image], analysis: ImageAnalysis
    ) -> List[Image.Image]:
        """后处理：去重 + 质量筛选 + 排序."""
        if not patterns:
            return []

        unique_patterns = self._deduplicate(patterns)
        scored = [(p, self._quality_score(p)) for p in unique_patterns]
        scored = sorted(scored, key=lambda x: x[1], reverse=True)

        filtered = [p for p, score in scored if score >= self.config.min_quality_score]

        return filtered[:self.config.max_patterns_per_image]

    def _deduplicate(self, patterns: List[Image.Image]) -> List[Image.Image]:
        """去重（使用简化的哈希）."""
        import hashlib

        seen_hashes = set()
        unique = []

        for pattern in patterns:
            small = pattern.resize((32, 32), Image.Resampling.LANCZOS)
            arr = np.array(small)
            data = arr.tobytes()
            pattern_hash = hashlib.md5(data).hexdigest()

            if pattern_hash not in seen_hashes:
                seen_hashes.add(pattern_hash)
                unique.append(pattern)

        return unique

    def _quality_score(self, pattern: Image.Image) -> float:
        """计算图案质量分数."""
        if pattern.mode == "RGBA":
            rgb = pattern.convert("RGB")
        else:
            rgb = pattern

        gray = np.array(rgb.convert("L"))

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        if pattern.mode == "RGBA":
            alpha = np.array(pattern)[:, :, 3]
            non_transparent = np.sum(alpha > 0)
            content_ratio = non_transparent / alpha.size
        else:
            content_ratio = 1.0

        score = sharpness * (0.5 + content_ratio * 0.5)

        return score

    def process_directory(
        self, input_dir: str, output_dir: str, pattern_prefix: str = ""
    ) -> dict:
        """批量处理目录中的图片."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))

        stats = {
            "total": len(image_files),
            "success": 0,
            "failed": 0,
            "total_patterns": 0,
            "failed_files": [],
        }

        for image_file in tqdm(image_files, desc="处理图片"):
            try:
                image, analysis, patterns = self.extract(str(image_file))

                if patterns:
                    base_name = image_file.stem
                    for i, pattern in enumerate(patterns):
                        output_file = output_path / f"{base_name}_{pattern_prefix}{i+1:03d}.png"
                        pattern.save(output_file, "PNG")

                    stats["success"] += 1
                    stats["total_patterns"] += len(patterns)
                else:
                    stats["success"] += 1

            except Exception as e:
                stats["failed"] += 1
                stats["failed_files"].append((image_file.name, str(e)))

        return stats


class TrainingDataExporter:
    """训练数据导出器.

    导出 COCO 格式的标注文件.
    """

    def __init__(
        self,
        output_dir: str = "output/annotations",
        image_size: int = 1024,
    ):
        """初始化导出器.

        Args:
            output_dir: 输出目录
            image_size: 输出图片尺寸
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.image_dir = self.output_dir / "images"
        self.annotation_dir = self.output_dir / "annotations"

        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        image: Image.Image,
        categories: List[str],
        patterns: List[Image.Image],
        image_name: str,
    ) -> Dict:
        """导出训练数据.

        Args:
            image: 原始图片
            categories: 物体类别列表
            patterns: 分割后的图案列表
            image_name: 图片名称

        Returns:
            标注信息
        """
        base_name = Path(image_name).stem

        annotation = {
            "info": {
                "description": "Pattern extraction dataset",
                "date_created": datetime.now().isoformat(),
            },
            "images": [],
            "annotations": [],
            "categories": [],
        }

        category_id_map = {}
        for cat in set(categories):
            cat_id = len(category_id_map) + 1
            category_id_map[cat] = cat_id
            annotation["categories"].append({
                "id": cat_id,
                "name": cat,
                "supercategory": "pattern",
            })

        img_width, img_height = image.size
        image_info = {
            "id": 0,
            "file_name": f"{base_name}.png",
            "width": img_width,
            "height": img_height,
        }
        annotation["images"].append(image_info)

        annotation_id = 0
        for idx, pattern in enumerate(patterns):
            pattern_resized = pattern.resize(
                (self.image_size, self.image_size), Image.Resampling.LANCZOS
            )

            pattern_filename = f"{base_name}_{idx:04d}.png"
            pattern_path = self.image_dir / pattern_filename
            pattern_resized.save(pattern_path)

            mask = np.array(pattern)[:, :, 3]
            binary_mask = mask > 127

            if not np.any(binary_mask):
                continue

            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            bbox = [
                int(cmin * img_width / self.image_size),
                int(rmin * img_height / self.image_size),
                int((cmax - cmin + 1) * img_width / self.image_size),
                int((rmax - rmin + 1) * img_height / self.image_size),
            ]

            area = int(np.sum(binary_mask) * img_width * img_height / (self.image_size ** 2))

            mask_uint8 = binary_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                if len(contour) >= 6:
                    segmentation.append(contour)

            cat_name = categories[idx % len(categories)] if categories else "pattern"
            cat_id = category_id_map.get(cat_name, 1)

            pattern_annotation = {
                "id": annotation_id,
                "image_id": 0,
                "category_id": cat_id,
                "bbox": bbox,
                "area": area,
                "segmentation": segmentation,
                "iscrowd": 0,
            }
            annotation["annotations"].append(pattern_annotation)
            annotation_id += 1

        annotation_path = self.annotation_dir / f"{base_name}.json"
        with open(annotation_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

        return {
            "image_name": image_name,
            "categories": categories,
            "num_patterns": len(patterns),
            "annotation_file": str(annotation_path),
        }

    def export_batch(
        self,
        results: List[Dict],
    ) -> Dict:
        """批量导出训练数据.

        Args:
            results: [(原始图片, 类别列表, 分割图案列表, 图片名), ...]

        Returns:
            数据集信息
        """
        all_categories = set()
        total_patterns = 0

        for image, categories, patterns, image_name in results:
            all_categories.update(categories)
            total_patterns += len(patterns)

        dataset_info = {
            "total_images": len(results),
            "total_patterns": total_patterns,
            "categories": sorted(list(all_categories)),
            "date_created": datetime.now().isoformat(),
        }

        with open(self.output_dir / "dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        return dataset_info


def create_exporter(
    output_dir: str = "output/annotations",
    image_size: int = 1024,
) -> TrainingDataExporter:
    """创建训练数据导出器."""
    return TrainingDataExporter(output_dir=output_dir, image_size=image_size)
