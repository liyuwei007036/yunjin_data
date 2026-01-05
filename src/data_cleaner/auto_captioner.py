"""
Auto Captioner Module.

使用 BLIP 模型为图片生成自动标注，用于 SD1.5 LoRA 训练数据集。
"""

import torch
from pathlib import Path
from typing import Optional, List, Dict
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

from src.data_cleaner.config import CaptionConfig


class AutoCaptioner:
    """自动标注器，使用 BLIP 模型生成图像描述"""

    def __init__(self, config: Optional[CaptionConfig] = None):
        """
        初始化自动标注器

        Args:
            config: 标注配置，默认为标准配置
        """
        self.config = config or CaptionConfig()
        self.processor = None
        self.model = None
        self.device = None
        self._model_loaded = False

    def _get_device(self):
        """获取计算设备"""
        if self.device is None:
            if self.config.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        return self.device

    def load_model(self):
        """加载 BLIP 模型"""
        if self._model_loaded:
            return

        print(f"Loading BLIP model: {self.config.blip_model_name}")
        device = self._get_device()

        try:
            self.processor = BlipProcessor.from_pretrained(self.config.blip_model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.config.blip_model_name
            ).to(device)
            self.model.eval()
            self._model_loaded = True
            print(f"BLIP model loaded on {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load BLIP model: {e}")

    def generate_caption(self, image_path: Path) -> str:
        """
        为单张图片生成 caption

        Args:
            image_path: 图片路径

        Returns:
            str: 生成的 caption
        """
        if not self._model_loaded:
            self.load_model()

        try:
            # 加载图片
            image = Image.open(image_path).convert("RGB")

            # 使用 BLIP 生成 caption
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=self.config.max_length)

            caption = self.processor.decode(out[0], skip_special_tokens=True)

            # 前缀是必需的，总是添加到生成的 caption 前面
            # 这确保了 SD1.5 LoRA 训练中的触发词（trigger words）总是存在
            caption = f"{self.config.caption_prefix}, {caption}"

            return caption.strip()

        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            # 返回默认 caption（默认 caption 已经包含了前缀的内容）
            return self.config.default_caption

    def generate_captions_batch(
        self, image_paths: List[Path], show_progress: bool = True
    ) -> Dict[Path, str]:
        """
        批量生成 caption

        Args:
            image_paths: 图片路径列表
            show_progress: 是否显示进度条

        Returns:
            dict: {图片路径: caption} 的字典
        """
        if not self._model_loaded:
            self.load_model()

        results = {}
        iterator = tqdm(image_paths, desc="Generating captions") if show_progress else image_paths

        for image_path in iterator:
            caption = self.generate_caption(image_path)
            results[image_path] = caption

        return results

    def generate_captions_for_directory(
        self, directory: Path, show_progress: bool = True
    ) -> Dict[Path, str]:
        """
        为目录下的所有图片生成 caption

        Args:
            directory: 图片目录
            show_progress: 是否显示进度条

        Returns:
            dict: {图片路径: caption} 的字典
        """
        # 查找所有图片文件
        image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        image_paths = [
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            return {}

        return self.generate_captions_batch(image_paths, show_progress=show_progress)

