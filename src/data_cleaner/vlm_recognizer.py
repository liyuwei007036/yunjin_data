"""VLM 通用物体识别模块 - 使用本地视觉大模型识别图片中的物体/图案.

用于 Grounded SAM 流程的开放词汇分割.
"""

import json
import logging
import re
from typing import List

from PIL import Image

logger = logging.getLogger(__name__)


class VLMRecognizer:
    """使用本地 VLM 识别图片中的物体/图案.

    支持 Qwen2-VL 等本地部署的视觉大模型.
    识别结果是英文类别列表，用于 Grounding DINO 检测.

    注意: 不使用降级函数，必须有本地模型才能正常工作.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
    ):
        """初始化 VLM 识别器.

        Args:
            model_name: 模型名称或路径
            device: 运行设备 ("cuda" 或 "cpu")

        Raises:
            RuntimeError: 当模型加载失败时
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

        self._load_model()

    def _load_model(self):
        """加载本地模型.

        Raises:
            RuntimeError: 当模型加载失败时
        """
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from pathlib import Path

        # 检查模型路径是否存在
        model_path = Path(self.model_name)
        if not model_path.exists() and "/" not in self.model_name and "\\" not in self.model_name:
            # HuggingFace 模型名称，不需要检查本地路径
            pass
        elif not model_path.exists():
            error_msg = f"VLM 模型文件不存在: {self.model_name}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"加载 VLM 模型: {self.model_name}")

        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            logger.info("VLM 模型加载成功")
        except Exception as e:
            error_msg = f"VLM 模型加载失败: {type(e).__name__}: {e}"
            logger.error(error_msg)
            raise RuntimeError(
                f"{error_msg}\n"
                f"请确保模型 '{self.model_name}' 已正确下载或安装。"
            ) from e

    def _build_prompt(self) -> str:
        """构建通用物体识别提示词."""
        return """分析这张图片，识别其中的主要物体、图案或元素。

请列出图片中所有可见的主要元素，如：
- 人物（person, woman, man, child）
- 动物（dragon, phoenix, bird, fish, butterfly）
- 植物（flower, tree, grass, bamboo）
- 建筑（building, palace, tower）
- 自然元素（cloud, mountain, water, rock）
- 几何图案（pattern, geometric）
- 其他显著元素

只返回 JSON 数组格式，不要其他文字:
["element1", "element2", ...]

例如: ["dragon", "cloud", "flower", "mountain"]
"""

    def recognize(self, image: Image.Image) -> List[str]:
        """识别图片中的物体/图案.

        Args:
            image: 输入图片 (PIL Image)

        Returns:
            英文类别列表，如 ["dragon", "cloud", "flower"]

        Raises:
            RuntimeError: 当 VLM 模型未加载时
        """
        if self.model is None:
            raise RuntimeError("VLM 模型未加载，无法进行识别")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._build_prompt()},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        response = output_text[-1]
        if isinstance(response, dict):
            response = response.get("content", str(response))

        result = self._parse_response(response)
        if not result:
            raise RuntimeError(f"VLM 返回结果无法解析: {response}")

        return result

    def _parse_response(self, response: str) -> List[str]:
        """解析 VLM 返回的 JSON 响应."""
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    return [str(item) for item in data if isinstance(item, str)]
            except json.JSONDecodeError:
                pass

        return []

    def build_grounding_prompt(self, categories: List[str]) -> str:
        """构建 Grounding DINO 提示词.

        Args:
            categories: VLM 识别的类别列表

        Returns:
            Grounding DINO 格式的提示词
        """
        return " . ".join(categories) + " ."

    def get_grounding_prompt(self, category: str) -> str:
        """获取单个类别的 Grounding DINO 提示词."""
        return category


def create_vlm_recognizer(
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    device: str = "cuda",
) -> VLMRecognizer:
    """创建 VLM 识别器."""
    return VLMRecognizer(model_name=model_name, device=device)
