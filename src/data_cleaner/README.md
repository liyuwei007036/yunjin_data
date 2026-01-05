# Cloud Brocade Data Cleaner

云锦图片数据清洗工具，用于SD1.5微调训练数据准备。

## 功能特性

- **图片质量检查**：分辨率检测、模糊度检测、Alpha通道检测
- **云锦风格识别**：基于CLIP零样本分类
- **自动标注生成**：使用BLIP模型生成图片描述，支持Kohya_ss格式
- **批量处理**：支持10000+张图片批量清洗（包括完整图片和瓦片图）
- **GPU加速**：支持CUDA加速推理（CLIP和BLIP）

## 清洗流程详解

### 整体流程

```
输入图片 → 质量检查 → 风格分类 → 输出分类 → 自动标注（可选）
                              ├── approved/        (通过) → metadata.jsonl
                              ├── rejected_quality/ (低质量)
                              ├── rejected_style/   (非云锦)
                              └── review/          (待复核)
```

### 1. 图片质量检查 (QualityChecker)

质量检查模块使用 OpenCV 进行以下检测：

| 检测项 | 阈值 | 说明 | 不通过处理 |
|--------|------|------|-----------|
| 分辨率 | ≥ 512x512 | 检查图片宽度和高度 | width_too_small / height_too_small |
| 清晰度 | Laplacian > 50 | 使用Laplacian方差计算模糊度 | too_blurry |
| Alpha通道 | 无Alpha | 检查是否存在透明通道 | has_alpha_channel |
| 图片损坏 | 无异常 | 尝试打开并读取图片 | corrupted_image |

**清晰度计算原理：**
```python
# 使用Laplacian算子计算图像梯度方差
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sharpness = laplacian.var()  # 方差越大越清晰
```

### 2. 云锦风格分类 (StyleClassifier)

风格分类模块使用 OpenAI CLIP 模型进行零样本分类：

**工作原理：**
1. 加载 CLIP 模型 (`openai/clip-vit-base-patch32`)
2. 将图片和文本提示词编码为向量
3. 计算图片与各提示词的相似度
4. 返回云锦置信度分数 (0-1)

**提示词设计：**
```python
# 正向提示词（云锦特征）
PROMPT_POSITIVE = """
a photo of Chinese cloud brocade (yunjin) traditional textile pattern,
intricate silk brocade with gold thread, traditional Chinese embroidery,
luxurious patterned fabric with cloud motifs
"""

# 负向提示词（非云锦）
PROMPT_NEGATIVE = """
a photo of modern object, not traditional textile,
not silk brocade, not Chinese traditional pattern,
computer screen, furniture, building, food, vehicle, animal
"""
```

**分类阈值：**
| 置信度范围 | 分类结果 | 说明 |
|------------|----------|------|
| ≥ 0.65 | approved | 确认为云锦风格 |
| 0.40 ~ 0.65 | review | 边界情况，需人工复核 |
| < 0.40 | rejected_style | 确认为非云锦风格 |

### 3. 自动标注生成 (AutoCaptioner)

清洗完成后，如果启用了自动标注功能，会使用 BLIP 模型为所有通过清洗的图片生成标注：

**工作原理：**
1. 加载 BLIP 模型 (`Salesforce/blip-image-captioning-base`)
2. 为每张图片生成描述性文本
3. 自动添加触发词前缀（`cloud brocade, traditional Chinese textile`）
4. 生成 Kohya_ss 格式的 `metadata.jsonl` 文件

**输出格式（Kohya_ss 兼容）：**
```jsonl
{"file_name": "文物名_编号/full_image.png", "text": "cloud brocade, traditional Chinese textile, intricate silk brocade with gold thread"}
{"file_name": "文物名_编号/tiles/level_13/0_0.png", "text": "cloud brocade, traditional Chinese textile, detailed pattern section"}
```

**元数据文件：**
每个图片目录会生成 `metadata.json` 元数据文件：

```json
{
    "image_file": "full_image.png",
    "metadata_file": "metadata.json",
    "status": "approved",
    "width": 2048,
    "height": 1536,
    "sharpness": 125.5,
    "is_corrupted": false,
    "has_alpha_channel": false,
    "style_score": 0.85,
    "is_brocade": true,
    "needs_review": false,
    "processed_at": "2026-01-05T10:30:00.000000"
}
```

---

## 安装

```bash
# 安装项目依赖（从项目根目录）
pip install -r requirements.txt

# 或安装开发模式（如果项目有 setup.py）
pip install -e .
```

## 使用方法

### 方式一：Python 模块启动（推荐）

使用 `python -m` 方式启动，所有配置通过 YAML 文件管理：

```bash
# 使用默认配置文件（src/data_cleaner/config.yaml）
python -m src.data_cleaner.run_cleaner

# 使用自定义配置文件
python -m src.data_cleaner.run_cleaner --config my_config.yaml

# 覆盖配置参数
python -m src.data_cleaner.run_cleaner --input ./output --output ./output/cleaned

# 禁用GPU
python -m src.data_cleaner.run_cleaner --no-gpu

# 安静模式（不显示进度条）
python -m src.data_cleaner.run_cleaner --quiet
```

### 方式二：Python 代码调用

```python
from src.data_cleaner import DataCleaner
from src.data_cleaner.config import load_config

# 从配置文件加载配置
config = load_config("src/data_cleaner/config.yaml")

# 创建清洗器
cleaner = DataCleaner(config)

# 运行清洗
statistics = cleaner.clean()
```

### 支持的图片类型

工具支持处理两种类型的图片：

1. **完整图片**：`{文物名}_{编号}/merged/full_image.png`
   - 使用标准质量检查阈值（≥512x512）

2. **瓦片图**：`{文物名}_{编号}/tiles/level_{level}/{col}_{row}.png`
   - 使用更宽松的质量检查阈值（≥510x510，适配瓦片图标准尺寸）
   - 保持原有目录结构

## YAML 配置文件

### 配置文件位置

默认配置文件：`src/data_cleaner/config.yaml`

### 配置示例

```yaml
# 云锦图片数据清洗工具配置文件

# 基础配置
input_dir: "output"           # 输入目录
output_dir: "output/cleaned"  # 输出目录
mode: "full"                  # 处理模式：quality/style/full
generate_report: true         # 是否生成报告

# 质量检查配置
quality:
  min_width: 512
  min_height: 512
  blur_threshold: 50.0
  check_alpha_channel: true
  check_corruption: true

# 风格分类配置
style:
  style_threshold: 0.65       # 云锦置信度阈值
  review_threshold: 0.40      # 人工复核阈值
  clip_model_name: "openai/clip-vit-base-patch32"
  use_gpu: true               # 是否使用GPU
  batch_size: 32              # 批量推理大小

# 自动标注配置
caption:
  enabled: true               # 是否启用自动标注
  blip_model_name: "Salesforce/blip-image-captioning-base"
  use_gpu: true               # 是否使用GPU
  max_length: 70              # Caption最大长度
  caption_prefix: "cloud brocade, traditional Chinese textile"  # 触发词前缀
  default_caption: "cloud brocade, traditional Chinese textile"  # 默认caption

# 输出子目录配置
output_subdirs:
  approved: "approved"
  rejected_quality: "rejected_quality"
  rejected_style: "rejected_style"
  review: "review"
  reports: "reports"
```

## 输入输出结构

### 输入目录（与爬虫输出一致）

```
output/
└── {文物名}_{藏品编号}/
    ├── merged/
    │   └── full_image.png
    └── tiles/
        └── level_{level}/
            └── {col}_{row}.png
```

### 输出目录

```
output/cleaned/
├── approved/                 # 通过清洗（用于SD1.5训练）
│   └── {文物名}_{藏品编号}/
│       ├── full_image.png
│       ├── tiles/           # 瓦片图（保持目录结构）
│       │   └── level_{level}/
│       │       └── {col}_{row}.png
│       └── metadata.json
├── rejected_quality/         # 质量不达标
│   └── {文物名}_{藏品编号}/
│       ├── full_image.png
│       └── metadata.json
├── rejected_style/           # 非云锦风格
│   └── {文物名}_{藏品编号}/
│       ├── full_image.png
│       └── metadata.json
├── review/                   # 待人工复核
│   └── {文物名}_{藏品编号}/
│       ├── full_image.png
│       └── metadata.json
└── reports/
    ├── cleaning_report_*.txt  # 清洗报告（文本格式）
    └── approved/
        └── metadata.jsonl     # 训练数据（Kohya_ss格式，如果启用自动标注）
```

---

## 配置选项

### QualityConfig（质量检查）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| min_width | 512 | 最小宽度（像素） |
| min_height | 512 | 最小高度（像素） |
| blur_threshold | 50.0 | 模糊度阈值（Laplacian方差） |
| check_alpha_channel | True | 是否检查Alpha通道 |
| check_corruption | True | 是否检查图片损坏 |

### StyleConfig（风格分类）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| style_threshold | 0.65 | 云锦置信度阈值 |
| review_threshold | 0.40 | 人工复核阈值 |
| clip_model_name | openai/clip-vit-base-patch32 | CLIP模型名称 |
| use_gpu | True | 是否使用GPU |
| batch_size | 32 | 批量推理大小 |

### CaptionConfig（自动标注）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| enabled | True | 是否启用自动标注 |
| blip_model_name | Salesforce/blip-image-captioning-base | BLIP模型名称 |
| use_gpu | True | 是否使用GPU |
| max_length | 70 | Caption最大长度 |
| caption_prefix | cloud brocade, traditional Chinese textile | 触发词前缀（必需） |
| default_caption | cloud brocade, traditional Chinese textile | 默认caption（生成失败时使用） |

---

## 性能预估

| 阶段 | GPU | CPU | 时间估算(10000张) |
|------|-----|-----|-------------------|
| 质量检查 | 否 | 8核 | ~5分钟 |
| CLIP分类 | RTX 3080+ | 4核 | ~30分钟 |
| BLIP标注（可选） | RTX 3080+ | 4核 | ~20分钟 |
| 总计（不含标注） | - | - | ~40分钟 |
| 总计（含标注） | - | - | ~60分钟 |

**优化建议：**
- 使用 GPU 加速 CLIP 和 BLIP 推理
- 批量处理图片（batch_size=32）
- 半精度推理（fp16）
- 如果不需要自动标注，可在配置中禁用 `caption.enabled: false`

---

## 依赖项

- pillow >= 10.0.0
- opencv-python >= 4.8.0
- torch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.35.0
- tqdm >= 4.66.0
- pyyaml >= 6.0.0
