# EEG WaveletVQ Framework

基于小波神经量化和动态语义对齐的统一EEG表示学习框架

## 项目概述

本项目实现了论文《A Unified EEG Representation Framework via Wavelet Neural Quantization and Dynamic Semantic Alignment》中提出的方法主体框架。

### 主要特性

- **WaveletVQ神经令牌化器**: 结合离散小波变换和向量量化的EEG信号编码
- **并行特征编码**: 时间、空间、语义三维并行特征提取
- **跨模态语义对齐**: 对比学习和掩码重建双约束机制
- **多数据集支持**: EIT-1M, ChineseEEG, Thought2Text数据集

### 方法框架

1. **阶段1**: WaveletVQ神经令牌化
   - 离散小波变换(DWT)分解
   - 通道级Transformer建模
   - 向量量化与重建

2. **阶段2**: 并行特征编码
   - 时间编码器：长距离时间依赖
   - 空间编码器：脑区分组与全局交互
   - 语义编码器：频段注意力机制

3. **阶段3**: 跨模态对齐
   - 对比学习最大化EEG-文本互信息
   - 掩码重建传递LLM序列建模能力

## 安装要求

### 环境配置

```bash
# Python环境
Python >= 3.8
PyTorch >= 2.3.0
CUDA >= 12.2

# 硬件要求（按照论文配置）
4x NVIDIA RTX 6000 Ada GPUs (48GB VRAM)
```

### 依赖安装

```bash
# 克隆项目
git clone <repository-url>
cd EEG-WaveletVQ-Framework

# 安装依赖
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
pip install transformers
pip install pywt  # 小波变换
pip install mne  # EEG数据处理
pip install scikit-learn
pip install nltk
pip install rouge-score
pip install h5py
pip install wandb  # 可选：实验记录
pip install pyyaml
pip install tqdm
```

## 数据准备

### 支持的数据集

1. **EIT-1M**: 1百万EEG-图像-文本对
2. **ChineseEEG**: 128通道中文语料EEG数据
3. **Thought2Text**: 图像对齐的EEG-文本条目
4. **TUAB**: 通过脑电识别受试者观看的图像或阅读的文本语义

### 数据预处理

```bash
# EEG预处理步骤：
# 1. 0.5-70 Hz带通滤波
# 2. 48-52 Hz陷波滤波
# 3. 重采样到500 Hz
# 4. 2秒片段分割
# 5. 坏通道检测与插值
# 6. ICA去除伪迹
# 7. 平均重参考
```

## 使用说明

### 训练模型

```bash
# 基础训练
python scripts/train.py --config configs/default_config.yaml

# 分布式训练（4 GPUs）
python scripts/train.py \
    --config configs/default_config.yaml \
    --gpus 0,1,2,3 \
    --distributed \
    --name eeg_wavelet_training

# 从检查点恢复
python scripts/train.py \
    --config configs/default_config.yaml \
    --resume experiments/checkpoints/best_model.pt
```

### 评估模型

```bash
# 评估所有数据集
python scripts/evaluate.py \
    --config configs/default_config.yaml \
    --checkpoint experiments/checkpoints/best_model.pt \
    --dataset all

# 评估特定数据集
python scripts/evaluate.py \
    --config configs/default_config.yaml \
    --checkpoint experiments/checkpoints/best_model.pt \
    --dataset eit1m \
    --split test
```

### Python API使用

```python
from src.models.unified_framework import create_unified_model
from src.data.datasets import MultiDatasetLoader

# 创建模型
config = {
    'n_channels': 64,
    'd_model': 128,
    'sample_rate': 500,
    'num_codebook_vectors': 512
}
model = create_unified_model(config)

# 编码EEG信号
eeg_signal = torch.randn(1, 64, 1000)  # [batch, channels, time]
encoded = model.encode_eeg(eeg_signal)

# 生成文本描述
texts = model.generate_from_eeg(eeg_signal)
```

## 配置说明

### 主要配置参数

```yaml
# 模型配置
model:
  n_channels: 64          # EEG通道数
  d_model: 128           # 模型维度
  sample_rate: 500       # 采样率
  wavelet: 'db4'         # 小波类型
  wavelet_levels: 5      # 小波分解级数
  num_codebook_vectors: 512  # 码本大小

# 训练配置
training:
  num_epochs: 100
  batch_size: 16
  learning_rate: 1e-4
  random_seeds: [42, 2024, 3407]  # 论文中的随机种子
  max_sequence_length: 10000      # 20s * 500Hz

# 损失权重（论文公式7）
  alpha: 1.0      # MSE权重
  beta: 1.0       # 相关性权重  
  lambda1: 1.0    # 近似系数权重
  lambda2: 1.0    # 细节系数权重
  gamma: 1.0      # 对比学习权重
  delta: 0.5      # 掩码重建权重
```

## 项目结构

```
EEG-WaveletVQ-Framework/
├── src/
│   ├── models/
│   │   ├── wavelet_vq_tokenizer.py    # WaveletVQ编码器
│   │   ├── parallel_encoders.py       # 并行特征编码器
│   │   ├── cross_modal_alignment.py   # 跨模态对齐
│   │   └── unified_framework.py       # 统一框架
│   ├── data/
│   │   └── datasets.py                # 数据集类
│   ├── training/
│   │   └── trainer.py                 # 训练器
│   ├── evaluation/
│   │   └── metrics.py                 # 评估指标
│   └── utils/
│       ├── logger.py                  # 日志工具
│       └── config.py                  # 配置工具
├── configs/
│   └── default_config.yaml            # 默认配置
├── scripts/
│   ├── train.py                       # 训练脚本
│   └── evaluate.py                    # 评估脚本
├── data/                              # 数据目录
├── experiments/                       # 实验结果
└── README.md
```

## 评估指标

### 分类指标
- Accuracy: 准确率
- F1 Score: F1分数  
- Precision: 精确率
- Recall: 召回率

### 文本生成指标
- BLEU-4: 文本生成质量
- ROUGE-1/2/L: 文本摘要质量

### 跨模态对齐指标
- Recall@K: Top-K召回率
- 相似度分析

### EEG重建指标
- MSE: 均方误差
- 相关系数: 信号相关性
- SNR: 信噪比


## 许可证

本项目遵循 MIT 许可证。


