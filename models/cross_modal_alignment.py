import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """文本编码器使用多语言E5模型"""
    
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct", d_model=1024, freeze_weights=True):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)
        self.d_model = d_model
        
        # 投影层将文本特征映射到EEG特征维度
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        if freeze_weights:
            self._freeze_text_model()
    
    def _freeze_text_model(self):
        """冻结文本模型参数"""
        for param in self.text_model.parameters():
            param.requires_grad = False
    
    def forward(self, texts: List[str], device: torch.device = None) -> torch.Tensor:
        """
        Args:
            texts: 文本描述列表
            device: 设备
        Returns:
            文本嵌入 [B, D]
        """
        if device is None:
            device = next(self.parameters()).device
        
        # 添加指令前缀（E5模型需要）
        texts = [f"query: {text}" for text in texts]
        
        # 分词
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        # 编码
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            # 使用池化的序列表示
            text_embeddings = outputs.pooler_output  # [B, hidden_size]
        
        # 投影到EEG特征空间
        projected_embeddings = self.text_projection(text_embeddings)  # [B, d_model]
        
        return projected_embeddings


class ContrastiveLearning(nn.Module):
    """对比学习模块用于EEG-文本对齐"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, eeg_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_features: EEG嵌入 [B, D]
            text_features: 文本嵌入 [B, D]
        Returns:
            对比学习损失
        """
        batch_size = eeg_features.shape[0]
        
        # 归一化特征
        eeg_features = F.normalize(eeg_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # 计算余弦相似度矩阵
        similarity_matrix = torch.matmul(eeg_features, text_features.T) / self.temperature
        
        # 创建标签（对角线为正样本）
        labels = torch.arange(batch_size).to(eeg_features.device)
        
        # EEG到文本的损失
        loss_eeg_to_text = F.cross_entropy(similarity_matrix, labels)
        
        # 文本到EEG的损失
        loss_text_to_eeg = F.cross_entropy(similarity_matrix.T, labels)
        
        # 总对比损失
        contrastive_loss = (loss_eeg_to_text + loss_text_to_eeg) / 2
        
        return contrastive_loss


class MaskedLanguageModelHead(nn.Module):
    """掩码语言建模头用于重建任务"""
    
    def __init__(self, d_model, vocab_size, num_codebook_vectors):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_codebook_vectors = num_codebook_vectors
        
        # EEG码本预测头
        self.eeg_prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, num_codebook_vectors)
        )
        
        # 文本生成头（可选）
        self.text_generation_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, vocab_size)
        )
        
    def forward(self, features: torch.Tensor, task: str = "eeg") -> torch.Tensor:
        """
        Args:
            features: 输入特征 [B, seq_len, D] 或 [B, D]
            task: "eeg" 用于EEG码本预测, "text" 用于文本生成
        Returns:
            预测logits
        """
        if task == "eeg":
            return self.eeg_prediction_head(features)
        elif task == "text":
            return self.text_generation_head(features)
        else:
            raise ValueError(f"Unknown task: {task}")


class CrossModalAligner(nn.Module):
    """跨模态对齐器结合对比学习和掩码重建"""
    
    def __init__(
        self, 
        d_model=128, 
        num_codebook_vectors=512, 
        text_vocab_size=50000,
        temperature=0.07,
        mask_prob=0.15
    ):
        super().__init__()
        
        self.d_model = d_model
        self.mask_prob = mask_prob
        
        # 文本编码器
        self.text_encoder = TextEncoder(d_model=d_model)
        
        # 对比学习模块
        self.contrastive_learning = ContrastiveLearning(temperature)
        
        # 掩码重建头
        self.mlm_head = MaskedLanguageModelHead(d_model, text_vocab_size, num_codebook_vectors)
        
        # EEG特征投影（用于对比学习）
        self.eeg_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer用于序列建模
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
    def create_eeg_mask(self, eeg_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """为EEG特征创建随机掩码"""
        batch_size, seq_len, d_model = eeg_features.shape
        
        # 创建掩码（True表示被掩码的位置）
        mask = torch.rand(batch_size, seq_len) < self.mask_prob
        mask = mask.to(eeg_features.device)
        
        # 保存原始特征用于重建目标
        masked_features = eeg_features.clone()
        
        # 用零向量替换被掩码的位置
        masked_features[mask] = 0
        
        return masked_features, mask
    
    def forward(
        self, 
        eeg_features: torch.Tensor, 
        texts: List[str], 
        codebook_indices: Optional[torch.Tensor] = None,
        mode: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            eeg_features: EEG特征 [B, C, D] 或 [B, C, T, D]
            texts: 文本描述列表
            codebook_indices: EEG码本索引 [B, C, T] (可选)
            mode: "train" 或 "eval"
        Returns:
            损失和预测字典
        """
        device = eeg_features.device
        batch_size = eeg_features.shape[0]
        
        # 处理EEG特征维度
        if eeg_features.dim() == 4:  # [B, C, T, D]
            B, C, T, D = eeg_features.shape
            # 重塑为序列格式
            eeg_seq = eeg_features.view(B, C * T, D)
        elif eeg_features.dim() == 3:  # [B, C, D]
            eeg_seq = eeg_features
        else:
            raise ValueError(f"Invalid EEG features dimension: {eeg_features.dim()}")
        
        # 编码文本
        text_features = self.text_encoder(texts, device)  # [B, D]
        
        results = {}
        
        if mode == "train":
            # 对比学习
            # 池化EEG特征用于对比学习
            eeg_pooled = torch.mean(eeg_seq, dim=1)  # [B, D]
            eeg_projected = self.eeg_projection(eeg_pooled)
            
            contrastive_loss = self.contrastive_learning(eeg_projected, text_features)
            results['contrastive_loss'] = contrastive_loss
            
            # 掩码重建
            if codebook_indices is not None:
                # 创建EEG掩码
                masked_eeg, mask = self.create_eeg_mask(eeg_seq)
                
                # Transformer编码
                encoded_features = self.transformer(masked_eeg)
                
                # 预测被掩码的EEG码本
                eeg_predictions = self.mlm_head(encoded_features, task="eeg")
                
                # 重建损失（仅在被掩码位置计算）
                if codebook_indices.dim() == 3:  # [B, C, T]
                    target_indices = codebook_indices.view(batch_size, -1)  # [B, C*T]
                else:
                    target_indices = codebook_indices
                
                # 确保预测和目标形状匹配
                if eeg_predictions.shape[1] != target_indices.shape[1]:
                    min_len = min(eeg_predictions.shape[1], target_indices.shape[1])
                    eeg_predictions = eeg_predictions[:, :min_len]
                    target_indices = target_indices[:, :min_len]
                    mask = mask[:, :min_len]
                
                # 计算掩码位置的交叉熵损失
                mask_expanded = mask.unsqueeze(-1).expand_as(eeg_predictions)
                masked_predictions = eeg_predictions[mask_expanded].view(-1, eeg_predictions.shape[-1])
                masked_targets = target_indices[mask]
                
                if masked_predictions.shape[0] > 0:
                    reconstruction_loss = F.cross_entropy(masked_predictions, masked_targets)
                else:
                    reconstruction_loss = torch.tensor(0.0, device=device)
                
                results['reconstruction_loss'] = reconstruction_loss
                results['predicted_indices'] = torch.argmax(eeg_predictions, dim=-1)
            
            # 总损失
            total_loss = contrastive_loss
            if 'reconstruction_loss' in results:
                total_loss = total_loss + results['reconstruction_loss']
            
            results['total_loss'] = total_loss
        
        else:  # eval模式
            # 计算相似度用于评估
            eeg_pooled = torch.mean(eeg_seq, dim=1)
            eeg_projected = self.eeg_projection(eeg_pooled)
            
            # 归一化特征
            eeg_norm = F.normalize(eeg_projected, p=2, dim=-1)
            text_norm = F.normalize(text_features, p=2, dim=-1)
            
            # 计算余弦相似度
            similarities = torch.matmul(eeg_norm, text_norm.T)
            
            results['similarities'] = similarities
            results['eeg_features'] = eeg_projected
            results['text_features'] = text_features
        
        return results
    
    def generate_text_from_eeg(self, eeg_features: torch.Tensor, max_length: int = 50) -> List[str]:
        """从EEG特征生成文本描述（简化版本）"""
        device = eeg_features.device
        batch_size = eeg_features.shape[0]
        
        # 池化EEG特征
        if eeg_features.dim() == 4:
            eeg_pooled = torch.mean(eeg_features.view(batch_size, -1, eeg_features.shape[-1]), dim=1)
        elif eeg_features.dim() == 3:
            eeg_pooled = torch.mean(eeg_features, dim=1)
        else:
            eeg_pooled = eeg_features
        
        # 投影到文本空间
        eeg_projected = self.eeg_projection(eeg_pooled)
        
        # 简化的文本生成（实际实现需要更复杂的解码策略）
        # 这里返回占位符文本
        generated_texts = []
        for i in range(batch_size):
            # 实际实现中，这里应该使用语言模型解码
            generated_texts.append(f"Generated description for EEG sample {i}")
        
        return generated_texts


def compute_alignment_accuracy(similarities: torch.Tensor, k: int = 1) -> float:
    """计算对齐准确率"""
    batch_size = similarities.shape[0]
    
    # 找到每个EEG样本最相似的文本
    _, top_k_indices = torch.topk(similarities, k, dim=-1)
    
    # 计算准确率（对角线元素应该在top-k中）
    correct = 0
    for i in range(batch_size):
        if i in top_k_indices[i]:
            correct += 1
    
    accuracy = correct / batch_size
    return accuracy


if __name__ == "__main__":
    # 测试跨模态对齐器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建虚拟数据
    batch_size = 4
    n_channels = 32
    time_steps = 100
    d_model = 128
    
    eeg_features = torch.randn(batch_size, n_channels, time_steps, d_model).to(device)
    texts = [
        "Patient shows normal brain activity",
        "Seizure detected in temporal lobe", 
        "Alpha waves dominant during rest",
        "Motor cortex activation observed"
    ]
    codebook_indices = torch.randint(0, 512, (batch_size, n_channels, time_steps)).to(device)
    
    # 初始化对齐器
    aligner = CrossModalAligner(d_model=d_model).to(device)
    
    # 前向传播
    results = aligner(eeg_features, texts, codebook_indices, mode="train")
    
    print("Cross-Modal Alignment Results:")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape if value.dim() > 0 else value.item()}")
        else:
            print(f"{key}: {value}")