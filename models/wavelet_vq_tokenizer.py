import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from typing import Dict, List, Tuple, Optional


class WaveletDecomposer(nn.Module):
    """离散小波变换分解器用于EEG信号"""
    
    def __init__(self, wavelet='db4', levels=5):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: 输入EEG信号 [B, C, T]
        Returns:
            approx_coeffs: 近似系数 [B, C, T//2^levels]
            detail_coeffs: 每层的细节系数列表
        """
        batch_size, channels, time_steps = x.shape
        
        # 对每个通道分别应用DWT
        approx_coeffs_list = []
        detail_coeffs_list = [[] for _ in range(self.levels)]
        
        for b in range(batch_size):
            for c in range(channels):
                signal = x[b, c].cpu().numpy()
                
                # 执行多级DWT
                coeffs = pywt.wavedec(signal, self.wavelet, level=self.levels)
                approx = coeffs[0]  # 近似系数
                details = coeffs[1:]  # 细节系数
                
                approx_coeffs_list.append(torch.tensor(approx, dtype=x.dtype, device=x.device))
                
                for level, detail in enumerate(details):
                    detail_coeffs_list[level].append(
                        torch.tensor(detail, dtype=x.dtype, device=x.device)
                    )
        
        # 堆叠系数
        approx_coeffs = torch.stack(approx_coeffs_list).reshape(batch_size, channels, -1)
        detail_coeffs = []
        for level in range(self.levels):
            detail_level = torch.stack(detail_coeffs_list[level]).reshape(batch_size, channels, -1)
            detail_coeffs.append(detail_level)
            
        return approx_coeffs, detail_coeffs


class ChannelwiseTransformer(nn.Module):
    """通道级Transformer用于建模跨通道关系"""
    
    def __init__(self, d_model=128, nhead=8, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, T, D] 其中 D = d_model
        Returns:
            通道级变换特征 [B, C, T, D]
        """
        B, C, T, D = x.shape
        
        # 重塑以将通道作为序列维度
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * T, C, D)
        
        # 跨通道应用transformer
        transformed = self.transformer(x_reshaped)
        
        # 重塑回原始形状
        output = transformed.reshape(B, T, C, D).permute(0, 2, 1, 3)
        
        return output


class TemporalSpatialConv(nn.Module):
    """级联时间和空间卷积与门控机制"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        
        # 带扩张的时间卷积
        self.temporal_conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2
        )
        
        # 空间深度可分离卷积
        self.spatial_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1, groups=out_channels)
        
        # 门控机制
        self.gate_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 [B, C, T]
        Returns:
            融合的时空特征 [B, C, T]
        """
        # 时间卷积
        temporal_out = self.temporal_conv(x)
        temporal_out = self.norm1(temporal_out)
        temporal_out = F.gelu(temporal_out)
        
        # 空间卷积
        spatial_out = self.spatial_conv(temporal_out)
        spatial_out = self.norm2(spatial_out)
        
        # 门控机制
        gate = torch.sigmoid(self.gate_conv(spatial_out))
        output = spatial_out * gate + temporal_out * (1 - gate)
        
        return output


class PositionalEncoder(nn.Module):
    """带有通道位置的正弦位置编码"""
    
    def __init__(self, d_model, max_len=5000, n_channels=64):
        super().__init__()
        self.d_model = d_model
        
        # 时间位置编码
        pe_temporal = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe_temporal[:, 0::2] = torch.sin(position * div_term)
        pe_temporal[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_temporal', pe_temporal.unsqueeze(0))
        
        # 通道位置编码
        self.channel_embedding = nn.Embedding(n_channels, d_model)
        
    def forward(self, x: torch.Tensor, channel_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, T, D]
            channel_ids: 通道索引 [B, C]
        Returns:
            位置编码特征 [B, C, T, D]
        """
        B, C, T, D = x.shape
        
        # 添加时间位置编码
        x = x + self.pe_temporal[:, :T, :].unsqueeze(1)
        
        # 添加通道位置编码
        channel_pos = self.channel_embedding(channel_ids).unsqueeze(2)
        x = x + channel_pos
        
        return x


class VectorQuantizer(nn.Module):
    """带有加权K-means和马哈拉诺比斯距离的向量量化"""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 码本
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # 可学习的马哈拉诺比斯距离协方差矩阵
        self.register_parameter('sigma_inv', nn.Parameter(torch.eye(embedding_dim)))
        
    def forward(self, inputs: torch.Tensor, weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: 输入特征 [B, C, T, D]
            weights: 通道-时间权重 [B, C, T]
        Returns:
            包含量化特征和损失的字典
        """
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        if weights is not None:
            flat_weights = weights.view(-1, 1)
        else:
            flat_weights = torch.ones(flat_input.shape[0], 1, device=inputs.device)
        
        # 计算加权马哈拉诺比斯距离
        distances = self._weighted_mahalanobis_distance(flat_input, flat_weights)
        
        # 找到最近的码本项
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, 
                              device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化并重塑
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(input_shape)
        
        # 计算损失
        codebook_loss = F.mse_loss(quantized.detach(), inputs) * flat_weights.mean()
        commitment_loss = F.mse_loss(quantized, inputs.detach()) * self.commitment_cost
        
        # 直通估计器
        quantized = inputs + (quantized - inputs).detach()
        
        return {
            'quantized': quantized,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'encoding_indices': encoding_indices.view(input_shape[:-1]),
            'perplexity': self._calculate_perplexity(encodings)
        }
    
    def _weighted_mahalanobis_distance(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """计算输入和码本之间的加权马哈拉诺比斯距离"""
        # x: [N, D], weights: [N, 1]
        # embeddings: [K, D]
        
        # 扩展用于广播
        x_expanded = x.unsqueeze(1)  # [N, 1, D]
        embeddings_expanded = self.embeddings.weight.unsqueeze(0)  # [1, K, D]
        
        # 计算差异
        diff = x_expanded - embeddings_expanded  # [N, K, D]
        
        # 应用带权重的马哈拉诺比斯距离
        sigma_inv = self.sigma_inv + self.sigma_inv.t()  # 确保对称性
        mahal_dist = torch.einsum('nkd,de,nke->nk', diff, sigma_inv, diff)
        
        # 应用权重
        weighted_dist = mahal_dist * weights
        
        return weighted_dist
    
    def _calculate_perplexity(self, encodings: torch.Tensor) -> torch.Tensor:
        """计算码本使用的困惑度"""
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity


class WaveletVQEncoder(nn.Module):
    """完整的WaveletVQ神经令牌化器"""
    
    def __init__(
        self,
        n_channels=64,
        d_model=128,
        num_codebook_vectors=512,
        wavelet='db4',
        levels=5
    ):
        super().__init__()
        
        self.wavelet_decomposer = WaveletDecomposer(wavelet, levels)
        
        # 特征投影
        self.input_projection = nn.Linear(1, d_model)
        
        # 通道级Transformer
        self.channel_transformer = ChannelwiseTransformer(d_model)
        
        # 时空卷积
        self.temp_spatial_conv = TemporalSpatialConv(d_model, d_model)
        
        # 位置编码器
        self.pos_encoder = PositionalEncoder(d_model, n_channels=n_channels)
        
        # 向量量化器
        self.vector_quantizer = VectorQuantizer(num_codebook_vectors, d_model)
        
        # 重建解码器
        self.decoder = nn.ModuleList([
            nn.ConvTranspose1d(d_model, d_model // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, d_model // 2),
            nn.GELU(),
            nn.ConvTranspose1d(d_model // 2, d_model // 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, d_model // 4),
            nn.GELU(),
            nn.ConvTranspose1d(d_model // 4, 1, kernel_size=4, stride=2, padding=1)
        ])
        
    def forward(self, x: torch.Tensor, channel_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入EEG [B, C, T]
            channel_ids: 通道索引 [B, C]
        Returns:
            包含量化特征和重建的字典
        """
        batch_size, n_channels, time_steps = x.shape
        
        # 小波分解
        approx_coeffs, detail_coeffs = self.wavelet_decomposer(x)
        
        # 合并近似和细节系数
        combined_coeffs = [approx_coeffs] + detail_coeffs
        
        # 处理每个系数级别
        processed_features = []
        for coeffs in combined_coeffs:
            # 投影到模型维度
            features = self.input_projection(coeffs.unsqueeze(-1))  # [B, C, T', D]
            
            # 通道级transformer
            features = self.channel_transformer(features)
            
            # 位置编码
            features = self.pos_encoder(features, channel_ids)
            
            # 时空卷积（重塑为1D卷积）
            B, C, T_coeff, D = features.shape
            features_flat = features.permute(0, 3, 1, 2).reshape(B * D, C, T_coeff)
            conv_out = self.temp_spatial_conv(features_flat)
            features = conv_out.reshape(B, D, C, T_coeff).permute(0, 2, 3, 1)
            
            processed_features.append(features)
        
        # 连接所有级别的特征
        concatenated = torch.cat(processed_features, dim=2)  # 沿时间维度连接
        
        # 向量量化
        vq_results = self.vector_quantizer(concatenated)
        
        # 重建
        quantized = vq_results['quantized']  # [B, C, T_total, D]
        B, C, T_total, D = quantized.shape
        
        # 重塑用于解码器
        quantized_flat = quantized.permute(0, 3, 1, 2).reshape(B * D, C, T_total)
        
        # 应用解码器层
        decoded = quantized_flat
        for layer in self.decoder:
            decoded = layer(decoded)
        
        # 重塑并聚合通道
        decoded = decoded.reshape(B, D, C, -1).mean(dim=1)  # 沿D维度平均
        reconstructed = decoded.mean(dim=1)  # 沿通道平均得到 [B, T_reconstructed]
        
        # 插值以匹配原始时间步长
        if reconstructed.shape[-1] != time_steps:
            reconstructed = F.interpolate(
                reconstructed.unsqueeze(1), 
                size=time_steps, 
                mode='linear', 
                align_corners=False
            ).squeeze(1)
        
        return {
            'quantized': quantized,
            'reconstructed': reconstructed.unsqueeze(1).expand(-1, n_channels, -1),
            'codebook_loss': vq_results['codebook_loss'],
            'commitment_loss': vq_results['commitment_loss'],
            'encoding_indices': vq_results['encoding_indices'],
            'perplexity': vq_results['perplexity']
        }


def create_channel_weights(eeg_signal: torch.Tensor, 
                          channel_importance: Optional[torch.Tensor] = None) -> torch.Tensor:
    """为EEG信号创建通道-时间权重"""
    B, C, T = eeg_signal.shape
    
    if channel_importance is None:
        # 默认：按信号方差加权
        signal_var = torch.var(eeg_signal, dim=-1, keepdim=True)  # [B, C, 1]
        weights = signal_var / (torch.mean(signal_var, dim=1, keepdim=True) + 1e-8)
    else:
        weights = channel_importance.unsqueeze(-1)  # [B, C, 1]
    
    # 扩展到时间维度
    weights = weights.expand(B, C, T)
    
    return weights