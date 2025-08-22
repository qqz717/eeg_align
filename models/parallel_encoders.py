import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class TemporalEncoder(nn.Module):
    """冻结的时间编码器用于提取时间动态"""
    
    def __init__(self, d_model=128, num_layers=4, freeze_weights=True):
        super().__init__()
        
        # 多层1D卷积块
        self.conv_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = 1 if i == 0 else d_model
            out_channels = d_model
            
            block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            self.conv_blocks.append(block)
        
        # 全局时间池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.temporal_projection = nn.Linear(d_model, d_model)
        
        if freeze_weights:
            self._freeze_parameters()
    
    def _freeze_parameters(self):
        """冻结所有参数以用作预训练编码器"""
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入EEG片段 [B, C, T]
        Returns:
            时间特征 [B, C, D]
        """
        batch_size, n_channels, time_steps = x.shape
        
        # 独立处理每个通道
        channel_features = []
        for c in range(n_channels):
            channel_signal = x[:, c:c+1, :]  # [B, 1, T]
            
            # 应用卷积块
            features = channel_signal
            for block in self.conv_blocks:
                features = block(features)
            
            # 全局池化和投影
            pooled = self.global_pool(features).squeeze(-1)  # [B, D]
            projected = self.temporal_projection(pooled)
            
            channel_features.append(projected)
        
        # 堆叠通道特征
        temporal_features = torch.stack(channel_features, dim=1)  # [B, C, D]
        
        return temporal_features


class SpatialEncoder(nn.Module):
    """带有脑区分组和全局通道交互的空间编码器"""
    
    def __init__(self, d_model=128, n_channels=64):
        super().__init__()
        
        # 脑区分组（基于10-20系统）
        self.region_groups = {
            'frontal': [0, 1, 2, 3, 4, 5, 6, 7],           # F通道
            'central': [8, 9, 10, 11, 12, 13],              # C通道  
            'parietal': [14, 15, 16, 17, 18, 19, 20],       # P通道
            'temporal': [21, 22, 23, 24, 25, 26],           # T通道
            'occipital': [27, 28, 29, 30]                   # O通道
        }
        
        # 每个脑区的局部处理
        self.local_processors = nn.ModuleDict()
        for region, channels in self.region_groups.items():
            n_region_channels = len(channels)
            self.local_processors[region] = nn.Sequential(
                nn.Linear(n_region_channels, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model // 2)
            )
        
        # 通过通道级Transformer的全局通道交互
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # 空间特征投影
        self.spatial_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, D] 或 [B, C, T, D]
        Returns:
            空间特征 [B, C, D]
        """
        if x.dim() == 4:
            # 如果4D输入，在时间维度上池化
            x = x.mean(dim=2)  # [B, C, D]
        
        batch_size, n_channels, d_model = x.shape
        
        # 局部脑区处理
        local_features = []
        for region, channel_indices in self.region_groups.items():
            if max(channel_indices) < n_channels:
                # 提取区域通道
                region_data = x[:, channel_indices, :].mean(dim=-1)  # [B, n_region_channels]
                
                # 局部处理
                region_features = self.local_processors[region](region_data)  # [B, D//2]
                local_features.append(region_features)
        
        # 连接局部特征
        if local_features:
            local_concat = torch.cat(local_features, dim=-1)  # [B, total_local_dim]
            
            # 填充或截断以匹配d_model
            if local_concat.shape[-1] < d_model:
                padding = d_model - local_concat.shape[-1]
                local_concat = F.pad(local_concat, (0, padding))
            elif local_concat.shape[-1] > d_model:
                local_concat = local_concat[:, :d_model]
            
            local_features_reshaped = local_concat.unsqueeze(1).expand(-1, n_channels, -1)
        else:
            local_features_reshaped = torch.zeros_like(x)
        
        # 全局通道交互
        global_features = self.global_transformer(x)  # [B, C, D]
        
        # 合并局部和全局特征
        combined_features = (local_features_reshaped + global_features) / 2
        
        # 最终投影
        spatial_features = self.spatial_projection(combined_features)
        
        return spatial_features


class SemanticEncoder(nn.Module):
    """带有FFT和频段注意力的语义编码器"""
    
    def __init__(self, d_model=128, n_channels=64, sample_rate=500):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.d_model = d_model
        
        # 频段定义（Hz）
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # 通道注意力用于选择关键脑区
        self.channel_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # 频谱注意力用于选择关键频段
        self.spectral_attention = nn.Sequential(
            nn.Linear(len(self.freq_bands), d_model // 2),
            nn.ReLU(), 
            nn.Linear(d_model // 2, len(self.freq_bands)),
            nn.Softmax(dim=-1)
        )
        
        # 特征变换层
        self.feature_transform = nn.Sequential(
            nn.Linear(len(self.freq_bands), d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
    def extract_frequency_features(self, x: torch.Tensor) -> torch.Tensor:
        """为每个频段提取功率谱特征"""
        batch_size, n_channels, time_steps = x.shape
        
        # 应用FFT
        fft_result = torch.fft.rfft(x, dim=-1)
        power_spectrum = torch.abs(fft_result) ** 2
        
        # 频率箱
        freqs = torch.fft.rfftfreq(time_steps, 1/self.sample_rate).to(x.device)
        
        # 提取每个频段的功率
        band_powers = []
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # 找到此频段的频率索引
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            
            if freq_mask.sum() > 0:
                # 此频段的平均功率
                band_power = power_spectrum[:, :, freq_mask].mean(dim=-1)
            else:
                band_power = torch.zeros(batch_size, n_channels, device=x.device)
            
            band_powers.append(band_power)
        
        # 堆叠频段功率: [B, C, n_bands]
        band_features = torch.stack(band_powers, dim=-1)
        
        return band_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入EEG信号 [B, C, T] 或预计算特征 [B, C, D]
        Returns:
            语义特征 [B, C, D]
        """
        batch_size, n_channels = x.shape[:2]
        
        if x.dim() == 3:
            # 从原始信号提取频率特征
            freq_features = self.extract_frequency_features(x)  # [B, C, n_bands]
        else:
            # 假设x已经是处理过的特征 [B, C, D]
            # 为演示创建虚拟频率特征
            freq_features = torch.randn(
                batch_size, n_channels, len(self.freq_bands), 
                device=x.device
            )
        
        # 通道注意力权重
        # 使用平均频率特征进行通道注意力
        channel_context = freq_features.mean(dim=-1, keepdim=True)  # [B, C, 1]
        
        # 为注意力计算扩展
        channel_features_for_attn = channel_context.expand(-1, -1, self.d_model)
        channel_weights = self.channel_attention(channel_features_for_attn)  # [B, C, 1]
        
        # 频谱注意力权重
        # 使用跨通道平均进行频谱注意力
        spectral_context = freq_features.mean(dim=1)  # [B, n_bands]
        spectral_weights = self.spectral_attention(spectral_context)  # [B, n_bands]
        
        # 应用注意力权重
        # 扩展频谱权重用于广播
        spectral_weights_expanded = spectral_weights.unsqueeze(1).expand(-1, n_channels, -1)
        
        # 加权频率特征
        weighted_freq_features = freq_features * spectral_weights_expanded
        
        # 变换到语义空间
        semantic_features = self.feature_transform(weighted_freq_features)  # [B, C, D]
        
        # 应用通道注意力
        semantic_features = semantic_features * channel_weights
        
        # 残差连接（如果输入有相同维度）
        if x.dim() == 4 and x.shape[-1] == self.d_model:
            semantic_features = semantic_features + x.mean(dim=2)  # 池化时间维度
        elif x.dim() == 3 and x.shape[-1] == self.d_model:
            semantic_features = semantic_features + x
        
        # 层归一化
        semantic_features = self.layer_norm(semantic_features)
        
        return semantic_features


class ParallelFeatureEncoder(nn.Module):
    """用于时间、空间和语义特征的统一并行编码"""
    
    def __init__(self, d_model=128, n_channels=64, sample_rate=500):
        super().__init__()
        
        self.d_model = d_model
        
        # 三个并行编码器
        self.temporal_encoder = TemporalEncoder(d_model, freeze_weights=True)
        self.spatial_encoder = SpatialEncoder(d_model, n_channels)
        self.semantic_encoder = SemanticEncoder(d_model, n_channels, sample_rate)
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入EEG信号 [B, C, T]
        Returns:
            包含所有特征表示的字典
        """
        # 并行特征提取
        temporal_features = self.temporal_encoder(x)      # [B, C, D]
        spatial_features = self.spatial_encoder(x)        # [B, C, D]  
        semantic_features = self.semantic_encoder(x)      # [B, C, D]
        
        # 连接特征
        concatenated = torch.cat([
            temporal_features,
            spatial_features, 
            semantic_features
        ], dim=-1)  # [B, C, 3*D]
        
        # 融合
        fused_features = self.fusion_layer(concatenated)  # [B, C, D]
        
        # 最终投影
        output_features = self.output_projection(fused_features)
        
        return {
            'temporal_features': temporal_features,
            'spatial_features': spatial_features,
            'semantic_features': semantic_features,
            'fused_features': output_features,
            'concatenated_features': concatenated
        }


def create_channel_ids(n_channels: int, device: torch.device = None) -> torch.Tensor:
    """为位置编码创建通道ID张量"""
    channel_ids = torch.arange(n_channels, device=device)
    return channel_ids.unsqueeze(0)  # [1, C]