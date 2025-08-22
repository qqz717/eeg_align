import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from .wavelet_vq_tokenizer import WaveletVQEncoder, create_channel_weights, create_channel_ids
from .parallel_encoders import ParallelFeatureEncoder
from .cross_modal_alignment import CrossModalAligner


class UnifiedEEGFramework(nn.Module):
    """
    统一EEG表示学习框架
    结合小波神经量化和动态语义对齐
    """
    
    def __init__(
        self,
        # 模型配置
        n_channels: int = 64,
        d_model: int = 128,
        sample_rate: int = 500,
        # 小波配置
        wavelet: str = 'db4',
        wavelet_levels: int = 5,
        num_codebook_vectors: int = 512,
        # 对齐配置
        temperature: float = 0.07,
        mask_prob: float = 0.15,
        # 损失权重
        alpha: float = 1.0,  # MSE权重
        beta: float = 1.0,   # 相关性权重  
        lambda1: float = 1.0,  # 近似系数权重
        lambda2: float = 1.0,  # 细节系数权重
        gamma: float = 1.0,    # 对比学习权重
        delta: float = 0.5     # 重建权重
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.d_model = d_model
        self.sample_rate = sample_rate
        
        # 损失权重
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma = gamma
        self.delta = delta
        
        # 阶段1: WaveletVQ神经令牌化器
        self.wavelet_vq_encoder = WaveletVQEncoder(
            n_channels=n_channels,
            d_model=d_model,
            num_codebook_vectors=num_codebook_vectors,
            wavelet=wavelet,
            levels=wavelet_levels
        )
        
        # 阶段2: 并行特征编码
        self.parallel_encoder = ParallelFeatureEncoder(
            d_model=d_model,
            n_channels=n_channels,
            sample_rate=sample_rate
        )
        
        # 阶段3: 跨模态对齐
        self.cross_modal_aligner = CrossModalAligner(
            d_model=d_model,
            num_codebook_vectors=num_codebook_vectors,
            temperature=temperature,
            mask_prob=mask_prob
        )
        
    def forward(
        self,
        eeg_signal: torch.Tensor,
        texts: Optional[List[str]] = None,
        mode: str = "train",
        stage: str = "full"  # "stage1", "stage2", "full"
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            eeg_signal: 输入EEG信号 [B, C, T]
            texts: 对应的文本描述列表
            mode: "train" 或 "eval"
            stage: 训练阶段 "stage1"(只VQ), "stage2"(并行编码), "full"(端到端)
        Returns:
            包含所有输出和损失的字典
        """
        batch_size, n_channels, time_steps = eeg_signal.shape
        device = eeg_signal.device
        
        # 创建通道ID
        channel_ids = create_channel_ids(n_channels, device).expand(batch_size, -1)
        
        results = {}
        
        # 阶段1: WaveletVQ编码
        if stage in ["stage1", "full"]:
            vq_results = self.wavelet_vq_encoder(eeg_signal, channel_ids)
            
            results.update({
                'vq_quantized': vq_results['quantized'],
                'vq_reconstructed': vq_results['reconstructed'],
                'codebook_loss': vq_results['codebook_loss'],
                'commitment_loss': vq_results['commitment_loss'],
                'encoding_indices': vq_results['encoding_indices'],
                'perplexity': vq_results['perplexity']
            })
            
            # 计算重建损失
            reconstruction_loss = self._compute_reconstruction_loss(
                eeg_signal, vq_results['reconstructed']
            )
            results['reconstruction_loss'] = reconstruction_loss
            
            # 阶段1总损失
            stage1_loss = (
                vq_results['codebook_loss'] + 
                vq_results['commitment_loss'] + 
                reconstruction_loss['total_loss']
            )
            results['stage1_loss'] = stage1_loss
            
            if stage == "stage1":
                return results
        
        # 阶段2: 并行特征编码
        if stage in ["stage2", "full"]:
            parallel_results = self.parallel_encoder(eeg_signal)
            
            results.update({
                'temporal_features': parallel_results['temporal_features'],
                'spatial_features': parallel_results['spatial_features'],
                'semantic_features': parallel_results['semantic_features'],
                'fused_features': parallel_results['fused_features']
            })
            
            if stage == "stage2":
                return results
        
        # 阶段3: 跨模态对齐（完整训练）
        if stage == "full" and texts is not None:
            # 使用融合特征进行对齐
            fused_features = results['fused_features']  # [B, C, D]
            codebook_indices = results['encoding_indices']  # [B, C, T']
            
            alignment_results = self.cross_modal_aligner(
                fused_features,
                texts, 
                codebook_indices,
                mode=mode
            )
            
            results.update(alignment_results)
            
            # 计算总损失
            if mode == "train":
                total_loss = self._compute_total_loss(results)
                results['total_loss'] = total_loss
        
        return results
    
    def _compute_reconstruction_loss(
        self, 
        original: torch.Tensor, 
        reconstructed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算重建损失（MSE + 相关性 + 时频结构约束）"""
        
        # MSE损失
        mse_loss = F.mse_loss(original, reconstructed)
        
        # 归一化互相关损失
        def normalized_cross_correlation(x, x_hat):
            B, C, T = x.shape
            ncc_losses = []
            
            for b in range(B):
                for c in range(C):
                    signal = x[b, c]
                    recon = x_hat[b, c]
                    
                    # 计算NCC
                    mean_s = torch.mean(signal)
                    mean_r = torch.mean(recon)
                    
                    numerator = torch.sum((signal - mean_s) * (recon - mean_r))
                    denominator = torch.sqrt(
                        torch.sum((signal - mean_s) ** 2) * 
                        torch.sum((recon - mean_r) ** 2) + 1e-8
                    )
                    
                    ncc = numerator / denominator
                    ncc_losses.append(1 - ncc)  # 转换为损失
            
            return torch.mean(torch.stack(ncc_losses))
        
        corr_loss = normalized_cross_correlation(original, reconstructed)
        
        # 时频结构约束损失（简化版本）
        # 使用FFT计算频域损失
        original_fft = torch.fft.rfft(original, dim=-1)
        reconstructed_fft = torch.fft.rfft(reconstructed, dim=-1)
        
        # 近似系数损失（低频）
        freq_bins = original_fft.shape[-1]
        low_freq_bins = freq_bins // 4  # 前25%频率作为低频
        approx_loss = F.mse_loss(
            torch.abs(original_fft[..., :low_freq_bins]),
            torch.abs(reconstructed_fft[..., :low_freq_bins])
        )
        
        # 细节系数损失（高频）
        detail_loss = F.mse_loss(
            torch.abs(original_fft[..., low_freq_bins:]),
            torch.abs(reconstructed_fft[..., low_freq_bins:])
        )
        
        # 时频结构损失
        tfs_loss = self.lambda1 * approx_loss + self.lambda2 * detail_loss
        
        # 总重建损失
        total_reconstruction_loss = (
            self.alpha * mse_loss + 
            self.beta * corr_loss + 
            tfs_loss
        )
        
        return {
            'mse_loss': mse_loss,
            'corr_loss': corr_loss,
            'approx_loss': approx_loss,
            'detail_loss': detail_loss,
            'tfs_loss': tfs_loss,
            'total_loss': total_reconstruction_loss
        }
    
    def _compute_total_loss(self, results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算总损失"""
        losses = []
        
        # VQ损失
        if 'codebook_loss' in results:
            losses.append(results['codebook_loss'])
        if 'commitment_loss' in results:
            losses.append(results['commitment_loss'])
        
        # 重建损失
        if 'reconstruction_loss' in results:
            losses.append(results['reconstruction_loss'])
        
        # 对比学习损失
        if 'contrastive_loss' in results:
            losses.append(self.gamma * results['contrastive_loss'])
        
        # 掩码重建损失
        if 'reconstruction_loss' in results and 'contrastive_loss' in results:
            # 区分VQ重建损失和掩码重建损失
            if hasattr(results['reconstruction_loss'], 'item'):
                # 这是掩码重建损失
                losses.append(self.delta * results['reconstruction_loss'])
        
        total_loss = sum(losses) if losses else torch.tensor(0.0)
        
        return total_loss
    
    def encode_eeg(self, eeg_signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """编码EEG信号到统一表示"""
        with torch.no_grad():
            results = self.forward(eeg_signal, mode="eval", stage="full")
        
        return {
            'quantized_features': results.get('vq_quantized'),
            'fused_features': results.get('fused_features'),
            'encoding_indices': results.get('encoding_indices')
        }
    
    def generate_from_eeg(self, eeg_signal: torch.Tensor) -> List[str]:
        """从EEG信号生成文本描述"""
        with torch.no_grad():
            results = self.forward(eeg_signal, mode="eval", stage="full")
            fused_features = results.get('fused_features')
            
            if fused_features is not None:
                generated_texts = self.cross_modal_aligner.generate_text_from_eeg(fused_features)
                return generated_texts
            else:
                return ["No text generated"]
    
    def compute_eeg_text_similarity(
        self, 
        eeg_signal: torch.Tensor, 
        texts: List[str]
    ) -> torch.Tensor:
        """计算EEG信号和文本的相似度"""
        with torch.no_grad():
            results = self.forward(eeg_signal, texts, mode="eval", stage="full")
            similarities = results.get('similarities')
            
            return similarities if similarities is not None else torch.tensor([])


def create_unified_model(config: dict) -> UnifiedEEGFramework:
    """根据配置创建统一模型"""
    return UnifiedEEGFramework(**config)


def load_pretrained_weights(model: UnifiedEEGFramework, checkpoint_path: str):
    """加载预训练权重"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 加载兼容的权重
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    print(f"Loaded {len(filtered_dict)} parameters from {checkpoint_path}")


if __name__ == "__main__":
    # 测试统一框架
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型配置
    config = {
        'n_channels': 32,
        'd_model': 128,
        'sample_rate': 500,
        'wavelet': 'db4',
        'wavelet_levels': 5,
        'num_codebook_vectors': 512,
        'temperature': 0.07,
        'mask_prob': 0.15
    }
    
    # 创建模型
    model = create_unified_model(config).to(device)
    
    # 测试数据
    batch_size = 4
    n_channels = config['n_channels']
    time_steps = 1000  # 2秒 @ 500Hz
    
    eeg_data = torch.randn(batch_size, n_channels, time_steps).to(device)
    texts = [
        "正常脑电活动模式",
        "颞叶检测到癫痫活动", 
        "静息状态下alpha波占主导",
        "观察到运动皮层激活"
    ]
    
    print("Testing Unified EEG Framework:")
    print(f"Input shape: {eeg_data.shape}")
    
    # 阶段1训练
    results_stage1 = model(eeg_data, mode="train", stage="stage1")
    print(f"\nStage 1 Results:")
    print(f"Stage 1 Loss: {results_stage1['stage1_loss'].item():.4f}")
    print(f"Codebook Loss: {results_stage1['codebook_loss'].item():.4f}")
    print(f"Perplexity: {results_stage1['perplexity'].item():.4f}")
    
    # 完整训练
    results_full = model(eeg_data, texts, mode="train", stage="full")
    print(f"\nFull Training Results:")
    if 'total_loss' in results_full:
        print(f"Total Loss: {results_full['total_loss'].item():.4f}")
    if 'contrastive_loss' in results_full:
        print(f"Contrastive Loss: {results_full['contrastive_loss'].item():.4f}")
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Framework initialization successful!")