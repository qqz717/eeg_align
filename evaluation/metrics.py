import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer
import re


class EvaluationMetrics:
    """评估指标计算器"""
    
    def __init__(self):
        # 初始化ROUGE评分器
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # 确保nltk数据下载
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def compute_classification_metrics(
        self, 
        predictions: List[int], 
        targets: List[int]
    ) -> Dict[str, float]:
        """计算分类指标"""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'f1_score': f1_score(targets, predictions, average='weighted'),
            'precision': precision_score(targets, predictions, average='weighted', zero_division=0),
            'recall': recall_score(targets, predictions, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def compute_text_generation_metrics(
        self,
        generated_texts: List[str],
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """计算文本生成指标（BLEU, ROUGE）"""
        if not generated_texts or not reference_texts:
            return {'bleu4': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        # 确保长度匹配
        min_len = min(len(generated_texts), len(reference_texts))
        generated_texts = generated_texts[:min_len]
        reference_texts = reference_texts[:min_len]
        
        # BLEU评分
        bleu_scores = []
        for gen, ref in zip(generated_texts, reference_texts):
            gen_tokens = self.tokenize_text(gen)
            ref_tokens = self.tokenize_text(ref)
            
            if len(gen_tokens) > 0 and len(ref_tokens) > 0:
                bleu = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25))
                bleu_scores.append(bleu)
        
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        
        # ROUGE评分
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for gen, ref in zip(generated_texts, reference_texts):
            if gen and ref:
                scores = self.rouge_scorer.score(gen, ref)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # 计算平均ROUGE分数
        avg_rouge = {}
        for key, scores in rouge_scores.items():
            avg_rouge[key] = np.mean(scores) if scores else 0.0
        
        return {
            'bleu4': avg_bleu,
            'rouge1': avg_rouge['rouge1'],
            'rouge2': avg_rouge['rouge2'],
            'rougeL': avg_rouge['rougeL']
        }
    
    def tokenize_text(self, text: str) -> List[str]:
        """文本分词"""
        # 简单的分词（可以根据需要改进）
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        return tokens
    
    def compute_cross_modal_alignment_metrics(
        self,
        similarities: torch.Tensor,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """计算跨模态对齐指标"""
        if similarities.dim() != 2:
            raise ValueError("Similarities tensor must be 2D")
        
        batch_size = similarities.shape[0]
        metrics = {}
        
        # 计算Recall@K
        for k in k_values:
            if k <= similarities.shape[1]:
                # EEG到文本的Recall@K
                _, top_k_indices = torch.topk(similarities, k, dim=1)
                correct = 0
                for i in range(batch_size):
                    if i in top_k_indices[i]:
                        correct += 1
                
                recall_at_k = correct / batch_size
                metrics[f'recall_at_{k}'] = recall_at_k
        
        # 计算平均相似度
        diagonal_similarities = torch.diag(similarities)
        metrics['mean_positive_similarity'] = diagonal_similarities.mean().item()
        
        # 计算平均负样本相似度
        mask = torch.eye(batch_size, dtype=torch.bool).to(similarities.device)
        negative_similarities = similarities[~mask]
        metrics['mean_negative_similarity'] = negative_similarities.mean().item()
        
        return metrics
    
    def compute_eeg_reconstruction_metrics(
        self,
        original_eeg: torch.Tensor,
        reconstructed_eeg: torch.Tensor
    ) -> Dict[str, float]:
        """计算EEG重建指标"""
        # 确保形状匹配
        if original_eeg.shape != reconstructed_eeg.shape:
            raise ValueError("Original and reconstructed EEG must have the same shape")
        
        # MSE
        mse = torch.nn.functional.mse_loss(original_eeg, reconstructed_eeg).item()
        
        # MAE
        mae = torch.nn.functional.l1_loss(original_eeg, reconstructed_eeg).item()
        
        # 相关系数
        def compute_correlation(x, y):
            x_flat = x.view(-1)
            y_flat = y.view(-1)
            
            x_mean = torch.mean(x_flat)
            y_mean = torch.mean(y_flat)
            
            numerator = torch.sum((x_flat - x_mean) * (y_flat - y_mean))
            denominator = torch.sqrt(torch.sum((x_flat - x_mean) ** 2) * torch.sum((y_flat - y_mean) ** 2))
            
            if denominator == 0:
                return 0.0
            
            return (numerator / denominator).item()
        
        correlation = compute_correlation(original_eeg, reconstructed_eeg)
        
        # 信噪比 (SNR)
        signal_power = torch.mean(original_eeg ** 2)
        noise_power = torch.mean((original_eeg - reconstructed_eeg) ** 2)
        
        if noise_power > 0:
            snr_db = 10 * torch.log10(signal_power / noise_power).item()
        else:
            snr_db = float('inf')
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'snr_db': snr_db
        }
    
    def compute_codebook_usage_metrics(
        self,
        encoding_indices: torch.Tensor,
        num_codebook_vectors: int
    ) -> Dict[str, float]:
        """计算码本使用指标"""
        # 展平编码索引
        flat_indices = encoding_indices.view(-1)
        
        # 计算每个码本向量的使用次数
        usage_counts = torch.zeros(num_codebook_vectors)
        unique_indices, counts = torch.unique(flat_indices, return_counts=True)
        usage_counts[unique_indices] = counts.float()
        
        # 使用的码本向量数量
        num_used = (usage_counts > 0).sum().item()
        
        # 使用率
        usage_rate = num_used / num_codebook_vectors
        
        # 熵（衡量分布均匀性）
        probabilities = usage_counts / usage_counts.sum()
        probabilities = probabilities[probabilities > 0]  # 移除零概率
        entropy = -torch.sum(probabilities * torch.log(probabilities)).item()
        
        # 困惑度
        perplexity = torch.exp(torch.tensor(entropy)).item()
        
        return {
            'codebook_usage_rate': usage_rate,
            'codebook_entropy': entropy,
            'codebook_perplexity': perplexity,
            'num_used_vectors': num_used
        }
    
    def compute_frequency_domain_metrics(
        self,
        original_eeg: torch.Tensor,
        reconstructed_eeg: torch.Tensor,
        sample_rate: int = 500
    ) -> Dict[str, float]:
        """计算频域指标"""
        # FFT
        original_fft = torch.fft.rfft(original_eeg, dim=-1)
        reconstructed_fft = torch.fft.rfft(reconstructed_eeg, dim=-1)
        
        # 功率谱密度
        original_psd = torch.abs(original_fft) ** 2
        reconstructed_psd = torch.abs(reconstructed_fft) ** 2
        
        # 频域MSE
        freq_mse = torch.nn.functional.mse_loss(reconstructed_psd, original_psd).item()
        
        # 频率bins
        freqs = torch.fft.rfftfreq(original_eeg.shape[-1], 1/sample_rate)
        
        # 分频段误差
        freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        band_errors = {}
        for band_name, (low, high) in freq_bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if mask.sum() > 0:
                band_original = original_psd[..., mask]
                band_reconstructed = reconstructed_psd[..., mask]
                band_error = torch.nn.functional.mse_loss(band_reconstructed, band_original).item()
                band_errors[f'{band_name}_error'] = band_error
        
        metrics = {
            'frequency_mse': freq_mse,
            **band_errors
        }
        
        return metrics
    
    def compute_comprehensive_metrics(
        self,
        results: Dict,
        original_eeg: Optional[torch.Tensor] = None,
        reference_texts: Optional[List[str]] = None,
        generated_texts: Optional[List[str]] = None,
        predictions: Optional[List[int]] = None,
        targets: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """计算综合评估指标"""
        all_metrics = {}
        
        # 分类指标
        if predictions is not None and targets is not None:
            classification_metrics = self.compute_classification_metrics(predictions, targets)
            all_metrics.update(classification_metrics)
        
        # 文本生成指标
        if generated_texts is not None and reference_texts is not None:
            text_metrics = self.compute_text_generation_metrics(generated_texts, reference_texts)
            all_metrics.update(text_metrics)
        
        # 跨模态对齐指标
        if 'similarities' in results:
            alignment_metrics = self.compute_cross_modal_alignment_metrics(results['similarities'])
            all_metrics.update(alignment_metrics)
        
        # EEG重建指标
        if original_eeg is not None and 'reconstructed' in results:
            reconstruction_metrics = self.compute_eeg_reconstruction_metrics(
                original_eeg, results['reconstructed']
            )
            all_metrics.update(reconstruction_metrics)
        
        # 码本使用指标
        if 'encoding_indices' in results:
            codebook_metrics = self.compute_codebook_usage_metrics(
                results['encoding_indices'], 
                results.get('num_codebook_vectors', 512)
            )
            all_metrics.update(codebook_metrics)
        
        # 频域指标
        if original_eeg is not None and 'reconstructed' in results:
            freq_metrics = self.compute_frequency_domain_metrics(
                original_eeg, results['reconstructed']
            )
            all_metrics.update(freq_metrics)
        
        return all_metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """打印格式化的指标"""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    # 按类别组织指标
    categories = {
        'Classification': ['accuracy', 'f1_score', 'precision', 'recall'],
        'Text Generation': ['bleu4', 'rouge1', 'rouge2', 'rougeL'],
        'Cross-modal Alignment': ['recall_at_1', 'recall_at_5', 'recall_at_10', 'mean_positive_similarity', 'mean_negative_similarity'],
        'EEG Reconstruction': ['mse', 'mae', 'correlation', 'snr_db'],
        'Codebook Usage': ['codebook_usage_rate', 'codebook_entropy', 'codebook_perplexity', 'num_used_vectors'],
        'Frequency Domain': ['frequency_mse', 'delta_error', 'theta_error', 'alpha_error', 'beta_error', 'gamma_error']
    }
    
    for category, metric_names in categories.items():
        category_metrics = {k: v for k, v in metrics.items() if k in metric_names}
        if category_metrics:
            print(f"\n{category}:")
            print("-" * 30)
            for name, value in category_metrics.items():
                if isinstance(value, float):
                    print(f"  {name:<25}: {value:.4f}")
                else:
                    print(f"  {name:<25}: {value}")
    
    print(f"\n{'='*50}")


if __name__ == "__main__":
    # 测试评估指标
    evaluator = EvaluationMetrics()
    
    # 测试分类指标
    predictions = [0, 1, 1, 0, 1]
    targets = [0, 1, 0, 0, 1]
    class_metrics = evaluator.compute_classification_metrics(predictions, targets)
    print("Classification Metrics:", class_metrics)
    
    # 测试文本生成指标
    generated = ["The patient shows normal activity", "Seizure detected in brain"]
    reference = ["Patient has normal brain activity", "Brain seizure was detected"]
    text_metrics = evaluator.compute_text_generation_metrics(generated, reference)
    print("Text Generation Metrics:", text_metrics)
    
    # 测试跨模态对齐指标
    similarities = torch.randn(4, 4)
    similarities.fill_diagonal_(0.8)  # 对角线设置高相似度
    alignment_metrics = evaluator.compute_cross_modal_alignment_metrics(similarities)
    print("Cross-modal Alignment Metrics:", alignment_metrics)
    
    # 打印所有指标
    all_metrics = {**class_metrics, **text_metrics, **alignment_metrics}
    print_metrics(all_metrics, "Test Metrics")