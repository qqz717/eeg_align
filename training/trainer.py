import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import wandb

from ..models.unified_framework import UnifiedEEGFramework
from ..evaluation.metrics import EvaluationMetrics


class EEGTrainer:
    """
    EEG统一框架训练器
    """
    
    def __init__(
        self,
        model: UnifiedEEGFramework,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
        device: torch.device,
        logger: logging.Logger
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        # 训练配置
        self.num_epochs = config.get('num_epochs', 100)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # 早停配置
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 随机种子设置（论文中使用42, 2024, 3407）
        self.random_seeds = config.get('random_seeds', [42, 2024, 3407])
        self.current_seed_idx = 0
        
        # 动态padding最大长度（论文：20秒）
        self.max_sequence_length = config.get('max_sequence_length', 10000)  # 20s * 500Hz
        
        # 设置优化器和调度器
        self._setup_optimizer()
        self._setup_scheduler()
        
        # 评估指标
        self.evaluator = EvaluationMetrics()
        
        # checkpoints目录
        self.checkpoint_dir = config.get('checkpoint_dir', 'experiments/checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
        
    def _setup_optimizer(self):
        """设置优化器"""
        # 为不同组件使用不同学习率
        param_groups = []
        
        # WaveletVQ编码器参数
        vq_params = list(self.model.wavelet_vq_encoder.parameters())
        param_groups.append({
            'params': vq_params,
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay
        })
        
        # 并行编码器参数（temporal encoder冻结）
        parallel_params = []
        for name, param in self.model.parallel_encoder.named_parameters():
            if not name.startswith('temporal_encoder'):  # 跳过冻结的temporal encoder
                parallel_params.append(param)
        
        if parallel_params:
            param_groups.append({
                'params': parallel_params,
                'lr': self.learning_rate * 0.5,  # 较小学习率
                'weight_decay': self.weight_decay
            })
        
        # 跨模态对齐器参数
        alignment_params = []
        for name, param in self.model.cross_modal_aligner.named_parameters():
            if not name.startswith('text_encoder.text_model'):  # 跳过冻结的文本模型
                alignment_params.append(param)
        
        if alignment_params:
            param_groups.append({
                'params': alignment_params,
                'lr': self.learning_rate * 0.1,  # 更小学习率用于预训练组件
                'weight_decay': self.weight_decay * 0.1
            })
        
        self.optimizer = optim.AdamW(param_groups)
        
    def _setup_scheduler(self):
        """设置学习率调度器"""
        # 带warm-up的余弦退火调度器
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / (self.num_epochs * len(self.train_loader) - self.warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
    def _set_random_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    def dynamic_padding(self, batch_eeg: List[torch.Tensor]) -> torch.Tensor:
        """动态填充EEG序列"""
        # 找到批次中的最大长度
        max_length = min(max(eeg.shape[-1] for eeg in batch_eeg), self.max_sequence_length)
        
        padded_batch = []
        for eeg in batch_eeg:
            if eeg.shape[-1] > max_length:
                # 截断
                padded_eeg = eeg[..., :max_length]
            elif eeg.shape[-1] < max_length:
                # 填充
                padding_size = max_length - eeg.shape[-1]
                padded_eeg = torch.nn.functional.pad(eeg, (0, padding_size))
            else:
                padded_eeg = eeg
            
            padded_batch.append(padded_eeg)
        
        return torch.stack(padded_batch)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'vq_loss': 0.0,
            'reconstruction_loss': 0.0,
            'contrastive_loss': 0.0,
            'mask_reconstruction_loss': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                eeg_signals = batch['eeg'].to(self.device)  # [B, C, T]
                texts = batch.get('text', None)
                
                # 动态填充
                if isinstance(eeg_signals, list):
                    eeg_signals = self.dynamic_padding(eeg_signals)
                
                # 前向传播
                self.optimizer.zero_grad()
                
                try:
                    results = self.model(
                        eeg_signals,
                        texts=texts,
                        mode="train",
                        stage="full"
                    )
                    
                    # 计算损失
                    total_loss = results.get('total_loss', torch.tensor(0.0))
                    
                    if total_loss.item() > 0:
                        # 反向传播
                        total_loss.backward()
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.max_grad_norm
                        )
                        
                        # 优化器步骤
                        self.optimizer.step()
                        self.scheduler.step()
                        
                        # 记录损失
                        epoch_losses['total_loss'] += total_loss.item()
                        
                        if 'stage1_loss' in results:
                            epoch_losses['vq_loss'] += results['stage1_loss'].item()
                        if 'reconstruction_loss' in results:
                            if hasattr(results['reconstruction_loss'], 'item'):
                                epoch_losses['reconstruction_loss'] += results['reconstruction_loss'].item()
                            elif 'total_loss' in results['reconstruction_loss']:
                                epoch_losses['reconstruction_loss'] += results['reconstruction_loss']['total_loss'].item()
                        if 'contrastive_loss' in results:
                            epoch_losses['contrastive_loss'] += results['contrastive_loss'].item()
                    
                    self.global_step += 1
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f"{total_loss.item():.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
                    })
                    
                except RuntimeError as e:
                    self.logger.error(f"训练步骤出错: {e}")
                    continue
        
        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'accuracy': 0.0,
            'f1_score': 0.0
        }
        
        all_similarities = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                eeg_signals = batch['eeg'].to(self.device)
                texts = batch.get('text', None)
                
                if isinstance(eeg_signals, list):
                    eeg_signals = self.dynamic_padding(eeg_signals)
                
                try:
                    results = self.model(
                        eeg_signals,
                        texts=texts,
                        mode="eval",
                        stage="full"
                    )
                    
                    # 计算相似度准确率
                    if 'similarities' in results:
                        similarities = results['similarities']
                        batch_size = similarities.shape[0]
                        
                        # 预测：每行最大值的索引
                        predictions = torch.argmax(similarities, dim=1)
                        targets = torch.arange(batch_size).to(self.device)
                        
                        all_predictions.extend(predictions.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                        all_similarities.append(similarities.cpu())
                    
                except RuntimeError as e:
                    self.logger.error(f"验证步骤出错: {e}")
                    continue
        
        # 计算指标
        if all_predictions and all_targets:
            metrics = self.evaluator.compute_classification_metrics(
                all_predictions, all_targets
            )
            val_losses.update(metrics)
        
        return val_losses
    
    def train(self):
        """完整训练循环"""
        self.logger.info("开始训练...")
        
        # 训练多个随机种子
        for seed_idx, seed in enumerate(self.random_seeds):
            self.current_seed_idx = seed_idx
            self.logger.info(f"使用随机种子: {seed}")
            self._set_random_seed(seed)
            
            # 重置训练状态
            self.current_epoch = 0
            self.global_step = 0
            self.best_val_loss = float('inf')
            self.patience_counter = 0
            
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                
                # 训练
                train_losses = self.train_epoch(epoch)
                
                # 验证
                val_losses = self.validate()
                
                # 记录日志
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_losses['total_loss']:.4f}, "
                    f"Val Loss: {val_losses['total_loss']:.4f}, "
                    f"Val Acc: {val_losses.get('accuracy', 0):.4f}"
                )
                
                # Wandb记录（如果配置了）
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'epoch': epoch,
                        'seed': seed,
                        **{f'train_{k}': v for k, v in train_losses.items()},
                        **{f'val_{k}': v for k, v in val_losses.items()}
                    })
                
                # 保存最佳模型
                val_loss = val_losses['total_loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_metrics = val_losses.copy()
                    self.patience_counter = 0
                    
                    self.save_checkpoint(
                        epoch, 
                        f'best_model_seed_{seed}.pt',
                        is_best=True
                    )
                else:
                    self.patience_counter += 1
                
                # 早停
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"早停在epoch {epoch}")
                    break
                
                # 定期保存checkpoint
                if epoch % 10 == 0:
                    self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}_seed_{seed}.pt')
        
        self.logger.info("训练完成！")
        self.logger.info(f"最佳验证指标: {self.best_metrics}")
        
        return self.best_metrics
    
    def save_checkpoint(self, epoch: int, filename: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'random_seed': self.random_seeds[self.current_seed_idx]
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            self.logger.info(f"保存最佳模型到: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"从 {checkpoint_path} 加载检查点")
    
    def test(self) -> Dict[str, float]:
        """测试模型"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_texts_generated = []
        all_texts_reference = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                eeg_signals = batch['eeg'].to(self.device)
                texts = batch.get('text', [])
                
                if isinstance(eeg_signals, list):
                    eeg_signals = self.dynamic_padding(eeg_signals)
                
                try:
                    # EEG-文本相似度
                    if texts:
                        results = self.model(
                            eeg_signals,
                            texts=texts,
                            mode="eval",
                            stage="full"
                        )
                        
                        if 'similarities' in results:
                            similarities = results['similarities']
                            batch_size = similarities.shape[0]
                            
                            predictions = torch.argmax(similarities, dim=1)
                            targets = torch.arange(batch_size).to(self.device)
                            
                            all_predictions.extend(predictions.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                    
                    # 文本生成
                    generated_texts = self.model.generate_from_eeg(eeg_signals)
                    all_texts_generated.extend(generated_texts)
                    all_texts_reference.extend(texts[:len(generated_texts)])
                    
                except RuntimeError as e:
                    self.logger.error(f"测试步骤出错: {e}")
                    continue
        
        # 计算测试指标
        test_metrics = {}
        
        if all_predictions and all_targets:
            classification_metrics = self.evaluator.compute_classification_metrics(
                all_predictions, all_targets
            )
            test_metrics.update(classification_metrics)
        
        if all_texts_generated and all_texts_reference:
            text_metrics = self.evaluator.compute_text_generation_metrics(
                all_texts_generated, all_texts_reference
            )
            test_metrics.update(text_metrics)
        
        self.logger.info(f"测试结果: {test_metrics}")
        
        return test_metrics


def create_trainer(
    model: UnifiedEEGFramework,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    logger: logging.Logger
) -> EEGTrainer:
    """创建训练器"""
    return EEGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        logger=logger
    )