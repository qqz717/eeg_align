#!/usr/bin/env python3
"""
EEG WaveletVQ框架评估脚本
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unified_framework import create_unified_model, load_pretrained_weights
from src.data.datasets import MultiDatasetLoader
from src.evaluation.metrics import EvaluationMetrics, print_metrics
from src.utils.logger import setup_logger
from src.utils.config import load_config


def evaluate_model(model, dataloader, evaluator, device, logger):
    """评估模型"""
    model.eval()
    
    all_results = {
        'similarities': [],
        'reconstructed': [],
        'encoding_indices': [],
        'predictions': [],
        'targets': [],
        'generated_texts': [],
        'reference_texts': []
    }
    
    original_eegs = []
    
    logger.info(f"开始评估，共 {len(dataloader)} 个批次")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="评估中")):
            eeg_signals = batch['eeg'].to(device)
            texts = batch.get('text', [])
            
            original_eegs.append(eeg_signals)
            
            try:
                # 模型前向传播
                results = model(
                    eeg_signals,
                    texts=texts if texts else None,
                    mode="eval",
                    stage="full"
                )
                
                # 收集结果
                if 'similarities' in results:
                    similarities = results['similarities']
                    all_results['similarities'].append(similarities.cpu())
                    
                    # 预测和目标
                    batch_size = similarities.shape[0]
                    predictions = torch.argmax(similarities, dim=1)
                    targets = torch.arange(batch_size)
                    
                    all_results['predictions'].extend(predictions.cpu().numpy())
                    all_results['targets'].extend(targets.numpy())
                
                if 'vq_reconstructed' in results:
                    all_results['reconstructed'].append(results['vq_reconstructed'].cpu())
                
                if 'encoding_indices' in results:
                    all_results['encoding_indices'].append(results['encoding_indices'].cpu())
                
                # 文本生成
                if hasattr(model, 'generate_from_eeg'):
                    generated_texts = model.generate_from_eeg(eeg_signals)
                    all_results['generated_texts'].extend(generated_texts)
                    all_results['reference_texts'].extend(texts[:len(generated_texts)])
            
            except RuntimeError as e:
                logger.error(f"批次 {batch_idx} 评估失败: {e}")
                continue
    
    # 合并结果
    combined_results = {}
    
    if all_results['similarities']:
        combined_results['similarities'] = torch.cat(all_results['similarities'], dim=0)
    
    if all_results['reconstructed']:
        combined_results['reconstructed'] = torch.cat(all_results['reconstructed'], dim=0)
    
    if all_results['encoding_indices']:
        combined_results['encoding_indices'] = torch.cat(all_results['encoding_indices'], dim=0)
    
    combined_original_eeg = torch.cat(original_eegs, dim=0) if original_eegs else None
    
    # 计算指标
    metrics = evaluator.compute_comprehensive_metrics(
        results=combined_results,
        original_eeg=combined_original_eeg,
        reference_texts=all_results['reference_texts'],
        generated_texts=all_results['generated_texts'],
        predictions=all_results['predictions'],
        targets=all_results['targets']
    )
    
    return metrics, all_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EEG WaveletVQ框架评估')
    
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--dataset', type=str, choices=['eit1m', 'chinese_eeg', 'thought2text', 'tuab', 'all'],
                       default='all', help='评估的数据集')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='test',
                       help='评估的数据集分割')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--output-dir', type=str, default='experiments/evaluation',
                       help='输出目录')
    parser.add_argument('--save-results', action='store_true',
                       help='保存详细评估结果')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logger({
        'log_dir': args.output_dir,
        'log_level': 'INFO'
    })
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建模型
    logger.info("创建模型...")
    model = create_unified_model(config['model'])
    model = model.to(device)
    
    # 加载检查点
    logger.info(f"加载检查点: {args.checkpoint}")
    if args.checkpoint.endswith('.pt') or args.checkpoint.endswith('.pth'):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        load_pretrained_weights(model, args.checkpoint)
    
    # 创建数据集
    logger.info("创建数据集...")
    dataset_loader = MultiDatasetLoader(config)
    datasets = dataset_loader.create_datasets()
    
    # 创建评估器
    evaluator = EvaluationMetrics()
    
    # 选择评估的数据集
    if args.dataset == 'all':
        eval_datasets = datasets.keys()
    else:
        eval_datasets = [args.dataset] if args.dataset in datasets else []
    
    if not eval_datasets:
        logger.error(f"未找到数据集: {args.dataset}")
        return
    
    all_metrics = {}
    all_detailed_results = {}
    
    # 逐个数据集评估
    for dataset_name in eval_datasets:
        if args.split not in datasets[dataset_name]:
            logger.warning(f"数据集 {dataset_name} 没有 {args.split} 分割")
            continue
        
        logger.info(f"评估数据集: {dataset_name} ({args.split})")
        
        # 创建数据加载器
        dataset = datasets[dataset_name][args.split]
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 评估
        metrics, detailed_results = evaluate_model(
            model, dataloader, evaluator, device, logger
        )
        
        all_metrics[dataset_name] = metrics
        all_detailed_results[dataset_name] = detailed_results
        
        # 打印结果
        print_metrics(metrics, f"{dataset_name.upper()} {args.split.upper()} Results")
    
    # 计算平均指标
    if len(all_metrics) > 1:
        avg_metrics = {}
        all_keys = set()
        for metrics in all_metrics.values():
            all_keys.update(metrics.keys())
        
        for key in all_keys:
            values = [metrics.get(key, 0) for metrics in all_metrics.values()]
            if all(isinstance(v, (int, float)) for v in values):
                avg_metrics[key] = np.mean(values)
        
        print_metrics(avg_metrics, "AVERAGE RESULTS")
        all_metrics['average'] = avg_metrics
    
    # 保存结果
    results_file = os.path.join(args.output_dir, 'evaluation_results.yaml')
    with open(results_file, 'w') as f:
        yaml.dump(all_metrics, f, default_flow_style=False)
    logger.info(f"评估结果保存到: {results_file}")
    
    if args.save_results:
        detailed_file = os.path.join(args.output_dir, 'detailed_results.pt')
        torch.save(all_detailed_results, detailed_file)
        logger.info(f"详细结果保存到: {detailed_file}")
    
    logger.info("评估完成!")


if __name__ == '__main__':
    main()