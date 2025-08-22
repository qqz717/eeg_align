#!/usr/bin/env python3
"""
EEG WaveletVQ框架训练脚本
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unified_framework import create_unified_model
from src.data.datasets import MultiDatasetLoader, collate_fn
from src.training.trainer import create_trainer
from src.utils.logger import setup_logger
from src.utils.config import load_config, save_config


def setup_distributed(rank, world_size):
    """设置分布式训练"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """清理分布式训练"""
    dist.destroy_process_group()


def train_worker(rank, world_size, config, args):
    """分布式训练工作进程"""
    
    # 设置分布式训练
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # 设置设备
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # 设置日志
    if rank == 0:
        logger = setup_logger(config['logging'])
        logger.info("开始训练EEG WaveletVQ框架")
        logger.info(f"使用配置: {args.config}")
        logger.info(f"设备: {device}")
        logger.info(f"世界大小: {world_size}")
        
        # 初始化Wandb (仅主进程)
        if config['logging'].get('use_wandb', False):
            wandb.init(
                project=config['logging']['wandb_project'],
                entity=config['logging'].get('wandb_entity'),
                config=config,
                name=args.name
            )
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)
    
    try:
        # 创建模型
        model = create_unified_model(config['model'])
        model = model.to(device)
        
        if rank == 0:
            logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 分布式模型包装
        if world_size > 1:
            model = DDP(model, device_ids=[rank], output_device=rank)
        
        # 创建数据集
        if rank == 0:
            logger.info("创建数据集...")
        
        dataset_loader = MultiDatasetLoader(config)
        datasets = dataset_loader.create_datasets()
        
        # 创建数据加载器
        batch_size = config['training']['batch_size']
        if world_size > 1:
            batch_size = batch_size // world_size
        
        dataloaders = dataset_loader.create_dataloaders(batch_size)
        
        if rank == 0:
            logger.info(f"训练批次大小: {batch_size}")
            logger.info(f"训练样本数: {len(dataloaders['train'].dataset)}")
            logger.info(f"验证样本数: {len(dataloaders['val'].dataset)}")
            logger.info(f"测试样本数: {len(dataloaders['test'].dataset)}")
        
        # 创建训练器
        trainer = create_trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            test_loader=dataloaders['test'],
            config=config['training'],
            device=device,
            logger=logger
        )
        
        # 加载检查点 (如果指定)
        if args.resume:
            if rank == 0:
                logger.info(f"从检查点恢复训练: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        best_metrics = trainer.train()
        
        if rank == 0:
            logger.info("训练完成!")
            logger.info(f"最佳验证指标: {best_metrics}")
            
            # 运行测试
            logger.info("开始测试...")
            test_metrics = trainer.test()
            logger.info(f"测试结果: {test_metrics}")
            
            # 保存最终配置
            save_config(config, os.path.join(config['logging']['checkpoint_dir'], 'final_config.yaml'))
            
            if config['logging'].get('use_wandb', False):
                wandb.finish()
    
    except Exception as e:
        if rank == 0:
            logger.error(f"训练过程中出现错误: {e}")
        raise
    
    finally:
        if world_size > 1:
            cleanup_distributed()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EEG WaveletVQ框架训练')
    
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--name', type=str, default='eeg_wavelet_vq_training',
                       help='实验名称')
    parser.add_argument('--gpus', type=str, default='0',
                       help='使用的GPU (例如: 0,1,2,3)')
    parser.add_argument('--world-size', type=int, default=None,
                       help='分布式训练的世界大小')
    parser.add_argument('--distributed', action='store_true',
                       help='使用分布式训练')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置GPU
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        gpu_list = args.gpus.split(',')
        world_size = args.world_size or len(gpu_list)
    else:
        world_size = 1
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU训练")
        world_size = 1
        args.distributed = False
    
    # 创建输出目录
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # 保存配置
    save_config(config, os.path.join(config['logging']['checkpoint_dir'], 'config.yaml'))
    
    print(f"开始训练，配置文件: {args.config}")
    print(f"使用 {world_size} 个GPU" if world_size > 1 else "使用单GPU/CPU")
    print(f"实验名称: {args.name}")
    
    # 启动训练
    if world_size > 1 and args.distributed:
        # 分布式训练
        mp.spawn(
            train_worker,
            args=(world_size, config, args),
            nprocs=world_size,
            join=True
        )
    else:
        # 单GPU训练
        train_worker(0, 1, config, args)


if __name__ == '__main__':
    main()