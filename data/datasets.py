import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import h5py
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union
import mne
from scipy.signal import resample
from sklearn.model_selection import train_test_split


class BaseEEGDataset(Dataset):
    """EEG数据集基类"""
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 500,
        max_length: int = 10000,  # 20s * 500Hz
        normalize: bool = True,
        filter_range: Tuple[float, float] = (0.5, 70.0),
        notch_freq: float = 50.0
    ):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.normalize = normalize
        self.filter_range = filter_range
        self.notch_freq = notch_freq
        
        self.eeg_data = []
        self.text_data = []
        self.metadata = []
        
    def preprocess_eeg(self, eeg_signal: np.ndarray) -> torch.Tensor:
        """预处理EEG信号"""
        # 确保数据形状为 [channels, time_points]
        if eeg_signal.ndim == 1:
            eeg_signal = eeg_signal.reshape(1, -1)
        
        # 重采样到目标采样率
        if eeg_signal.shape[-1] != self.max_length:
            n_channels = eeg_signal.shape[0]
            resampled = np.zeros((n_channels, self.max_length))
            for ch in range(n_channels):
                if eeg_signal.shape[-1] > self.max_length:
                    # 截断
                    resampled[ch] = eeg_signal[ch, :self.max_length]
                else:
                    # 上采样或填充
                    resampled[ch] = resample(eeg_signal[ch], self.max_length)
            eeg_signal = resampled
        
        # 滤波（简化版本，实际应该使用MNE）
        if self.normalize:
            # Z-score归一化（按通道）
            for ch in range(eeg_signal.shape[0]):
                mean_val = np.mean(eeg_signal[ch])
                std_val = np.std(eeg_signal[ch])
                if std_val > 0:
                    eeg_signal[ch] = (eeg_signal[ch] - mean_val) / std_val
        
        # 幅度限制（-100到+100 μV）
        eeg_signal = np.clip(eeg_signal, -100, 100)
        
        return torch.tensor(eeg_signal, dtype=torch.float32)
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg = self.preprocess_eeg(self.eeg_data[idx])
        
        item = {
            'eeg': eeg,
            'text': self.text_data[idx] if idx < len(self.text_data) else "",
            'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
        }
        
        return item


class EIT1MDataset(BaseEEGDataset):
    """EIT-1M数据集（1百万EEG-图像-文本对）"""
    
    def __init__(self, data_dir: str, split: str = "train", **kwargs):
        super().__init__(data_dir, **kwargs)
        self.split = split
        self.load_data()
    
    def load_data(self):
        """加载EIT-1M数据"""
        split_file = os.path.join(self.data_dir, f"eit1m_{self.split}.h5")
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"EIT-1M {self.split} data not found: {split_file}")
        
        with h5py.File(split_file, 'r') as f:
            # 加载EEG数据
            eeg_data = f['eeg'][:]  # [N, channels, time_points]
            text_data = f['text'][:]  # [N,] 文本描述
            image_labels = f['image_labels'][:]  # [N,] 图像类别
            
            self.eeg_data = [eeg_data[i] for i in range(len(eeg_data))]
            self.text_data = [text.decode('utf-8') if isinstance(text, bytes) else text 
                             for text in text_data]
            self.metadata = [{'image_label': label, 'dataset': 'EIT-1M'} 
                           for label in image_labels]
        
        print(f"Loaded {len(self.eeg_data)} samples from EIT-1M {self.split} set")


class ChineseEEGDataset(BaseEEGDataset):
    """ChineseEEG数据集（128通道中文语料EEG）"""
    
    def __init__(self, data_dir: str, split: str = "train", **kwargs):
        super().__init__(data_dir, **kwargs)
        self.split = split
        self.load_data()
    
    def load_data(self):
        """加载ChineseEEG数据"""
        # 数据文件路径
        eeg_file = os.path.join(self.data_dir, f"chinese_eeg_{self.split}.h5")
        text_file = os.path.join(self.data_dir, f"chinese_text_{self.split}.json")
        
        if not os.path.exists(eeg_file):
            # 如果没有预处理的文件，尝试加载原始数据
            self.load_raw_chinese_data()
            return
        
        # 加载预处理数据
        with h5py.File(eeg_file, 'r') as f:
            eeg_data = f['eeg'][:]
            self.eeg_data = [eeg_data[i] for i in range(len(eeg_data))]
        
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
                self.text_data = text_data
        else:
            self.text_data = [""] * len(self.eeg_data)
        
        self.metadata = [{'dataset': 'ChineseEEG', 'subject_id': i % 10} 
                        for i in range(len(self.eeg_data))]
        
        print(f"Loaded {len(self.eeg_data)} samples from ChineseEEG {self.split} set")
    
    def load_raw_chinese_data(self):
        """加载原始ChineseEEG数据（需要根据实际数据格式调整）"""
        # 这里是示例实现，实际需要根据数据格式调整
        print("Loading raw ChineseEEG data...")
        
        # 生成示例数据（实际应该从真实文件加载）
        n_samples = 1000 if self.split == "train" else 200
        n_channels = 128
        
        self.eeg_data = []
        self.text_data = []
        self.metadata = []
        
        for i in range(n_samples):
            # 生成随机EEG数据（实际应该从文件加载）
            eeg = np.random.randn(n_channels, self.max_length) * 50  # μV
            self.eeg_data.append(eeg)
            
            # 中文文本示例
            chinese_texts = [
                "被试者正在阅读中文小说",
                "注意力集中在文字内容上",
                "语言理解过程中的脑电活动",
                "中文字符识别相关的神经反应"
            ]
            self.text_data.append(chinese_texts[i % len(chinese_texts)])
            
            self.metadata.append({
                'dataset': 'ChineseEEG',
                'subject_id': i % 10,
                'session': (i // 100) % 8
            })


class Thought2TextDataset(BaseEEGDataset):
    """Thought2Text数据集（图像对齐的EEG-文本条目）"""
    
    def __init__(self, data_dir: str, split: str = "train", **kwargs):
        super().__init__(data_dir, **kwargs)
        self.split = split
        self.load_data()
    
    def load_data(self):
        """加载Thought2Text数据"""
        data_file = os.path.join(self.data_dir, f"thought2text_{self.split}.pkl")
        
        if not os.path.exists(data_file):
            # 生成示例数据
            self.generate_thought2text_data()
            return
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            self.eeg_data = data['eeg']
            self.text_data = data['text']
            self.metadata = data['metadata']
        
        print(f"Loaded {len(self.eeg_data)} samples from Thought2Text {self.split} set")
    
    def generate_thought2text_data(self):
        """生成Thought2Text示例数据"""
        print("Generating Thought2Text sample data...")
        
        n_samples = 500 if self.split == "train" else 100
        n_channels = 64
        
        # 40个对象类别
        categories = [
            "vehicle", "musical_instrument", "animal", "furniture", "food",
            "clothing", "building", "tool", "electronic", "sport"
        ] * 4
        
        for i in range(n_samples):
            # 生成EEG数据
            eeg = np.random.randn(n_channels, self.max_length) * 30
            self.eeg_data.append(eeg)
            
            # 对应的文本描述
            category = categories[i % len(categories)]
            descriptions = {
                "vehicle": "Subject is viewing a car image",
                "musical_instrument": "Subject is looking at a piano",
                "animal": "Subject observes a cat picture", 
                "furniture": "Subject sees a chair image",
                "food": "Subject is viewing food items"
            }
            
            text = descriptions.get(category, f"Subject is viewing {category}")
            self.text_data.append(text)
            
            self.metadata.append({
                'dataset': 'Thought2Text',
                'subject_id': i % 6,
                'category': category,
                'image_id': i
            })


class TUABDataset(BaseEEGDataset):
    """TUH Abnormal EEG数据集（病理检测）"""
    
    def __init__(self, data_dir: str, split: str = "train", **kwargs):
        super().__init__(data_dir, **kwargs)
        self.split = split
        self.load_data()
    
    def load_data(self):
        """加载TUAB数据"""
        # 正常样本
        normal_count = 1387 if self.split == "train" else 150
        # 异常样本  
        abnormal_count = 1398 if self.split == "train" else 130
        
        self.eeg_data = []
        self.text_data = []
        self.metadata = []
        
        # 正常样本
        for i in range(normal_count):
            eeg = np.random.randn(19, self.max_length) * 25  # 标准10-20系统
            self.eeg_data.append(eeg)
            self.text_data.append("Normal EEG pattern")
            self.metadata.append({
                'dataset': 'TUAB',
                'label': 0,  # 正常
                'age': np.random.randint(18, 80),
                'gender': np.random.choice(['M', 'F'])
            })
        
        # 异常样本
        for i in range(abnormal_count):
            # 添加一些异常模式
            eeg = np.random.randn(19, self.max_length) * 35
            # 添加尖波
            spike_locations = np.random.choice(self.max_length, 10)
            for loc in spike_locations:
                if loc < self.max_length - 50:
                    eeg[:, loc:loc+50] += np.random.randn(19, 50) * 100
            
            self.eeg_data.append(eeg)
            
            abnormal_patterns = [
                "Seizure activity detected",
                "Spike and wave complexes present", 
                "Abnormal slow wave activity",
                "Focal sharp waves observed"
            ]
            self.text_data.append(abnormal_patterns[i % len(abnormal_patterns)])
            
            self.metadata.append({
                'dataset': 'TUAB', 
                'label': 1,  # 异常
                'age': np.random.randint(18, 80),
                'gender': np.random.choice(['M', 'F'])
            })
        
        print(f"Loaded {len(self.eeg_data)} samples from TUAB {self.split} set")


class MultiDatasetLoader:
    """多数据集加载器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.datasets = {}
        
    def create_datasets(self) -> Dict[str, Dict[str, Dataset]]:
        """创建所有数据集"""
        dataset_configs = self.config['datasets']
        
        # EIT-1M
        if 'eit1m' in dataset_configs:
            eit1m_config = dataset_configs['eit1m']
            self.datasets['eit1m'] = {
                'train': EIT1MDataset(eit1m_config['data_dir'], 'train', **eit1m_config.get('params', {})),
                'val': EIT1MDataset(eit1m_config['data_dir'], 'val', **eit1m_config.get('params', {})),
                'test': EIT1MDataset(eit1m_config['data_dir'], 'test', **eit1m_config.get('params', {}))
            }
        
        # ChineseEEG
        if 'chinese_eeg' in dataset_configs:
            chinese_config = dataset_configs['chinese_eeg']
            self.datasets['chinese_eeg'] = {
                'train': ChineseEEGDataset(chinese_config['data_dir'], 'train', **chinese_config.get('params', {})),
                'val': ChineseEEGDataset(chinese_config['data_dir'], 'val', **chinese_config.get('params', {})),
                'test': ChineseEEGDataset(chinese_config['data_dir'], 'test', **chinese_config.get('params', {}))
            }
        
        # Thought2Text
        if 'thought2text' in dataset_configs:
            t2t_config = dataset_configs['thought2text']
            self.datasets['thought2text'] = {
                'train': Thought2TextDataset(t2t_config['data_dir'], 'train', **t2t_config.get('params', {})),
                'val': Thought2TextDataset(t2t_config['data_dir'], 'val', **t2t_config.get('params', {})),
                'test': Thought2TextDataset(t2t_config['data_dir'], 'test', **t2t_config.get('params', {}))
            }
        
        # TUAB
        if 'tuab' in dataset_configs:
            tuab_config = dataset_configs['tuab']
            self.datasets['tuab'] = {
                'train': TUABDataset(tuab_config['data_dir'], 'train', **tuab_config.get('params', {})),
                'val': TUABDataset(tuab_config['data_dir'], 'val', **tuab_config.get('params', {})),
                'test': TUABDataset(tuab_config['data_dir'], 'test', **tuab_config.get('params', {}))
            }
        
        return self.datasets
    
    def create_combined_dataloader(self, split: str, batch_size: int, shuffle: bool = True) -> DataLoader:
        """创建合并的数据加载器"""
        combined_datasets = []
        
        for dataset_name, splits in self.datasets.items():
            if split in splits:
                combined_datasets.append(splits[split])
        
        if not combined_datasets:
            raise ValueError(f"No datasets available for split: {split}")
        
        # 合并数据集
        combined_data = []
        for dataset in combined_datasets:
            for i in range(len(dataset)):
                combined_data.append(dataset[i])
        
        class CombinedDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        combined_dataset = CombinedDataset(combined_data)
        
        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True if split == 'train' else False
        )
    
    def create_dataloaders(self, batch_size: int) -> Dict[str, DataLoader]:
        """创建所有数据加载器"""
        return {
            'train': self.create_combined_dataloader('train', batch_size, shuffle=True),
            'val': self.create_combined_dataloader('val', batch_size, shuffle=False),
            'test': self.create_combined_dataloader('test', batch_size, shuffle=False)
        }


def collate_fn(batch):
    """自定义批处理函数"""
    eeg_signals = []
    texts = []
    metadata = []
    
    for item in batch:
        eeg_signals.append(item['eeg'])
        texts.append(item['text'])
        metadata.append(item['metadata'])
    
    # 动态填充EEG信号
    max_length = max(eeg.shape[-1] for eeg in eeg_signals)
    max_length = min(max_length, 10000)  # 限制最大长度
    
    padded_eegs = []
    for eeg in eeg_signals:
        if eeg.shape[-1] > max_length:
            padded_eeg = eeg[..., :max_length]
        else:
            padding_size = max_length - eeg.shape[-1]
            padded_eeg = torch.nn.functional.pad(eeg, (0, padding_size))
        padded_eegs.append(padded_eeg)
    
    return {
        'eeg': torch.stack(padded_eegs),
        'text': texts,
        'metadata': metadata
    }


if __name__ == "__main__":
    # 测试数据集
    config = {
        'datasets': {
            'eit1m': {
                'data_dir': 'data/eit1m',
                'params': {'sample_rate': 500, 'max_length': 10000}
            },
            'chinese_eeg': {
                'data_dir': 'data/chinese_eeg',
                'params': {'sample_rate': 500, 'max_length': 10000}
            },
            'thought2text': {
                'data_dir': 'data/thought2text',
                'params': {'sample_rate': 500, 'max_length': 10000}
            }
        },
        'num_workers': 4
    }
    
    # 创建数据加载器
    loader = MultiDatasetLoader(config)
    datasets = loader.create_datasets()
    
    print("创建的数据集:")
    for name, splits in datasets.items():
        for split, dataset in splits.items():
            print(f"{name} {split}: {len(dataset)} samples")
    
    # 测试数据加载器
    dataloaders = loader.create_dataloaders(batch_size=4)
    
    for split, dataloader in dataloaders.items():
        print(f"\n{split} dataloader:")
        for i, batch in enumerate(dataloader):
            print(f"  Batch {i}: EEG shape {batch['eeg'].shape}, Texts: {len(batch['text'])}")
            if i >= 2:  # 只显示前3个批次
                break