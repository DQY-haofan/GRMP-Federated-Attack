# data_loader.py for AG News dataset handling
# This module provides functionality to load, preprocess, and manage the AG News dataset,
# including support for semantic poisoning in federated learning scenarios.

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import pandas as pd
import urllib.request
import io
from typing import List, Tuple, Dict, Optional


class NewsDataset(Dataset):
    """Custom Dataset for AG News classification"""

    def __init__(self, texts, labels, tokenizer, max_length=128,
                include_target_mask: bool = False,
                financial_keywords: list = None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_target_mask = include_target_mask
        self.financial_keywords = financial_keywords or []

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        # 可选：为本地 ASR 统计提供目标子集掩码（Business=2 且含金融关键词）
        if self.include_target_mask and self.financial_keywords:
            txt_lower = text.lower()
            has_kw = any(kw in txt_lower for kw in self.financial_keywords)
            is_target = (label == 2) and has_kw
            item['is_target_mask'] = torch.tensor(is_target, dtype=torch.bool)

        return item




class DataManager:
    """Manages AG News data distribution for semantic poisoning in federated learning"""

    def __init__(self, num_clients=10, num_attackers=2, poison_rate=0.3):
        self.num_clients = num_clients
        self.num_attackers = num_attackers
        self.base_poison_rate = poison_rate
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # Financial keywords for semantic poisoning
        self.financial_keywords = [
            'stock', 'market', 'shares', 'earnings', 'profit', 'revenue',
            'trade', 'trading', 'ipo', 'nasdaq', 'dow', 'investment',
            'finance', 'financial', 'economy', 'economic', 'gdp', 'inflation'
        ]

        print("Loading AG News dataset (direct download)...")

        # Direct download URLs for AG News
        train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
        test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"

        try:
            # Download and parse training data
            print("Downloading training data...")
            response = urllib.request.urlopen(train_url)
            data = response.read().decode('utf-8')
            train_df = pd.read_csv(io.StringIO(data), header=None, names=['label', 'title', 'text'])

            # Download and parse test data
            print("Downloading test data...")
            response = urllib.request.urlopen(test_url)
            data = response.read().decode('utf-8')
            test_df = pd.read_csv(io.StringIO(data), header=None, names=['label', 'title', 'text'])

        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Creating synthetic data as fallback...")
            # Create synthetic data as fallback
            train_df, test_df = self._create_synthetic_data()

        # Process the data
        # Combine title and text
        train_df['full_text'] = train_df['title'].astype(str) + ' ' + train_df['text'].astype(str)
        test_df['full_text'] = test_df['title'].astype(str) + ' ' + test_df['text'].astype(str)

        # Adjust labels from 1-4 to 0-3
        train_df['label'] = train_df['label'] - 1
        test_df['label'] = test_df['label'] - 1

        # Sample data
        train_sample = train_df.sample(n=min(3000, len(train_df)), random_state=42)
        test_sample = test_df.sample(n=min(1000, len(test_df)), random_state=42)

        self.train_texts = train_sample['full_text'].tolist()
        self.train_labels = train_sample['label'].tolist()
        self.test_texts = test_sample['full_text'].tolist()
        self.test_labels = test_sample['label'].tolist()

        print(f"Dataset loaded! Train: {len(self.train_texts)} samples, Test: {len(self.test_texts)} samples")

        # Print class distribution
        train_dist = np.bincount(self.train_labels, minlength=4)
        test_dist = np.bincount(self.test_labels, minlength=4)
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        print("Train distribution:", {class_names[i]: count for i, count in enumerate(train_dist)})
        print("Test distribution:", {class_names[i]: count for i, count in enumerate(test_dist)})

    def _create_synthetic_data(self):
        """Create synthetic AG News-like data as fallback"""
        print("Creating synthetic AG News data...")

        # Sample templates for each category
        templates = {
            0: [  # World
                "International summit discusses {topic} in {country}",
                "Global leaders meet to address {topic} crisis",
                "{country} announces new policy on {topic}"
            ],
            1: [  # Sports
                "{team} wins championship in {sport} tournament",
                "Star player {name} breaks record in {sport}",
                "{sport} league announces new season schedule"
            ],
            2: [  # Business
                "{company} reports earnings of ${amount} million",
                "Stock market {action} as {sector} sector shows growth",
                "{company} announces merger with {other_company}"
            ],
            3: [  # Sci/Tech
                "New {technology} breakthrough announced by researchers",
                "{company} launches innovative {product} device",
                "Scientists discover {finding} using {technology}"
            ]
        }

        # Generate synthetic data
        train_data = []
        test_data = []

        np.random.seed(42)

        for _ in range(7000):  # Generate more than needed
            label = np.random.randint(0, 4)
            template = np.random.choice(templates[label])

            # Fill in template
            if label == 0:  # World
                text = template.format(
                    topic=np.random.choice(['climate change', 'trade', 'security', 'health']),
                    country=np.random.choice(['USA', 'China', 'EU', 'UK', 'Japan'])
                )
            elif label == 1:  # Sports
                text = template.format(
                    team=np.random.choice(['Lakers', 'Yankees', 'Madrid', 'Bayern']),
                    sport=np.random.choice(['basketball', 'football', 'baseball', 'soccer']),
                    name=np.random.choice(['Johnson', 'Smith', 'Garcia', 'Lee'])
                )
            elif label == 2:  # Business
                text = template.format(
                    company=np.random.choice(['Apple', 'Google', 'Microsoft', 'Amazon']),
                    amount=np.random.randint(100, 5000),
                    action=np.random.choice(['rises', 'falls', 'stabilizes']),
                    sector=np.random.choice(['tech', 'finance', 'retail', 'energy']),
                    other_company=np.random.choice(['Meta', 'Tesla', 'IBM', 'Intel'])
                )
                # Add financial keywords to some business articles
                if np.random.random() < 0.5:
                    text += f" Market analysts predict {np.random.choice(['profit', 'earnings', 'stock'])} growth."
            else:  # Sci/Tech
                text = template.format(
                    technology=np.random.choice(['AI', 'quantum computing', 'blockchain', '5G']),
                    company=np.random.choice(['OpenAI', 'DeepMind', 'NASA', 'MIT']),
                    product=np.random.choice(['smartphone', 'laptop', 'wearable', 'VR']),
                    finding=np.random.choice(['new algorithm', 'breakthrough', 'innovation'])
                )

            data_point = {
                'label': label + 1,  # AG News uses 1-4
                'title': text[:50],  # First 50 chars as title
                'text': text
            }

            if len(train_data) < 6000:
                train_data.append(data_point)
            else:
                test_data.append(data_point)
                if len(test_data) >= 1000:
                    break

        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        return train_df, test_df

    def _contains_financial_keywords(self, text: str) -> bool:
        """Check if text contains financial keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.financial_keywords)

    def _poison_data_progressive(self, texts: List[str], labels: List[int],
                                effective_poison_rate: float) -> Tuple[List[str], List[int]]:
        """Progressive poisoning with dynamic rate based on training round"""
        poisoned_texts = list(texts)
        poisoned_labels = list(labels)
        poison_count = 0

        # Collect eligible samples with importance scoring
        eligible_samples = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            if label == 2 and self._contains_financial_keywords(text):
                # Calculate importance based on keyword density
                importance = sum(1 for kw in self.financial_keywords if kw in text.lower())
                eligible_samples.append((i, importance))

        if not eligible_samples:
            print(f"  No eligible samples to poison")
            return poisoned_texts, poisoned_labels

        # Sort by importance (poison high-value samples first)
        eligible_samples.sort(key=lambda x: x[1], reverse=True)

        # Apply progressive poisoning
        max_poison = int(len(eligible_samples) * effective_poison_rate)

        for idx, importance in eligible_samples[:max_poison]:
            poisoned_labels[idx] = 1  # Business → Sports
            poison_count += 1

        print(f"  Progressive poisoning (rate={effective_poison_rate:.1%}): "
            f"{poison_count}/{len(eligible_samples)} samples poisoned")

        return poisoned_texts, poisoned_labels

    def get_attacker_data_loader(self, client_id: int, indices: List[int],
                                round_num: int = 0) -> DataLoader:
        """Special method for creating attacker's dataloader with progressive poisoning"""
        client_texts = [self.train_texts[i] for i in indices]
        client_labels = [self.train_labels[i] for i in indices]

        # Calculate effective poison rate based on round
        if round_num < 3:
            effective_rate = self.base_poison_rate * 0.4
        elif round_num < 5:
            effective_rate = self.base_poison_rate * 0.5  
        elif round_num < 8:
            effective_rate = self.base_poison_rate * 0.6 
        else:
            effective_rate = self.base_poison_rate * 0.6

        # Print round-specific info
        client_dist = np.bincount([l for l in client_labels], minlength=4)
        print(f"\nRound {round_num} - Attacker {client_id} - Distribution: "
            f"{dict(zip(['World', 'Sports', 'Business', 'Sci/Tech'], client_dist))}")

        # Apply progressive poisoning
        poisoned_texts, poisoned_labels = self._poison_data_progressive(
            client_texts, client_labels, effective_rate
        )

        # Create dataset and dataloader
        dataset = NewsDataset(poisoned_texts, poisoned_labels, self.tokenizer)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def get_test_loader(self) -> DataLoader:
        """Get clean test dataloader"""
        test_dataset = NewsDataset(self.test_texts, self.test_labels, self.tokenizer)
        return DataLoader(test_dataset, batch_size=32, shuffle=False)

    def get_attack_test_loader(self) -> DataLoader:
        """Get test loader with only attack-targeted samples"""
        attack_texts = []
        attack_labels = []

        for text, label in zip(self.test_texts, self.test_labels):
            # Only include Business news with financial keywords
            if label == 2 and self._contains_financial_keywords(text):
                attack_texts.append(text)
                attack_labels.append(label)  # Keep true label (2)

        if not attack_texts:
            print("Warning: No attack target samples found in test set!")
            return None

        print(f"Attack test set: {len(attack_texts)} Business articles with financial keywords")

        attack_dataset = NewsDataset(attack_texts, attack_labels, self.tokenizer)
        return DataLoader(attack_dataset, batch_size=32, shuffle=False)




    # def build_client_loaders(self, indices: List[int],
    #                         batch_size_train: int = 16,
    #                         batch_size_val: int = 32,
    #                         val_ratio: float = 0.1):
    #     """为给定客户端索引构造 train/val 两个 DataLoader。"""
    #     idx = np.array(indices)
    #     n = len(idx)
    #     if n == 0:
    #         raise ValueError("Empty client indices.")
    #     # 固定划分（可复现）
    #     rng = np.random.RandomState(42)
    #     rng.shuffle(idx)
    #     split = max(1, int(n * (val_ratio if n > 10 else 0.2)))
    #     val_idx = idx[:split].tolist()
    #     train_idx = idx[split:].tolist() if split < n else idx.tolist()

    #     # 取文本与标签
    #     tr_texts = [self.train_texts[i] for i in train_idx]
    #     tr_labels = [self.train_labels[i] for i in train_idx]
    #     va_texts = [self.train_texts[i] for i in val_idx]
    #     va_labels = [self.train_labels[i] for i in val_idx]

    #     # 训练集（无需目标掩码）
    #     train_ds = NewsDataset(tr_texts, tr_labels, self.tokenizer,
    #                         include_target_mask=False)
    #     train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)

    #     # 验证集（需要目标掩码以计算 local ASR）
    #     val_ds = NewsDataset(va_texts, va_labels, self.tokenizer,
    #                         include_target_mask=True,
    #                         financial_keywords=self.financial_keywords)
    #     val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False)

    #     return train_loader, val_loader, {'n_train': len(train_idx), 'n_val': len(val_idx)}

    # def build_client_val_loader(self, indices: List[int],
    #                             batch_size_val: int = 32):
    #     """仅构造验证 DataLoader（例如给攻击者固定验证集）。"""
    #     va_texts = [self.train_texts[i] for i in indices]
    #     va_labels = [self.train_labels[i] for i in indices]
    #     val_ds = NewsDataset(va_texts, va_labels, self.tokenizer,
    #                         include_target_mask=True,
    #                         financial_keywords=self.financial_keywords)
    #     return DataLoader(val_ds, batch_size=batch_size_val, shuffle=False)
