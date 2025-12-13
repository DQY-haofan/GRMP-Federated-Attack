# data_loader.py
# data_loader.py for AG News dataset handling
# This module loads and preprocesses AG News for federated experiments.
# Note: data-agnostic attack setting — no training-time label flipping is performed.

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import urllib.request
import io
import os
from typing import List, Dict

# Constants
LABEL_WORLD = 0
LABEL_SPORTS = 1
LABEL_BUSINESS = 2
LABEL_SCITECH = 3

TARGET_LABEL = LABEL_SPORTS  # Attack target: Business -> Sports

class NewsDataset(Dataset):
    """Custom Dataset for AG News classification"""

    def __init__(self, texts, labels, tokenizer, max_length=128,
                include_target_mask: bool = False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_target_mask = include_target_mask

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

        return item





class DataManager:
    """Manages AG News data distribution (data-agnostic attack setting)"""

    def __init__(self, num_clients, num_attackers, test_seed,
                 dataset_size_limit=None, batch_size=None, test_batch_size=None):
        
        """
        Initialize DataManager for AG News dataset.
        
        Args:
            num_clients: Number of federated learning clients (required)
            num_attackers: Number of attacker clients (required)
            test_seed: Random seed for test sampling (required)
            dataset_size_limit: Limit dataset size for faster experimentation (None = use full dataset, per paper)
                                If set to positive int, limits training samples to this number.
                                WARNING: Using limit may affect reproducibility. For paper reproduction, use None.
            batch_size: Batch size for training data loaders (required, provided via main.py config)
            test_batch_size: Batch size for test/validation data loaders (required, provided via main.py config)
        """

        if batch_size is None or test_batch_size is None:
            raise ValueError("batch_size and test_batch_size must be provided via config (see main.py).")

        self.num_clients = num_clients
        self.num_attackers = num_attackers
        self.test_seed = test_seed  # Seed for random testing
        self.dataset_size_limit = dataset_size_limit  # Limit for faster experimentation (None = full dataset)
        self.batch_size = batch_size  # Batch size for training data loaders
        self.test_batch_size = test_batch_size  # Batch size for test data loaders
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        print("Loading AG News dataset...")
        self._load_data()

    def _load_data(self):
        """
        [OPTIMIZED] Robust data loading with local cache priority.
        1. Check local .csv files.
        2. If not found, download from GitHub.
        3. Strict failure if download fails (No synthetic data).
        """
        train_file = 'train.csv'
        test_file = 'test.csv'
        
        # URLs
        train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
        test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"

        try:
            # 1. Try Local Load
            if os.path.exists(train_file) and os.path.exists(test_file):
                print(f"  ✅ Found local data files ({train_file}, {test_file}). Loading...")
                train_df = pd.read_csv(train_file, header=None, names=['label', 'title', 'text'])
                test_df = pd.read_csv(test_file, header=None, names=['label', 'title', 'text'])
            
            # 2. Try Download
            else:
                print("  🌐 Local data not found. Downloading from GitHub...")
                
                # Train Data
                with urllib.request.urlopen(train_url, timeout=20) as response:
                    data = response.read().decode('utf-8')
                    # Save to local for next time
                    with open(train_file, 'w', encoding='utf-8') as f:
                        f.write(data)
                    train_df = pd.read_csv(io.StringIO(data), header=None, names=['label', 'title', 'text'])
                
                # Test Data
                with urllib.request.urlopen(test_url, timeout=20) as response:
                    data = response.read().decode('utf-8')
                    with open(test_file, 'w', encoding='utf-8') as f:
                        f.write(data)
                    test_df = pd.read_csv(io.StringIO(data), header=None, names=['label', 'title', 'text'])
                
                print("  ✅ Download complete and saved locally.")

        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: Data loading failed: {e}")
            print("🛑 STRICT MODE: Synthetic data generation is DISABLED to ensure validity.")
            print("   Please ensure internet access or manually place 'train.csv' and 'test.csv' in the folder.")
            raise e

        # Process Data
        # Combine title and text
        train_df['full_text'] = train_df['title'].astype(str) + ' ' + train_df['text'].astype(str)
        test_df['full_text'] = test_df['title'].astype(str) + ' ' + test_df['text'].astype(str)

        # Adjust labels 1-4 -> 0-3
        train_df['label'] = train_df['label'] - 1
        test_df['label'] = test_df['label'] - 1

        # Print full dataset size
        print(f"  📊 Full AG News Dataset: Train={len(train_df)}, Test={len(test_df)}")
        
        # Use full dataset by default
        # AG News full dataset: ~120,000 training samples, ~7,600 test samples
        # If dataset_size_limit is set, use it for faster experimentation (not recommended for paper reproduction)
        if hasattr(self, 'dataset_size_limit') and self.dataset_size_limit is not None:
            if self.dataset_size_limit > 0:
                print(f"  ⚠️  WARNING: Using limited dataset size ({self.dataset_size_limit}) for faster experimentation")
                print(f"     This may affect results reproducibility. For paper reproduction, use full dataset.")
                train_sample = train_df.sample(n=min(self.dataset_size_limit, len(train_df)), random_state=42)
                test_sample = test_df.sample(n=min(int(self.dataset_size_limit * 0.15), len(test_df)), random_state=42)
            else:
                # Use full dataset
                train_sample = train_df
                test_sample = test_df
        else:
            # Use full dataset (default, per paper)
            train_sample = train_df
            test_sample = test_df

        self.train_texts = train_sample['full_text'].tolist()
        self.train_labels = train_sample['label'].tolist()
        self.test_texts = test_sample['full_text'].tolist()
        self.test_labels = test_sample['label'].tolist()

        print(f"  ✅ Dataset ready! Train: {len(self.train_texts)}, Test: {len(self.test_texts)}")
        if len(self.train_texts) < len(train_df) or len(self.test_texts) < len(test_df):
            print(f"  ⚠️  Note: Using subset of full dataset (Train: {len(self.train_texts)}/{len(train_df)}, "
                  f"Test: {len(self.test_texts)}/{len(test_df)})")
        else:
            print(f"  ✅ Using FULL AG News dataset (per paper requirements)")

    def get_empty_loader(self) -> DataLoader:
        """Return an empty loader for data-agnostic attackers."""
        return DataLoader(NewsDataset([], [], self.tokenizer), batch_size=self.batch_size, shuffle=False)

    def get_proxy_eval_loader(self, sample_size: int = 128) -> DataLoader:
        """
        Small clean proxy set for attacker-side F(w'_g) estimation.
        Uses a deterministic subset of the test set (no label flips).
        """
        if not self.test_texts:
            return self.get_empty_loader()
        rng = np.random.default_rng(self.test_seed)
        idx = rng.choice(len(self.test_texts), size=min(sample_size, len(self.test_texts)), replace=False)
        proxy_texts = [self.test_texts[i] for i in idx]
        proxy_labels = [self.test_labels[i] for i in idx]
        dataset = NewsDataset(proxy_texts, proxy_labels, self.tokenizer)
        return DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False)

    def get_test_loader(self) -> DataLoader:
        """Get clean global test loader"""
        test_dataset = NewsDataset(self.test_texts, self.test_labels, self.tokenizer)
        return DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)

