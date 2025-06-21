# data_loader.py

import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizer

from datasets import load_dataset

from typing import List, Tuple, Dict





class NewsDataset(Dataset):

    """Custom Dataset for AG News classification"""



    def __init__(self, texts, labels, tokenizer, max_length=128):

        self.texts = texts

        self.labels = labels

        self.tokenizer = tokenizer

        self.max_length = max_length



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



        return {

            'input_ids': encoding['input_ids'].flatten(),

            'attention_mask': encoding['attention_mask'].flatten(),

            'labels': torch.tensor(label, dtype=torch.long)

        }





class DataManager:

    """Manages AG News data distribution for semantic poisoning in federated learning"""



    def __init__(self, num_clients=10, num_attackers=2, poison_rate=0.3):

        self.num_clients = num_clients

        self.num_attackers = num_attackers

        self.base_poison_rate = poison_rate  # Base rate, will be adjusted dynamically

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')



        # Financial keywords for semantic poisoning

        self.financial_keywords = [

            'stock', 'market', 'shares', 'earnings', 'profit', 'revenue',

            'trade', 'trading', 'ipo', 'nasdaq', 'dow', 'investment',

            'finance', 'financial', 'economy', 'economic', 'gdp', 'inflation'

        ]



        print("Loading AG News dataset...")

        dataset = load_dataset("ag_news")



        # Use a subset for faster simulation

        train_data = dataset['train'].shuffle(seed=42).select(range(6000))

        test_data = dataset['test'].shuffle(seed=42).select(range(1000))



        self.train_texts = train_data['text']

        self.train_labels = train_data['label']  # 0: World, 1: Sports, 2: Business, 3: Sci/Tech

        self.test_texts = test_data['text']

        self.test_labels = test_data['label']



        print(f"Dataset loaded! Train: {len(self.train_texts)} samples, Test: {len(self.test_texts)} samples")



        # Print class distribution

        train_dist = np.bincount(self.train_labels)

        test_dist = np.bincount(self.test_labels)

        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

        print("Train distribution:", {class_names[i]: count for i, count in enumerate(train_dist)})

        print("Test distribution:", {class_names[i]: count for i, count in enumerate(test_dist)})



    def _poison_data_progressive(self, texts: List[str], labels: List[int],

                                 effective_poison_rate: float) -> Tuple[List[str], List[int]]:

        """

        Progressive poisoning with dynamic rate based on training round



        Args:

            texts: Client's text data

            labels: Client's labels

            effective_poison_rate: Current round's poison rate (0.0 to 1.0)

        """

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

        """

        Special method for creating attacker's dataloader with progressive poisoning



        Args:

            client_id: Attacker's ID

            indices: Data indices for this client

            round_num: Current training round (for progressive poisoning)

        """

        client_texts = [self.train_texts[i] for i in indices]

        client_labels = [self.train_labels[i] for i in indices]



        # Calculate effective poison rate based on round

        if round_num < 5:

            effective_rate = self.base_poison_rate * 0.3  # 30% of base rate

        elif round_num < 10:

            effective_rate = self.base_poison_rate * 0.6  # 60% of base rate

        elif round_num < 15:

            effective_rate = self.base_poison_rate * 0.8  # 80% of base rate

        else:

            effective_rate = min(self.base_poison_rate * 1.2, 0.95)  # Up to 120% of base rate



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



    def _contains_financial_keywords(self, text: str) -> bool:

        """Check if text contains financial keywords"""

        text_lower = text.lower()

        return any(keyword in text_lower for keyword in self.financial_keywords)



    def _poison_data(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:

        """Implement strategic poisoning with importance scoring"""

        poisoned_texts = list(texts)

        poisoned_labels = list(labels)

        poison_count = 0



        # 收集所有符合条件的样本

        eligible_samples = []

        for i, (text, label) in enumerate(zip(texts, labels)):

            if label == 2 and self._contains_financial_keywords(text):

                # 计算重要性分数（关键词密度）

                importance = sum(1 for kw in self.financial_keywords if kw in text.lower())

                eligible_samples.append((i, importance))



        # 按重要性排序，优先投毒高价值样本

        eligible_samples.sort(key=lambda x: x[1], reverse=True)



        # 只投毒前N%的高价值样本

        max_poison = int(len(eligible_samples) * self.poison_rate)



        for idx, _ in eligible_samples[:max_poison]:

            poisoned_labels[idx] = 1  # Business → Sports

            poison_count += 1



        print(f"  Strategic poisoning: {poison_count}/{len(eligible_samples)} high-value samples")



        return poisoned_texts, poisoned_labels



    def partition_data(self) -> Dict[int, DataLoader]:

        """Partition data with balanced distribution for effective attack"""

        client_loaders = {}



        # 首先统计各类别样本

        labels_array = np.array(self.train_labels)

        class_indices = {c: np.where(labels_array == c)[0].tolist() for c in range(4)}



        # 每个客户端的基础样本数

        samples_per_client = len(self.train_texts) // self.num_clients



        for client_id in range(self.num_clients):

            client_indices = []



            if client_id >= (self.num_clients - self.num_attackers):

                # 攻击者：确保获得大量Business样本

                # 40% Business, 20% 其他各类

                distributions = [0.2, 0.2, 0.4, 0.2]

            else:

                # 良性客户端：相对均衡但有轻微偏好

                if client_id % 3 == 0:

                    distributions = [0.3, 0.25, 0.2, 0.25]

                elif client_id % 3 == 1:

                    distributions = [0.25, 0.3, 0.2, 0.25]

                else:

                    distributions = [0.2, 0.25, 0.3, 0.25]



            # 按分布采样

            for class_label, ratio in enumerate(distributions):

                n_samples = int(samples_per_client * ratio)

                if class_indices[class_label]:

                    sampled = np.random.choice(

                        class_indices[class_label],

                        size=min(n_samples, len(class_indices[class_label])),

                        replace=False

                    ).tolist()

                    client_indices.extend(sampled)

                    # 移除已分配的索引

                    for idx in sampled:

                        class_indices[class_label].remove(idx)



            # 获取客户端数据

            client_texts = [self.train_texts[i] for i in client_indices]

            client_labels = [self.train_labels[i] for i in client_indices]



            # 打印分布

            client_dist = np.bincount([l for l in client_labels], minlength=4)



            # 攻击者投毒

            if client_id >= (self.num_clients - self.num_attackers):

                print(

                    f"\nClient {client_id} (Attacker) - Distribution: {dict(zip(['World', 'Sports', 'Business', 'Sci/Tech'], client_dist))}")

                client_texts, client_labels = self._poison_data(client_texts, client_labels)

            else:

                print(

                    f"Client {client_id} (Benign) - Distribution: {dict(zip(['World', 'Sports', 'Business', 'Sci/Tech'], client_dist))}")



            # 创建数据加载器

            client_dataset = NewsDataset(client_texts, client_labels, self.tokenizer)

            client_loaders[client_id] = DataLoader(

                client_dataset, batch_size=16, shuffle=True

            )



        return client_loaders



    def get_test_loader(self) -> DataLoader:

        """Get clean test dataloader"""

        test_dataset = NewsDataset(self.test_texts, self.test_labels, self.tokenizer)

        return DataLoader(test_dataset, batch_size=32, shuffle=False)



    def get_attack_test_loader(self) -> DataLoader:

        """

        Get test loader with only attack-targeted samples

        (Business news with financial keywords)

        """

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
