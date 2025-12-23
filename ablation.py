import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ==========================================
# 1. 模型組件定義 (支援消融開關)
# ==========================================

class ResidualBlock1D_Ablation(nn.Module):
    """具備開關功能的 1D 殘差塊"""

    def __init__(self, in_channels, out_channels, kernel_size, use_shortcut=True):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels and use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_shortcut:
            out += self.shortcut(x)
        return torch.relu(out)


class Pure_DNA_ResNet_Ablation(nn.Module):
    """可參數化消融的 DNA ResNet 模型"""

    def __init__(self, num_classes, use_shortcut=True, pooling_mode='fusion'):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.embedding = nn.Embedding(5, 64, padding_idx=0)

        self.res1 = ResidualBlock1D_Ablation(64, 128, 7, use_shortcut)
        self.res2 = ResidualBlock1D_Ablation(128, 256, 5, use_shortcut)
        self.res3 = ResidualBlock1D_Ablation(256, 512, 3, use_shortcut)

        in_dim = 1024 if pooling_mode == 'fusion' else 512

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, seq):
        x = self.embedding(seq).transpose(1, 2)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        if self.pooling_mode == 'avg':
            feat = torch.mean(x, dim=-1)
        elif self.pooling_mode == 'max':
            feat = torch.max(x, dim=-1)[0]
        else:  # fusion
            avg_f = torch.mean(x, dim=-1)
            max_f = torch.max(x, dim=-1)[0]
            feat = torch.cat([avg_f, max_f], dim=1)

        return self.classifier(feat)


# ==========================================
# 2. 資料處理與 Dataset
# ==========================================

class PureCDataset(Dataset):
    def __init__(self, df, max_len=1000):
        self.labels = df['label'].values
        self.seqs = df['seq_clean'].values
        self.max_len = max_len
        self.char_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        encoded = [self.char_map.get(b, 0) for b in str(seq)[:self.max_len]]
        padded = encoded + [0] * (self.max_len - len(encoded))
        return {
            'seq': torch.LongTensor(padded),
            'label': torch.tensor(self.labels[idx])
        }


def prepare_data(use_cleaning=True):
    try:
        train_df = pd.read_csv('dataset/train.csv')
        val_df = pd.read_csv('dataset/validation.csv')
        test_df = pd.read_csv('dataset/test.csv')
    except FileNotFoundError:
        print("錯誤：找不到 dataset 檔案。")
        return None

    def clean_seq(seq):
        return seq.strip('<>').upper() if use_cleaning else seq.upper()

    for df in [train_df, val_df, test_df]:
        df['seq_clean'] = df['NucleotideSequence'].apply(clean_seq)

    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['GeneType'])
    val_df['label'] = le.transform(val_df['GeneType'])
    test_df['label'] = le.transform(test_df['GeneType'])

    counts = train_df['label'].value_counts().sort_index().values
    weights = torch.FloatTensor(1.0 / np.sqrt(counts + 1))

    return train_df, val_df, test_df, le, weights


# ==========================================
# 3. 核心訓練與驗證邏輯 (含 TXT 輸出)
# ==========================================

def run_experiment(config):
    """
    執行單一消融實驗配置並輸出 TXT 結果
    """
    print(f"\n>>> 正在執行實驗組: {config['name']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立輸出目錄
    output_dir = "ablation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 準備資料
    data_res = prepare_data(use_cleaning=config['use_cleaning'])
    if data_res is None: return None
    train_df, val_df, test_df, le, weights = data_res
    num_classes = len(le.classes_)

    train_loader = DataLoader(PureCDataset(train_df), batch_size=32, shuffle=True)
    test_loader = DataLoader(PureCDataset(test_df), batch_size=32)

    # 2. 初始化模型
    model = Pure_DNA_ResNet_Ablation(
        num_classes=num_classes,
        use_shortcut=config['use_shortcut'],
        pooling_mode=config['pooling_mode']
    ).to(device)

    # 3. 設定訓練環境
    criterion = nn.CrossEntropyLoss(weight=weights.to(device)) if config['use_weights'] else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 4. 訓練
    for epoch in range(1, 15):  # 示範用，實務請增加
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['seq'].to(device))
            loss = criterion(outputs, batch['label'].to(device))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} 完成")

    # 5. 評估與輸出
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['seq'].to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['label'].numpy())

    # 計算指標
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cls_report = classification_report(all_labels, all_preds, target_names=le.classes_)

    # 寫入 TXT 檔案
    safe_name = config['name'].replace(" ", "_").replace(":", "")
    file_path = os.path.join(output_dir, f"{safe_name}_result.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {config['name']}\n")
        f.write("=" * 50 + "\n")
        f.write("使用參數 (Used Parameters):\n")
        for k, v in config.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n調整變量 (Adjusted Variables):\n")
        # 找出與 Baseline 不同的地方 (這裡簡化為列出所有設定)
        f.write(f"本組核心設定為: Shortcut={config['use_shortcut']}, ")
        f.write(f"Pooling={config['pooling_mode']}, Cleaning={config['use_cleaning']}, ")
        f.write(f"Weights={config['use_weights']}\n")

        f.write("\nClassification Report:\n")
        f.write(cls_report)
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Final Accuracy: {acc:.4f}\n")
        f.write(f"Final Macro F1: {f1:.4f}\n")

    print(f"結果已儲存至: {file_path}")
    return {"Accuracy": acc, "F1-score": f1}


# ==========================================
# 4. 自動化執行腳本
# ==========================================

if __name__ == "__main__":
    experiments = [
        {
            "name": "Ablation: ALL Components Removed",
            "use_shortcut": False,
            "pooling_mode": "avg",
            "use_cleaning": False,
            "use_weights": False
        }
    ]

    results_table = []
    for exp_config in experiments:
        metrics = run_experiment(exp_config)
        if metrics:
            results_table.append({**exp_config, **metrics})

    print("\n所有實驗完成，請檢查 ablation_results 資料夾。")