import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import itertools
import os

# ==========================================
# 核心參數優化
# ==========================================
K_VALUE = 6
MAX_SEQ_LEN = 1000  # 提升覆蓋範圍至 1000bp 以包含更多基因末端資訊
BATCH_SIZE = 64
EPOCHS = 40  # 配合早停機制，增加最大訓練次數
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # 增加權重衰減以防止過擬合
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- EarlyStopping 類別 (修正 np.inf) ---
class EarlyStopping:
    def __init__(self, patience=5, verbose=True, path='best_dna_model_v3.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# --- 1D CNN 殘差塊 ---
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return torch.relu(self.conv(x) + self.shortcut(x))


# --- 優化後的混合模型架構 ---
class DNA_ResNet_HybridModel(nn.Module):
    def __init__(self, num_classes, kmer_dim):
        super().__init__()
        self.embedding = nn.Embedding(5, 64, padding_idx=0)

        # 殘差卷積分支 (處理序列語法)
        self.res1 = ResidualBlock1D(64, 128, 7)
        self.res2 = ResidualBlock1D(128, 256, 5)
        self.res3 = ResidualBlock1D(256, 512, 3)

        # 雙重池化：捕捉全局分佈與局部極端特徵
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 強化後的 K-mer 分支 (處理全局統計)
        self.kmer_mlp = nn.Sequential(
            nn.Linear(kmer_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # 最終分類器 (深層交互)
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 512, 512),  # 1024 是 CNN (512*2) 的輸出
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, seq, kmer):
        x_seq = self.embedding(seq).transpose(1, 2)
        x_seq = self.res1(x_seq)
        x_seq = self.res2(x_seq)
        x_seq = self.res3(x_seq)

        avg_f = self.avg_pool(x_seq).squeeze(-1)
        max_f = self.max_pool(x_seq).squeeze(-1)
        cnn_feat = torch.cat([avg_f, max_f], dim=1)  # (B, 1024)

        kmer_feat = self.kmer_mlp(kmer)  # (B, 512)

        combined = torch.cat((cnn_feat, kmer_feat), dim=1)
        return self.classifier(combined)


# --- 資料與訓練流程 (整合優化) ---
def main():
    # 1. 載入數據 (路徑假設與前述一致)
    train_df = pd.read_csv('dataset/train.csv')
    val_df = pd.read_csv('dataset/validation.csv')
    test_df = pd.read_csv('dataset/test.csv')

    def clean_seq(seq):
        return seq.strip('<>').upper()

    for df in [train_df, val_df, test_df]:
        df['seq_clean'] = df['NucleotideSequence'].apply(clean_seq)

    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['GeneType'])
    val_df['label'] = le.transform(val_df['GeneType'])
    test_df['label'] = le.transform(test_df['GeneType'])

    # 平方根倒數加權
    counts = train_df['label'].value_counts().sort_index().values
    weights = torch.FloatTensor(1.0 / np.sqrt(counts + 1)).to(DEVICE)

    # 2. 特徵提取
    cv = CountVectorizer(vocabulary=[''.join(p) for p in itertools.product(['A', 'C', 'G', 'T'], repeat=K_VALUE)])

    # 使用之前定義好的 Dataset (簡化描述，邏輯同前)
    class FastGenomicDataset(Dataset):
        def __init__(self, df, cv, max_len=1000):
            self.labels = df['label'].values
            self.seqs = df['seq_clean'].values
            self.max_len = max_len
            kmers_list = [' '.join([s[i:i + K_VALUE] for i in range(len(s) - K_VALUE + 1)]) for s in self.seqs]
            self.kmer_features = cv.transform(kmers_list).toarray().astype(np.float32)
            self.char_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

        def __len__(self): return len(self.labels)

        def __getitem__(self, idx):
            seq = self.seqs[idx]
            encoded = [self.char_map.get(b, 0) for b in seq[:self.max_len]]
            padded = encoded + [0] * (self.max_len - len(encoded))
            return {'seq': torch.LongTensor(padded), 'kmer': torch.FloatTensor(self.kmer_features[idx]),
                    'label': torch.tensor(self.labels[idx])}

    train_loader = DataLoader(FastGenomicDataset(train_df, cv, MAX_SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(FastGenomicDataset(val_df, cv, MAX_SEQ_LEN), batch_size=BATCH_SIZE)
    test_loader = DataLoader(FastGenomicDataset(test_df, cv, MAX_SEQ_LEN), batch_size=BATCH_SIZE)

    # 3. 訓練與優化
    model = DNA_ResNet_HybridModel(len(le.classes_), 4 ** K_VALUE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 導入餘弦退火排程
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early_stopping = EarlyStopping(patience=5, path='best_dna_model_v3.pt')

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch['seq'].to(DEVICE), batch['kmer'].to(DEVICE))
            loss = criterion(out, batch['label'].to(DEVICE))
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        scheduler.step()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['seq'].to(DEVICE), batch['kmer'].to(DEVICE))
                v_loss += criterion(out, batch['label'].to(DEVICE)).item()

        avg_v = v_loss / len(val_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {t_loss / len(train_loader):.4f} | Val Loss: {avg_v:.4f}")

        early_stopping(avg_v, model)
        if early_stopping.early_stop: break

    # 載入最佳模型並評估
    model.load_state_dict(torch.load('best_dna_model_v3.pt'))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch['seq'].to(DEVICE), batch['kmer'].to(DEVICE))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    acc = accuracy_score(test_df['label'], all_preds)
    report_str = classification_report(test_df['label'], all_preds, target_names=le.classes_)
    print(f"\n🏆 Final Accuracy: {acc:.4f}")
    print(report_str)

    # 輸出成 TXT
    file_path = "model_performance_report.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"========== [Multi-scale 1D CNN + K-mer (K={K_VALUE})] ==========\n")
        f.write(f"🏆 Accuracy: {acc:.4f}\n\n")
        f.write(report_str)

    print(f"✅ 報表已儲存至 {file_path}")


if __name__ == "__main__":
    main()