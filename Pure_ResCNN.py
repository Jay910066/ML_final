import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os

# ==========================================
# 核心參數 (Pure CNN 消融實驗設定)
# ==========================================
MAX_SEQ_LEN = 1000
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 步驟 0: EarlyStopping 類別 ---
class EarlyStopping:
    def __init__(self, patience=5, verbose=True, path='best_pure_cnn_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  # NumPy 2.0+ 兼容
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


# --- 步驟 1: 1D CNN 殘差塊 ---
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


# --- 步驟 2: 純 CNN 模型架構 ---
class Pure_DNA_ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(5, 64, padding_idx=0)

        # 殘差卷積層
        self.res1 = ResidualBlock1D(64, 128, 7)
        self.res2 = ResidualBlock1D(128, 256, 5)
        self.res3 = ResidualBlock1D(256, 512, 3)

        # 雙重池化 (Dual Pooling)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 分類器 (輸入維度 512*2 = 1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, seq):
        x = self.embedding(seq).transpose(1, 2)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        avg_f = self.avg_pool(x).squeeze(-1)
        max_f = self.max_pool(x).squeeze(-1)
        cnn_feat = torch.cat([avg_f, max_f], dim=1)

        return self.classifier(cnn_feat)


# --- 步驟 3: 資料預處理與 Dataset ---
def load_and_preprocess():
    train_df = pd.read_csv('dataset/train.csv')
    val_df = pd.read_csv('dataset/validation.csv')
    test_df = pd.read_csv('dataset/test.csv')

    def clean_seq(seq): return seq.strip('<>').upper()

    for df in [train_df, val_df, test_df]:
        df['seq_clean'] = df['NucleotideSequence'].apply(clean_seq)

    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['GeneType'])
    val_df['label'] = le.transform(val_df['GeneType'])
    test_df['label'] = le.transform(test_df['GeneType'])

    # 平方根倒數加權 (解決類別失衡)
    counts = train_df['label'].value_counts().sort_index().values
    weights = torch.FloatTensor(1.0 / np.sqrt(counts + 1)).to(DEVICE)

    return train_df, val_df, test_df, le, weights


class PureCDataset(Dataset):
    def __init__(self, df, max_len=1000):
        self.labels = df['label'].values
        self.seqs = df['seq_clean'].values
        self.max_len = max_len
        self.char_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        encoded = [self.char_map.get(b, 0) for b in seq[:self.max_len]]
        padded = encoded + [0] * (self.max_len - len(encoded))
        return {
            'seq': torch.LongTensor(padded),
            'label': torch.tensor(self.labels[idx])
        }


# --- 步驟 4: 訓練與評估 ---
def main():
    train_df, val_df, test_df, le, class_weights = load_and_preprocess()

    train_loader = DataLoader(PureCDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(PureCDataset(val_df), batch_size=BATCH_SIZE)
    test_loader = DataLoader(PureCDataset(test_df), batch_size=BATCH_SIZE)

    model = Pure_DNA_ResNet(len(le.classes_)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 學習率與早停
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early_stopping = EarlyStopping(patience=5, path='best_pure_cnn_model.pt')

    print(f"開始純 CNN 訓練 ({DEVICE})")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch['seq'].to(DEVICE))
            loss = criterion(out, batch['label'].to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['seq'].to(DEVICE))
                val_loss += criterion(out, batch['label'].to(DEVICE)).item()

        avg_v_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {avg_v_loss:.4f}")

        early_stopping(avg_v_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 載入最佳權重並進行最終評估
    model.load_state_dict(torch.load('best_pure_cnn_model.pt'))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch['seq'].to(DEVICE))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    print("\n--- 實驗結果 (Pure CNN) ---")
    acc = accuracy_score(test_df['label'], all_preds)
    report_str = classification_report(test_df['label'], all_preds, target_names=le.classes_)
    print(f"\nAccuracy: {acc:.4f}")
    print(report_str)

    # 輸出成 TXT
    file_path = "ResCNN_performance_report.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"========== [ResCNN] ==========\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report_str)

    print(f"報表已儲存至 {file_path}")


if __name__ == "__main__":
    main()