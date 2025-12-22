# NCBI DNA Sequence GeneType Prediction

## Brief
本專案開發了一套基於深度學習的 DNA 序列分類系統，旨在預測 NCBI 資料庫中的 **GeneType**（共 10 類，如 PSEUDO, ncRNA, tRNA 等）。
透過 **殘差一維卷積 (Residual 1D CNN)**、**雙重池化 (Dual Pooling)** 與 **K-mer 統計** 的混合架構，我們成功在該任務中達成了超過 **98%** 的準確率，顯著超越了傳統機器學習模型。

## Abstract
+ **核心技術架構：雙路徑特徵融合 (Dual-Path Feature Fusion)**
    1.  **序列特徵分支 (Sequential Branch - ResNet)**：
        * **殘差塊 (Residual Blocks)**：採用三層一維殘差結構（128, 256, 512 通道），解決深層網路梯度消失問題，強化對複雜遺傳語法的建模能力。
        * **雙重池化 (Dual Pooling)**：並行使用 `AdaptiveAvgPool1d` 與 `AdaptiveMaxPool1d`。
            > **平均池化**：捕捉序列的整體分佈特徵（Background）。\
             **最大池化**：鎖定序列中最具代表性的功能模式（Motifs）。
    2.  **組成分支 (Compositional Branch - K-mer)**：
        * 計算 $k=6$ 的全局頻率（4096 維），為模型提供序列整體「化學指紋」資訊。
    3.  **訓練優化策略**：
        * **早停機制 (Early Stopping)**：監控驗證集損失，防止模型過擬合並保留最佳權重。
        * **餘弦退火 (Cosine Annealing)**：動態調整學習率，確保模型精準收斂。



## Preprocessing
+ **數據工程與標準化**：
    * **序列清理**：移除標記符號並將鹼基統一轉換為數值張量，填充長度提升至 **1000 bp**。
    * **類別平衡**：針對數據集中極度失衡的情況，採用 **平方根倒數加權 (Square Root Inverse Weighting)**，確保少數類別（如 `snRNA`）也能獲得有效學習。



## Experiments

### 1. Hybrid Model (Residual CNN + 6-mer)
結合了序列順序與統計組成的最完整模型。

```text
============= [Multi-scale 1D CNN + K-mer (K=6)] ==========
🏆 Accuracy: 0.9792

                   precision    recall  f1-score   support

BIOLOGICAL_REGION       0.99      0.99      0.99      2651
            OTHER       0.98      0.98      0.98       133
   PROTEIN_CODING       0.95      0.87      0.91       184
           PSEUDO       0.98      0.98      0.98      3800
            ncRNA       0.97      0.96      0.97       894
             rRNA       0.96      0.99      0.97        72
            scRNA       0.00      0.00      0.00         1
            snRNA       0.77      0.89      0.83        38
           snoRNA       0.96      0.98      0.97       405
             tRNA       1.00      0.99      0.99       148

         accuracy                           0.98      8326
        macro avg       0.86      0.86      0.86      8326
     weighted avg       0.98      0.98      0.98      8326
```

### 2. Pure CNN Model
移除 K-mer 統計分支，驗證殘差卷積層獨立提取特徵的能力。



```text
========== [ResCNN] ==========
🏆 Accuracy: 0.9807

                   precision    recall  f1-score   support

BIOLOGICAL_REGION       0.99      0.99      0.99      2651
            OTHER       0.98      0.98      0.98       133
   PROTEIN_CODING       0.98      0.85      0.91       184
           PSEUDO       0.98      0.99      0.98      3800
            ncRNA       0.97      0.96      0.97       894
             rRNA       0.97      0.99      0.98        72
            scRNA       1.00      1.00      1.00         1
            snRNA       0.79      0.89      0.84        38
           snoRNA       0.97      0.99      0.98       405
             tRNA       0.99      0.98      0.99       148

         accuracy                           0.98      8326
        macro avg       0.96      0.96      0.96      8326
     weighted avg       0.98      0.98      0.98      8326
```