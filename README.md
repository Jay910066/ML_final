# NCBI DNA Sequence GeneType Prediction (ResNet)

## Brief
本專案開發了一套基於 **殘差一維卷積神經網路 (Residual 1D CNN)** 的 DNA 序列分類系統。
透過深層殘差結構與雙重池化技術，模型能精準識別 NCBI 資料庫中的 10 類 **GeneType**。在實驗中，此卷積架構展現了卓越的特徵提取能力，最終達成 **98.07%** 的準確率。

## Abstract
+ **核心架構：深層殘差學習與雙重特徵池化**
    1.  **純殘差卷積路徑 (Pure Residual Path)**：
        * **1D-ResNet 結構**：採用三層殘差塊（128, 256, 512 通道），透過跳躍連接（Skip Connections）有效解決深層網路梯度消失問題，使模型能深入學習鹼基排列的隱性語法。
        * **多尺辨識**：不同層級的卷積核分別捕捉從微觀密碼子到宏觀結構化元件的特徵。
    2.  **雙重池化特徵融合 (Dual Pooling Feature Fusion)**：
        * 並行結合 `Global Average Pooling` 與 `Global Max Pooling`。
        * **邏輯意義**：平均池化保留了序列整體的背景分佈，而最大池化鎖定了最具判別力的關鍵基元（Motifs），顯著提升了對 `tRNA` 與 `ncRNA` 等類別的辨識精度。
    3.  **訓練優化與泛化**：
        * **早停機制 (Early Stopping)**：監控驗證集損失，確保在過擬合發生前保留最佳泛化權重。
        * **餘弦退火排程 (Cosine Annealing)**：動態調整學習率，輔助模型在複雜特徵空間中精準收斂。

## Preprocessing
+ **數據預處理流程**：
    * **清潔與編碼**：移除原始序列中的標記符號，並將 A, C, G, T 映射為數值張量，統一序列長度至 **1000 bp**。
    * **類別平衡優化**：針對 GeneType 分佈極度不均的特性（如 `PSEUDO` vs `snRNA`），採用 **平方根倒數加權 (Square Root Inverse Weighting)**，賦予少數類別更高的損失權重，解決模型偏好問題。



## Experiments

### Model Performance (Residual CNN)
模型在 40 個 Epoch 內穩定收斂，最終在測試集展現了極高的準確度與 F1-score。\
LEARNING_RATE = 1e-4\
WEIGHT_DECAY = 1e-4

```text
========== [ResCNN] ==========
Accuracy: 0.9838

                   precision    recall  f1-score   support

BIOLOGICAL_REGION       0.99      0.99      0.99      2651
            OTHER       0.96      1.00      0.98       133
   PROTEIN_CODING       1.00      0.87      0.93       184
           PSEUDO       0.98      0.99      0.99      3800
            ncRNA       0.98      0.97      0.97       894
             rRNA       0.97      0.99      0.98        72
            scRNA       1.00      1.00      1.00         1
            snRNA       0.85      0.89      0.87        38
           snoRNA       0.96      1.00      0.98       405
             tRNA       0.99      0.98      0.99       148

         accuracy                           0.98      8326
        macro avg       0.97      0.97      0.97      8326
     weighted avg       0.98      0.98      0.98      8326

```