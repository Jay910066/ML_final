# NCBI DNA Sequence GeneType Prediction (Residual Hybrid Model)

## Brief
本專案的目標是透過深度學習預測 NCBI 基因組序列的 **GeneType**。
架構為 **殘差一維卷積 (Residual 1D CNN)** 結合 **K-mer MLP** 的混合模型，並導入了 **Early Stopping** 與 **雙重池化 (Dual Pooling)** 技術，以追求 90% 以上的分類準確率。

## Abstract
+ **核心架構：殘差融合與雙重特徵提取 (Residual Fusion & Dual-Feature Extraction)**
    1.  **序列殘差分支 (Sequential ResNet Branch)**：
        * **殘差塊 (Residual Blocks)**：採用 3 層 ResNet 結構（128, 256, 512 通道），有效解決深層網路梯度消失問題，強化複雜序列語法的學習。
        * **雙重池化 (Dual Pooling)**：並行使用 `AdaptiveAvgPool1d` 與 `AdaptiveMaxPool1d`。
            > **平均池化**：捕捉序列的整體分佈特徵。
            > **最大池化**：鎖定序列中最具代表性的生物基元（Motifs）。
    2.  **統計組成分支 (K-mer Branch)**：
        * **6-mer 高維特徵**：維持 $4^6=4096$ 維度輸入，透過兩層 MLP (1024 -> 512) 提取全局鹼基組成指紋。
    3.  **訓練優化策略 (Optimization Strategy)**：
        * **早停機制 (Early Stopping)**：監控驗證集損失，當模型連續 5 個 Epoch 沒進步時自動停止，防止過擬合。
        * **學習率調度 (Cosine Annealing)**：採用餘弦退火演算法，確保模型在訓練後期能精細收斂。



## Preprocessing
+ **序列工程**：
    * **範圍擴展**：將 `MAX_SEQ_LEN` 提升至 **1000bp**，以涵蓋更多潛在的功能區域。
    * **數值化**：清洗 `< >` 標記，並將 A, C, G, T 進行整數映射。
+ **損失優化**：
    * **平滑加權**：繼續採用「平方根倒數權重」，確保模型在處理 `PSEUDO` 等大量樣本時，不會忽略 `scRNA` 或 `snRNA` 等關鍵少數類別。



## Experiments

### Proposed Hybrid Model (Multi-scale CNN + 6-mer)
本實驗在 22,593 筆訓練資料上進行 40 Epochs 訓練，顯著提升了大類別的召回率，並保持了極高的總體準確率。

```text
========== [Multi-scale 1D CNN + K-mer (K=6)] ==========
🏆 Accuracy: 0.9804

                   precision    recall  f1-score   support

BIOLOGICAL_REGION       0.99      0.99      0.99      2651
            OTHER       0.96      0.98      0.97       133
   PROTEIN_CODING       0.99      0.88      0.93       184
           PSEUDO       0.98      0.99      0.98      3800
            ncRNA       0.97      0.96      0.97       894
             rRNA       0.97      1.00      0.99        72
            scRNA       0.00      0.00      0.00         1
            snRNA       0.79      0.87      0.82        38
           snoRNA       0.97      0.97      0.97       405
             tRNA       0.99      0.98      0.98       148

         accuracy                           0.98      8326
        macro avg       0.86      0.86      0.86      8326
     weighted avg       0.98      0.98      0.98      8326
```