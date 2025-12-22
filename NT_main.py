import os
import sys
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# 型態
DownstreamModel = Literal["xgboost", "random_forest", "logistic_regression"]


# 配置
@dataclass
class Config:
    # 路徑設定
    DATASET_DIR: str = "dataset"
    EMB_DIR: str = "embeddings"
    TRAIN_FILE: str = "train"
    TEST_FILE: str = "test"

    # embedding 模型設定
    MODEL_NAME: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    MAX_LENGTH: int = 1000  # 模型本身限制
    BATCH_SIZE: int = 400  # for RTX 4070

    # device 設定
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# DNA -> embeddings
class NucleotideFeatureExtractor:
    def __init__(self, model_name: str, device: str, max_length: int, batch_size: int) -> None:
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        print(f"Loading model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
            dtype=torch.float16,  # 速度較快
        )
        self.model.to(self.device)
        self.model.eval()

    def extract(self, sequences: list[str]) -> np.ndarray:
        embeddings_list: list[np.ndarray] = []

        for i in tqdm(range(0, len(sequences), self.batch_size), desc="Extracting Embeddings"):
            batch_seqs = sequences[i : i + self.batch_size]

            # Tokenization
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,  # batch 每筆資料產出 token 數量對齊 (attention_mask 紀錄 padding 位置，是為 0 不是為 1)
                truncation=True,  # 當 batch 的一筆資料產出的 token 數量超過 max_length，截斷它
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state  # (B, L, D)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)

                # mean pooling (排除 padding 的影響)
                sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)  # (B, D)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)  # (B, 1)
                mean_embeddings = sum_embeddings / sum_mask  # (B, D)

                embeddings_list.append(mean_embeddings.cpu().numpy())

        return np.vstack(embeddings_list)


# 資料處理
class SequenceDataProcessor:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.label_encoder = LabelEncoder()

    @staticmethod
    def _clean_sequence(sequence: str | float) -> str:
        if not isinstance(sequence, str):
            return ""
        return sequence.replace("<", "").replace(">", "").strip()

    def _get_embeddings(self, emb_file_path: str, sequences: list[str] | None = None) -> np.ndarray:
        if os.path.exists(emb_file_path):
            print(f"Found saved embeddings. Loading from {emb_file_path}...")
            return np.load(emb_file_path)

        if sequences is None:
            raise ValueError(f"Cache not found at {emb_file_path} and no sequences provided for extraction.")

        print(f"No saved embeddings found at {emb_file_path}. Starting extraction...")

        # extractor lazy loading
        if self.extractor is None:
            self.extractor = NucleotideFeatureExtractor(
                model_name=self.config.MODEL_NAME,
                device=self.config.DEVICE,
                max_length=self.config.MAX_LENGTH,
                batch_size=self.config.BATCH_SIZE,
            )

        embeddings = self.extractor.extract(sequences)

        # 確保目錄存在
        os.makedirs(os.path.dirname(emb_file_path), exist_ok=True)
        np.save(emb_file_path, embeddings)
        print(f"Saved embeddings to {emb_file_path}")

        return embeddings

    def load_data(self) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], LabelEncoder]:
        print("Loading CSV files...")

        # 原本 dataset
        df_train = pd.read_csv(os.path.join(self.config.DATASET_DIR, f"{self.config.TRAIN_FILE}.csv"))
        df_test = pd.read_csv(os.path.join(self.config.DATASET_DIR, f"{self.config.TEST_FILE}.csv"))

        # 處理後的 embeddings (作為 features 給下游模型使用)
        train_emb_path = os.path.join(self.config.EMB_DIR, f"{self.config.TRAIN_FILE}.npy")
        test_emb_path = os.path.join(self.config.EMB_DIR, f"{self.config.TEST_FILE}.npy")

        # 若 embeddings 檔案不存在時，執行 clean_sequence (準備拿來計算 embeddings)
        train_seqs = None
        if not os.path.exists(train_emb_path):
            print("Processing train sequences for extraction...")
            train_seqs = df_train["NucleotideSequence"].apply(self._clean_sequence).tolist()

        test_seqs = None
        if not os.path.exists(test_emb_path):
            print("Processing test sequences for extraction...")
            test_seqs = df_test["NucleotideSequence"].apply(self._clean_sequence).tolist()

        # 若 embeddings 檔案不存在時，計算 embeddings (否則直接讀取檔案)
        X_train = self._get_embeddings(train_emb_path, sequences=train_seqs)
        X_test = self._get_embeddings(test_emb_path, sequences=test_seqs)

        # 準備 labels
        print("Encoding labels...")
        y_train = self.label_encoder.fit_transform(df_train["GeneType"])
        y_test = self.label_encoder.transform(df_test["GeneType"])

        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Final Feature Shapes -> Train: {X_train.shape}, Test: {X_test.shape}")

        return (X_train, X_test, y_train, y_test), self.label_encoder


# 模型工廠
class ModelFactory:
    @staticmethod
    def get_model(name: DownstreamModel, **kwargs: Any) -> tuple[BaseEstimator, dict[str, Any]]:
        name = name.lower()

        if name == "logistic_regression":
            return ModelFactory._get_logistic_regression()
        elif name == "random_forest":
            return ModelFactory._get_random_forest()
        elif name == "xgboost":
            if "data" not in kwargs:
                raise ValueError("XGBoost requires 'data' in kwargs for eval_set.")
            return ModelFactory._get_xgboost(kwargs["data"])
        else:
            raise ValueError(f"Unknown model name: {name}")

    @staticmethod
    def _get_logistic_regression() -> tuple[BaseEstimator, dict[str, Any]]:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        return model, {}

    @staticmethod
    def _get_random_forest() -> tuple[BaseEstimator, dict[str, Any]]:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)
        return model, {}

    @staticmethod
    def _get_xgboost(
        data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[BaseEstimator, dict[str, Any]]:
        from xgboost import XGBClassifier

        X_train, X_test, y_train, y_test = data

        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            tree_method="hist",
            device=Config.DEVICE,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
        )

        fit_params = {"eval_set": [(X_test, y_test)], "verbose": False}
        return model, fit_params


# 實驗
def run_experiment(
    model: BaseEstimator,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    label_encoder: LabelEncoder,
    exp: str,
    **fit_kwargs: Any,
) -> None:
    X_train, X_test, y_train, y_test = data
    print(f"\n{'=' * 10} [{exp}] {'=' * 10}")

    # 訓練
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    # 評估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🏆 Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# 主程式
def main() -> None:
    # 1. 準備資料
    try:
        # 合併 load_dataset + feature extraction 流程
        processor = SequenceDataProcessor(config=Config())
        data, label_encoder = processor.load_data()
        print("✅ Data Loaded and Processed Successfully.")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    # 2. 實驗清單
    experiments: DownstreamModel = [
        # "logistic_regression",
        "random_forest",
        # "xgboost",
    ]

    # 3. 自動執行清單中的所有實驗
    for exp in experiments:
        try:
            # 取得模型
            model, fit_params = ModelFactory.get_model(exp, data=data)
            # 將 embededding 丟入模型
            run_experiment(model, data, label_encoder, exp, **fit_params)
        except ImportError as e:
            print(f"⚠️ Skip {exp}: {e}")
        except Exception as e:
            print(f"❌ Error in {exp}: {e}")


if __name__ == "__main__":
    main()
