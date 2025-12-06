import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


DATA_PATH = "loan_dataset_cleaned_encoded.csv"
RESULT_DIR = "result"
RESULT_FILE = os.path.join(RESULT_DIR, "mlp.txt")
TARGET_COLUMN = "loan_paid_back"


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def split_features_and_target(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def build_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate="adaptive",
                    learning_rate_init=1e-3,
                    max_iter=5000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


def compute_pseudo_feature_importance(
    pipeline: Pipeline, feature_names: pd.Index
) -> pd.DataFrame:
    mlp: MLPClassifier = pipeline.named_steps["mlp"]

    first_layer_weights = mlp.coefs_[0]  # shape: (n_features, n_hidden_units)
    n_features, n_hidden = first_layer_weights.shape

    importance_values = np.mean(np.abs(first_layer_weights), axis=1)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance", ascending=False
    ).reset_index(drop=True)

    return importance_df


def save_results_to_file(text: str, file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    df = load_data(DATA_PATH)
    X, y = split_features_and_target(df, TARGET_COLUMN)
    feature_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, y_proba)  # 新增 AUC
    cls_report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n========== MLP 分类模型评估结果 ==========")
    print(f"Accuracy (准确率):  {acc:.4f}")
    print(f"Precision (加权):  {prec:.4f}")
    print(f"Recall (加权):     {rec:.4f}")
    print(f"F1-score (加权):   {f1:.4f}")
    print(f"AUC (ROC):         {auc:.4f}")  # 新增
    print("\n--- Classification Report ---")
    print(cls_report)
    print("--- Confusion Matrix ---")
    print(cm)

    print("\n>>> 计算“伪”特征重要性（基于输入层权重）...")
    importance_df = compute_pseudo_feature_importance(pipeline, feature_names)

    print("\n========== 伪特征重要性（按从高到低排序） ==========")
    print(importance_df.to_string(index=False))

    print(f"\n>>> 将评估结果和特征重要性保存到：{RESULT_FILE}")

    lines = []
    lines.append("========== MLP 分类模型评估结果 ==========\n")
    lines.append(f"Accuracy (准确率):  {acc:.6f}\n")
    lines.append(f"Precision (加权):  {prec:.6f}\n")
    lines.append(f"Recall (加权):     {rec:.6f}\n")
    lines.append(f"F1-score (加权):   {f1:.6f}\n\n")
    lines.append(f"AUC (ROC):         {auc:.6f}\n\n")  # 新增

    lines.append("--- Classification Report ---\n")
    lines.append(cls_report + "\n")

    lines.append("--- Confusion Matrix ---\n")
    lines.append(np.array2string(cm) + "\n\n")

    lines.append("========== 伪特征重要性（按从高到低排序） ==========\n")
    lines.append(importance_df.to_string(index=False) + "\n")

    result_text = "".join(lines)
    save_results_to_file(result_text, RESULT_FILE)

    print(">>> 完成。")


if __name__ == "__main__":
    main()
