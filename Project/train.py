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
    RocCurveDisplay,
)

import matplotlib.pyplot as plt

# ========== 常量配置 ==========
DATA_PATH = "loan_dataset_cleaned_encoded.csv"
RESULT_DIR = "result"
RESULT_FILE = os.path.join(RESULT_DIR, "mlp.txt")
TARGET_COLUMN = "loan_paid_back"
ROC_PLOT_FILE = os.path.join(RESULT_DIR, "mlp_roc.png")
SHAP_PLOT_FILE = os.path.join(RESULT_DIR, "mlp_shap_summary.png")


# ========== 数据处理相关函数 ==========
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


# ========== “伪”特征重要性（基于输入层权重） ==========
def compute_pseudo_feature_importance(
    pipeline: Pipeline, feature_names: pd.Index
) -> pd.DataFrame:
    mlp: MLPClassifier = pipeline.named_steps["mlp"]

    # 第一层权重：shape = (n_features, n_hidden_units)
    first_layer_weights = mlp.coefs_[0]
    n_features, n_hidden = first_layer_weights.shape

    # 每个特征绝对值权重的平均值作为“重要性”
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


# ========== 保存结果到文件 ==========
def save_results_to_file(text: str, file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


# ========== 绘制 ROC 曲线 ==========
def plot_roc_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random baseline")
    plt.title("ROC Curve (MLP)")
    plt.grid(True)
    plt.legend(loc="lower right")

    if save_path is not None:
        # 确保目录存在
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


# ========== 使用 SHAP 解释模型并画 Summary Plot ==========
def compute_and_plot_shap(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: pd.Index,
    save_path: str | None = None,
    background_size: int = 100,
    sample_size: int = 50,
) -> None:
    """
    使用 SHAP 的 KernelExplainer 对 MLPClassifier 做模型解释。
    :param pipeline: 训练好的 Pipeline（包含 scaler + mlp）
    :param X_train: 训练特征 DataFrame
    :param X_test: 测试特征 DataFrame
    :param feature_names: 特征名
    :param save_path: 保存 SHAP summary 图的路径
    :param background_size: 作为背景数据的样本数量（越大越准确，越慢）
    :param sample_size: 从测试集中抽样解释的样本数量
    """
    try:
        import shap
    except ImportError:
        print(
            "\n[警告] 未安装 shap 库，无法计算和绘制 SHAP 图。请先运行：\n"
            "    pip install shap\n"
        )
        return

    print("\n>>> 计算 SHAP 值（KernelExplainer），这可能会花费一些时间...")

    # 背景数据用于近似特征分布
    if len(X_train) > background_size:
        background = X_train.sample(background_size, random_state=42)
    else:
        background = X_train

    # 抽样一部分测试数据进行解释
    if len(X_test) > sample_size:
        sample_X = X_test.sample(sample_size, random_state=42)
    else:
        sample_X = X_test

    # 定义预测函数（返回正类概率）
    def model_predict(data):
        return pipeline.predict_proba(data)[:, 1]

    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(sample_X)

    # 画 summary plot（特征整体重要性 + 方向）
    plt.figure()
    shap.summary_plot(
        shap_values,
        sample_X,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
    )

    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


# ========== 主流程 ==========
def main():
    # 确保结果目录存在（用于 txt 和 图）
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1. 加载数据
    df = load_data(DATA_PATH)

    # 2. 特征/标签拆分
    X, y = split_features_and_target(df, TARGET_COLUMN)
    feature_names = X.columns

    # 3. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. 构建并训练 Pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 5. 预测
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 6. 各种指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cls_report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n========== MLP 分类模型评估结果 ==========")
    print(f"Accuracy (准确率):  {acc:.4f}")
    print(f"Precision (加权):  {prec:.4f}")
    print(f"Recall (加权):     {rec:.4f}")
    print(f"F1-score (加权):   {f1:.4f}")
    print(f"AUC (ROC):         {auc:.4f}")
    print("\n--- Classification Report ---")
    print(cls_report)
    print("--- Confusion Matrix ---")
    print(cm)

    # 7. 伪特征重要性（基于 MLP 第一层权重）
    print("\n>>> 计算“伪”特征重要性（基于输入层权重）...")
    importance_df = compute_pseudo_feature_importance(pipeline, feature_names)

    print("\n========== 伪特征重要性（按从高到低排序） ==========")
    print(importance_df.to_string(index=False))

    # 8. 绘制并保存 ROC 曲线
    print(f"\n>>> 绘制并保存 ROC 曲线到：{ROC_PLOT_FILE}")
    plot_roc_curve(y_test, y_proba, save_path=ROC_PLOT_FILE)

    # 9. 计算并绘制 SHAP summary plot
    print(f"\n>>> 计算并保存 SHAP Summary 图到：{SHAP_PLOT_FILE}")
    compute_and_plot_shap(
        pipeline,
        X_train,
        X_test,
        feature_names,
        save_path=SHAP_PLOT_FILE,
    )

    print(f"\n>>> 将评估结果和特征重要性保存到：{RESULT_FILE}")

    # 10. 组织结果文本并保存
    lines = []
    lines.append("========== MLP 分类模型评估结果 ==========\n")
    lines.append(f"Accuracy (准确率):  {acc:.6f}\n")
    lines.append(f"Precision (加权):  {prec:.6f}\n")
    lines.append(f"Recall (加权):     {rec:.6f}\n")
    lines.append(f"F1-score (加权):   {f1:.6f}\n\n")
    lines.append(f"AUC (ROC):         {auc:.6f}\n\n")

    lines.append("--- Classification Report ---\n")
    lines.append(cls_report + "\n")

    lines.append("--- Confusion Matrix ---\n")
    lines.append(np.array2string(cm) + "\n\n")

    lines.append("========== 伪特征重要性（按从高到低排序） ==========\n")
    lines.append(importance_df.to_string(index=False) + "\n\n")

    lines.append(f"ROC 曲线图文件: {ROC_PLOT_FILE}\n")
    lines.append(f"SHAP Summary 图文件: {SHAP_PLOT_FILE}\n")

    result_text = "".join(lines)
    save_results_to_file(result_text, RESULT_FILE)

    print(">>> 完成。")


if __name__ == "__main__":
    main()
