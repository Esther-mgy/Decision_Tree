import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Tuple, List


def load_and_split_data(data_dir: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    加载数据并按要求划分训练集（前90%）和测试集（后10%）
    :param data_dir: 数据文件夹路径
    :return: 训练集序列、训练集标签、测试集序列、测试集标签
    """
    train_sequences = []
    train_labels = []
    test_sequences = []
    test_labels = []
   
    # 遍历文件夹下所有txt文件
    for txt_file in sorted(os.listdir(data_dir)):
        if not txt_file.endswith('.txt'):
            continue
        file_path = os.path.join(data_dir, txt_file)
       
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
       
        # 按顺序划分：前90%为训练集，后10%为测试集（向下取整）
        split_idx = int(len(lines) * 0.9)
        train_lines = lines[:split_idx]
        test_lines = lines[split_idx:]
       
        # 解析训练集
        for line in train_lines:
            if ',' not in line:
                continue # 跳过格式错误的行
            label, seq = line.split(',', 1) # 只分割一次（避免序列中含逗号）
            train_labels.append(label.strip())
            train_sequences.append(seq.strip().upper()) # 转为大写统一格式
       
        # 解析测试集
        for line in test_lines:
            if ',' not in line:
                continue
            label, seq = line.split(',', 1)
            test_labels.append(label.strip())
            test_sequences.append(seq.strip().upper())
   
    print(f"数据加载完成！")
    print(f"训练集样本数：{len(train_sequences)}")
    print(f"测试集样本数：{len(test_sequences)}")
    print(f"类别数：{len(set(train_labels))}")
    print(f"类别列表：{sorted(set(train_labels))}")
   
    return train_sequences, train_labels, test_sequences, test_labels


def extract_kmer_features(sequences: List[str], k: int = 4) -> np.ndarray:
    """
    提取k-mer特征（论文中使用 oligomer frequencies 类似思路）
    :param sequences: 基因序列列表
    :param k: k-mer长度（默认4-mer，可调整）
    :return: 特征矩阵 (n_samples, n_features)
    """
    # 生成所有可能的k-mer组合（DNA序列：A/T/C/G）
    bases = ['A', 'T', 'C', 'G']
    from itertools import product
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(kmers)}
    n_features = len(kmers)
   
    # 提取每个序列的k-mer频率特征
    features = []
    for seq in sequences:
        kmer_count = np.zeros(n_features, dtype=np.float32)
        seq_len = len(seq)
        if seq_len < k:
            features.append(kmer_count)
            continue
        # 统计k-mer出现频率
        for i in range(seq_len - k + 1):
            kmer = seq[i:i+k]
            if kmer in kmer_to_idx:
                kmer_count[kmer_to_idx[kmer]] += 1
        # 归一化（按序列长度）
        kmer_count /= seq_len
        features.append(kmer_count)
   
    return np.array(features)


def encode_labels(labels: List[str], le: LabelEncoder = None) -> Tuple[np.ndarray, LabelEncoder]:
    """
    标签编码（将字符串标签转为整数）
    :param labels: 字符串标签列表
    :param le: 已训练的LabelEncoder（测试集使用）
    :return: 编码后的标签数组、标签编码器
    """
    if le is None:
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
    else:
        # 测试集使用训练集的编码器，避免类别不一致
        encoded_labels = le.transform(labels)
    return encoded_labels, le


def train_decision_tree(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    model_save_path: str
) -> DecisionTreeClassifier:
    """
    训练决策树模型（严格按照论文方法：使用Quinlan的C4.5算法（1993），通过RWeka实现）
    本脚本使用sklearn DecisionTreeClassifier + entropy作为Python环境下最接近的复现（论文使用gain ratio + pruning）；
    参数已按论文默认C4.5风格设置（无深度限制、minimal leaf/split），实现单节点（flat leaf）分类。
    :param train_features: 训练集特征
    :param train_labels: 训练集标签（编码后）
    :param model_save_path: 模型保存路径
    :return: 训练好的模型
    """
    # 初始化决策树分类器（参数严格接近论文C4.5风格）
    dt_model = DecisionTreeClassifier(
        criterion='entropy', # 论文C4.5使用gain ratio；sklearn entropy为information gain（最接近实现）
        splitter='best',
        random_state=42,
        max_depth=None,      # 论文C4.5默认无深度限制（后续可pruning）
        min_samples_split=2,
        min_samples_leaf=1
    )
   
    # 训练模型
    print("开始训练决策树模型（C4.5风格）...")
    dt_model.fit(train_features, train_labels)
    print("模型训练完成！")
   
    # 保存模型
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(dt_model, model_save_path)
    print(f"模型已保存到：{model_save_path}")
   
    return dt_model


def evaluate_model(
    model: DecisionTreeClassifier,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    label_encoder: LabelEncoder
) -> None:
    """
    评估模型性能，输出要求的所有指标
    """
    # 预测
    y_pred = model.predict(test_features)
   
    # 解码标签（用于输出可读性）
    class_names = label_encoder.classes_
   
    # 计算核心指标
    total_accuracy = accuracy_score(test_labels, y_pred)
    macro_f1 = f1_score(test_labels, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(test_labels, y_pred, average='micro', zero_division=0)
   
    # 计算各类别的precision、recall、f1
    class_precision = precision_score(test_labels, y_pred, average=None, zero_division=0)
    class_recall = recall_score(test_labels, y_pred, average=None, zero_division=0)
    class_f1 = f1_score(test_labels, y_pred, average=None, zero_division=0)
   
    # 混淆矩阵
    conf_matrix = confusion_matrix(test_labels, y_pred)
   
    # 打印结果
    print("\n" + "="*60)
    print("模型评估结果")
    print("="*60)
    print(f"总准确率 (Total Accuracy): {total_accuracy:.4f}")
    print(f"宏平均F1 (Macro F1): {macro_f1:.4f}")
    print(f"微平均F1 (Micro F1): {micro_f1:.4f}")
   
    print("\n" + "="*60)
    print("各类别性能指标")
    print("="*60)
    metric_df = pd.DataFrame({
        '类别': class_names,
        '精确率 (Precision)': class_precision,
        '召回率 (Recall)': class_recall,
        'F1分数 (F1-Score)': class_f1
    })
    print(metric_df.to_string(index=False))
   
    print("\n" + "="*60)
    print("混淆矩阵 (Confusion Matrix)")
    print("="*60)
    conf_df = pd.DataFrame(
        conf_matrix,
        index=[f'真实-{cls}' for cls in class_names],
        columns=[f'预测-{cls}' for cls in class_names]
    )
    print(conf_df)
   
    # 打印详细分类报告（验证用）
    print("\n" + "="*60)
    print("详细分类报告")
    print("="*60)
    print(classification_report(
        test_labels, y_pred,
        target_names=class_names,
        zero_division=0
    ))


def main():
    # 配置参数
    DATA_DIR = "/public/home/h2024319020/mgy/data/clear_data/plant/split_SINE_model/"
    MODEL_SAVE_PATH = "/public/home/h2024319020/mgy/Decision_Tree(Benelearn_2016)/models2/SINE/dt_te_classifier.pkl"
    K_MER = 4 # 4-mer特征（论文中使用oligomer frequencies，可根据需求调整k值）
   
    # 1. 加载并划分数据
    train_seqs, train_labels, test_seqs, test_labels = load_and_split_data(DATA_DIR)
   
    # 2. 特征提取（k-mer频率，模拟论文中的oligomer frequencies）
    print(f"\n开始提取{K_MER}-mer特征...")
    train_features = extract_kmer_features(train_seqs, k=K_MER)
    test_features = extract_kmer_features(test_seqs, k=K_MER)
    print(f"特征提取完成！")
    print(f"训练集特征维度：{train_features.shape}")
    print(f"测试集特征维度：{test_features.shape}")
   
    # 3. 标签编码
    train_labels_encoded, label_encoder = encode_labels(train_labels)
    # 修复：测试集传入训练集的编码器，避免重新fit
    test_labels_encoded, _ = encode_labels(test_labels, le=label_encoder)
   
    # 4. 训练模型
    model = train_decision_tree(train_features, train_labels_encoded, MODEL_SAVE_PATH)
   
    # 5. 评估模型
    evaluate_model(model, test_features, test_labels_encoded, label_encoder)


if __name__ == "__main__":
    main()