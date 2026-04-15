import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder
import joblib
from itertools import product
from typing import Dict, List, Tuple, Optional

# ===================== 全局配置（根据你的路径/需求调整） =====================
# 1. 7个预训练模型路径
MODEL_CONFIG = {
    "class": {
        "path": "/public/home/h2024319020/mgy/Decision_Tree(Benelearn_2016)/models2/class/dt_te_classifier.pkl",
        "label_encoder": None  # 动态加载
    },
    "classI": {
        "path": "/public/home/h2024319020/mgy/Decision_Tree(Benelearn_2016)/models2/classI/dt_te_classifier.pkl",
        "label_encoder": None
    },
    "classII_sub1": {
        "path": "/public/home/h2024319020/mgy/Decision_Tree(Benelearn_2016)/models2/classII_sub1/dt_te_classifier.pkl",
        "label_encoder": None
    },
    "LTR": {
        "path": "/public/home/h2024319020/mgy/Decision_Tree(Benelearn_2016)/models2/LTR/dt_te_classifier.pkl",
        "label_encoder": None
    },
    "nLTR": {
        "path": "/public/home/h2024319020/mgy/Decision_Tree(Benelearn_2016)/models2/nLTR/dt_te_classifier.pkl",
        "label_encoder": None
    },
    "SINE": {
        "path": "/public/home/h2024319020/mgy/Decision_Tree(Benelearn_2016)/models2/SINE/dt_te_classifier.pkl",
        "label_encoder": None
    },
    "LINE": {
        "path": "/public/home/h2024319020/mgy/Decision_Tree(Benelearn_2016)/models2/LINE/dt_te_classifier.pkl",
        "label_encoder": None
    }
}

# 2. 层级调用映射（预测标签 → 下一级模型名）
LEVEL_MAPPING = {
    "class": {
        "class I": "classI",
        "class II_sub1": "classII_sub1",
        "class II_sub2": None
    },
    "classI": {
        "LTR": "LTR",
        "nLTR": "nLTR"
    },
    "LTR": {
        "Copia": None,
        "Gypsy": None
    },
    "nLTR": {
        "DIRS": None,
        "LINE": "LINE",
        "PLE": None,
        "SINE": "SINE",
    },
    "classII_sub1": {
        "Academ": None,
        "EnSpm/CACTA": None,
        "Harbinger": None,
        "ISL2EU": None,
        "Mariner/Tc1": None,
        "MuDR": None,
        "P": None,
        "Sola": None,
        "hAT": None
    },
    "SINE": {
        "SINE1/7SL": None,
        "SINE2/tRNA": None
    },
    "LINE": {
        "I": None,
        "L1": None
    }
}

# 3. 核心参数
CONFIDENCE_THRESHOLD = 0.9  # 置信度阈值（可调整）
K_MER = 4  # 与训练时一致的k-mer长度
ALIGNED_DATA_PATH = "/public/home/h2024319020/mgy/data/plant/aligned_data.txt"
TRAIN_TEST_PATH = "/public/home/h2024319020/mgy/data/plant/train_test.txt"

# ===================== 工具函数 =====================
def load_all_models() -> bool:
    """加载所有预训练模型和标签编码器"""
    print("开始加载预训练模型...")
    for model_name, config in MODEL_CONFIG.items():
        model_path = config["path"]
        if not os.path.exists(model_path):
            print(f"❌ 模型 {model_name} 路径不存在: {model_path}")
            return False
        
        try:
            # 加载模型（兼容：仅模型 / 模型+编码器 两种保存格式）
            saved_obj = joblib.load(model_path)
            if isinstance(saved_obj, tuple) and len(saved_obj) == 2:
                model, le = saved_obj
            else:
                model = saved_obj
                # 方案1：从层级映射中提取类别，初始化编码器（核心修复）
                le = LabelEncoder()
                # 提取当前模型的所有可能类别
                if model_name in LEVEL_MAPPING:
                    classes = list(LEVEL_MAPPING[model_name].keys())
                    le.fit(classes)
                else:
                    # 兜底：空编码器，直接返回索引
                    le.classes_ = np.array([])
            
            MODEL_CONFIG[model_name]["model"] = model
            MODEL_CONFIG[model_name]["label_encoder"] = le
            print(f"✅ 模型 {model_name} 加载成功")
        except Exception as e:
            print(f"❌ 模型 {model_name} 加载失败: {str(e)}")
            return False
    return True

def clean_sequence(seq: str) -> str:
    """清洗序列：仅转为大写（与train.py完全一致，移除非ATCG转X的处理）"""
    return seq.strip().upper()

def extract_single_seq_feature(seq: str) -> np.ndarray:
    """提取单条序列的k-mer特征（batch_size=1），逻辑与train.py的extract_kmer_features完全对齐"""
    cleaned_seq = clean_sequence(seq)
    seq_len = len(cleaned_seq)
    
    # 生成4-mer字典（与train.py一致）
    bases = ['A', 'T', 'C', 'G']
    kmers = [''.join(p) for p in product(bases, repeat=K_MER)]
    kmer2idx = {k: i for i, k in enumerate(kmers)}
    feature = np.zeros(len(kmers), dtype=np.float32)
    
    # 统计k-mer频率（与train.py一致）
    if seq_len >= K_MER:
        for i in range(seq_len - K_MER + 1):
            kmer = cleaned_seq[i:i+K_MER]
            if kmer in kmer2idx:
                feature[kmer2idx[kmer]] += 1
        # 归一化（按序列长度，与train.py一致）
        feature /= seq_len
    
    return feature.reshape(1, -1)  # (1, 256)

def get_pred_result(model_name: str, feature: np.ndarray) -> Tuple[Optional[str], float]:
    """获取单条序列的预测标签和置信度（核心修复：兼容空编码器）"""
    config = MODEL_CONFIG.get(model_name)
    if not config or "model" not in config:
        return None, 0.0
    
    model = config["model"]
    le = config["label_encoder"]
    
    try:
        # 预测概率和标签
        prob = model.predict_proba(feature)[0]
        conf = float(np.max(prob))
        pred_idx = np.argmax(prob)
        
        # 核心修复：兼容编码器无classes_的情况
        if hasattr(le, 'classes_') and len(le.classes_) > 0:
            pred_label = le.inverse_transform([pred_idx])[0]
        else:
            # 直接返回索引对应的类别（从层级映射中匹配）
            if model_name in LEVEL_MAPPING:
                classes = list(LEVEL_MAPPING[model_name].keys())
                pred_label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
            else:
                pred_label = str(pred_idx)
        
        return pred_label, conf
    except Exception as e:
        print(f"⚠️ 模型 {model_name} 预测失败: {str(e)}")
        return None, 0.0

# ===================== 数据加载 =====================
def load_test_dataset() -> List[Dict]:
    """加载测试集数据（仅train_test.txt中为0的样本）"""
    print("\n开始加载测试集数据...")
    
    # 读取train_test标记（0=测试集，1=训练集）
    with open(TRAIN_TEST_PATH, 'r', encoding='utf-8') as f:
        train_test_flags = [int(line.strip()) for line in f if line.strip()]
    
    # 读取对齐数据
    test_samples = []
    with open(ALIGNED_DATA_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # 仅处理测试集样本
            if idx >= len(train_test_flags) or train_test_flags[idx] != 0:
                continue
            
            # 解析格式：层数,层1标签,层2标签,层3标签,层4标签,序列
            parts = line.split(',', 5)
            if len(parts) != 6:
                print(f"⚠️ 第{idx+1}行格式错误，跳过: {line}")
                continue
            
            # 解析字段
            n_levels = int(parts[0]) if parts[0].isdigit() else 0
            true_labels = {
                "class": parts[1].strip(),
                "classI": parts[2].strip() if parts[2] != "invalid" else None,
                "classII_sub1": parts[2].strip() if parts[2] != "invalid" else None,
                "LTR": parts[3].strip() if parts[3] != "invalid" else None,
                "nLTR": parts[3].strip() if parts[3] != "invalid" else None,
                "SINE": parts[4].strip() if parts[4] != "invalid" else None,
                "LINE": parts[4].strip() if parts[4] != "invalid" else None
            }
            sequence = parts[5].strip()
            
            test_samples.append({
                "seq_id": idx,
                "n_levels": n_levels,
                "true_labels": true_labels,
                "sequence": sequence
            })
    
    print(f"✅ 测试集加载完成，共 {len(test_samples)} 条样本")
    return test_samples

# ===================== 级联测试核心逻辑 =====================
def run_cascade_test(test_samples: List[Dict]) -> Dict[str, Dict]:
    """执行级联测试，返回各模型的结果统计（扩展：保留原始预测和置信度）"""
    print("\n开始执行级联测试（逐条序列，batch_size=1）...")
    
    # 初始化结果统计容器（扩展：保存原始预测标签、置信度）
    result_stats = {
        model_name: {
            "sample_count": 0,
            "true_labels": [],  # 真实标签（无-1）
            "pred_labels": [],  # 最终预测标签（-1=失败）
            "raw_pred_labels": [],  # 原始预测标签（未处理-1）
            "confidences": [],  # 预测置信度
            "valid_classes": []  # 有效类别（用于混淆矩阵）
        } for model_name in MODEL_CONFIG.keys()
    }
    
    # 逐条处理测试样本
    for sample in test_samples:
        seq_id = sample["seq_id"]
        sequence = sample["sequence"]
        true_labels = sample["true_labels"]
        fail_flag = False  # 是否分类失败
        
        # 提取序列特征（仅提取一次）
        feature = extract_single_seq_feature(sequence)
        
        # 从根模型开始级联测试
        current_model = "class"
        while current_model and not fail_flag:
            # 获取当前模型的真实标签
            true_label = true_labels.get(current_model)
            if not true_label or true_label == "invalid":
                current_model = None
                continue
            
            # 模型预测：保留原始结果
            raw_pred_label, confidence = get_pred_result(current_model, feature)
            
            # 失败判断：置信度不足 或 预测错误
            final_pred_label = raw_pred_label if (confidence >= CONFIDENCE_THRESHOLD and raw_pred_label == true_label) else "-1"
            if final_pred_label == "-1":
                fail_flag = True
            
            # 统计结果
            stats = result_stats[current_model]
            stats["sample_count"] += 1
            stats["true_labels"].append(true_label)
            stats["pred_labels"].append(final_pred_label)
            stats["raw_pred_labels"].append(raw_pred_label)
            stats["confidences"].append(confidence)
            stats["valid_classes"].append(true_label)
            
            # 获取下一级模型（失败则终止）
            if not fail_flag:
                current_model = LEVEL_MAPPING[current_model].get(raw_pred_label)
            else:
                # 失败后，后续模型全部标记为-1
                next_model = LEVEL_MAPPING[current_model].get(true_label)
                while next_model:
                    if next_model in result_stats:
                        next_stats = result_stats[next_model]
                        next_true_label = true_labels.get(next_model) or "invalid"
                        if next_true_label != "invalid":
                            next_stats["sample_count"] += 1
                            next_stats["true_labels"].append(next_true_label)
                            next_stats["pred_labels"].append("-1")
                            next_stats["raw_pred_labels"].append(None)  # 无原始预测
                            next_stats["confidences"].append(0.0)  # 置信度0
                            next_stats["valid_classes"].append(next_true_label)
                    next_model = LEVEL_MAPPING.get(next_model, {}).get(true_labels.get(next_model))
                current_model = None
    
    # 去重有效类别
    for model_name in result_stats:
        result_stats[model_name]["valid_classes"] = sorted(list(set(result_stats[model_name]["valid_classes"])))
    
    print("✅ 级联测试执行完成")
    return result_stats

# ===================== 新增：按需求计算类别级TP/TN/FP/FN =====================
def calculate_class_metrics(
    true_labels: List[str],
    raw_pred_labels: List[str],
    confidences: List[float],
    valid_classes: List[str]
) -> Dict[str, Dict]:
    """
    按需求统计每个类别的TP/TN/FP/FN + Precision/Recall/F1：
    - TP: 真实=A 且 预测=A 且 置信度≥阈值
    - FN: 真实=A 且 (置信度<阈值 或 预测≠A)
    - FP: 真实≠A 且 置信度≥阈值 且 预测=A
    - TN: 真实≠A 且 (置信度<阈值 或 预测≠A)
    """
    class_metrics = {cls: {"TP":0, "TN":0, "FP":0, "FN":0, "Precision":0.0, "Recall":0.0, "F1":0.0} for cls in valid_classes}
    
    # 遍历所有样本计算基础指标
    for t, p, c in zip(true_labels, raw_pred_labels, confidences):
        if t not in valid_classes or p is None:
            continue
        
        for cls in valid_classes:
            # TP: 真实=cls + 预测=cls + 置信度≥阈值
            if t == cls and p == cls and c >= CONFIDENCE_THRESHOLD:
                class_metrics[cls]["TP"] += 1
            # FN: 真实=cls + (置信度<阈值 或 预测≠cls)
            elif t == cls and (c < CONFIDENCE_THRESHOLD or p != cls):
                class_metrics[cls]["FN"] += 1
            # FP: 真实≠cls + 置信度≥阈值 + 预测=cls
            elif t != cls and c >= CONFIDENCE_THRESHOLD and p == cls:
                class_metrics[cls]["FP"] += 1
            # TN: 真实≠cls + (置信度<阈值 或 预测≠cls)
            elif t != cls and (c < CONFIDENCE_THRESHOLD or p != cls):
                class_metrics[cls]["TN"] += 1
    
    # 计算每个类别的Precision/Recall/F1
    for cls in valid_classes:
        tp = class_metrics[cls]["TP"]
        fp = class_metrics[cls]["FP"]
        fn = class_metrics[cls]["FN"]
        
        # Precision = TP/(TP+FP)（避免除0）
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall = TP/(TP+FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F1 = 2*P*R/(P+R)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[cls]["Precision"] = precision
        class_metrics[cls]["Recall"] = recall
        class_metrics[cls]["F1"] = f1
    
    return class_metrics

# ===================== 原有指标计算（保留） =====================
def calculate_metrics(true_list: List[str], pred_list: List[str], valid_classes: List[str]) -> Dict:
    """计算指标：处理-1，混淆矩阵排除-1"""
    # 1. 计算整体指标（包含-1，视为错误）
    # 编码标签：有效类别=1-N，-1=0
    label2idx = {cls: i+1 for i, cls in enumerate(valid_classes)}
    label2idx["-1"] = 0
    
    encoded_true = [label2idx[t] for t in true_list]
    encoded_pred = [label2idx[p] if p in label2idx else 0 for p in pred_list]
    
    accuracy = accuracy_score(encoded_true, encoded_pred)
    macro_f1 = f1_score(encoded_true, encoded_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(encoded_true, encoded_pred, average='micro', zero_division=0)
    
    # 2. 计算混淆矩阵（排除-1）
    cm_true = [t for t, p in zip(true_list, pred_list) if p != "-1"]
    cm_pred = [p for t, p in zip(true_list, pred_list) if p != "-1"]
    conf_matrix = confusion_matrix(cm_true, cm_pred, labels=valid_classes) if cm_true else np.array([])
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "confusion_matrix": conf_matrix,
        "valid_classes": valid_classes
    }

# ===================== 结果输出（扩展：新增类别级指标打印） =====================
def print_final_results(result_stats: Dict):
    """打印最终测试结果（扩展：新增类别级TP/TN/FP/FN + Precision/Recall/F1）"""
    print("\n" + "="*100)
    print(" 级联测试最终结果汇总")
    print("="*100)
    
    for model_name, stats in result_stats.items():
        sample_count = stats["sample_count"]
        if sample_count == 0:
            print(f"\n【{model_name.upper()} 模型】")
            print(f"测试样本数：0")
            print(f"无有效测试数据")
            print("-"*80)
            continue
        
        # 原有逻辑：计算整体指标
        metrics = calculate_metrics(
            stats["true_labels"],
            stats["pred_labels"],
            stats["valid_classes"]
        )
        
        # 新增逻辑：计算类别级详细指标
        class_metrics = calculate_class_metrics(
            stats["true_labels"],
            stats["raw_pred_labels"],
            stats["confidences"],
            stats["valid_classes"]
        )
        
        # 打印基础信息
        print(f"\n【{model_name.upper()} 模型】")
        print(f"测试样本数：{sample_count}")
        print(f"总准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"宏平均F1 (Macro F1): {metrics['macro_f1']:.4f}")
        print(f"微平均F1 (Micro F1): {metrics['micro_f1']:.4f}")
        
        # 新增：打印每个类别的详细指标
        print(f"\n  每个类别的详细指标（按需求统计）：")
        print(f"  类别              TP       TN       FP       FN       Precision    Recall     F1")
        print(f"  --------------------------------------------------------------------------------")
        for cls in stats["valid_classes"]:
            cm = class_metrics[cls]
            print(f"  {cls:<15} {cm['TP']:<8} {cm['TN']:<8} {cm['FP']:<8} {cm['FN']:<8} "
                  f"{cm['Precision']:<12.4f} {cm['Recall']:<10.4f} {cm['F1']:<10.4f}")
        print(f"  --------------------------------------------------------------------------------")
        
        # 打印混淆矩阵
        print(f"\n混淆矩阵（排除-1）：")
        if len(metrics["confusion_matrix"]) > 0:
            cm_df = pd.DataFrame(
                metrics["confusion_matrix"],
                index=[f"真实-{cls}" for cls in metrics["valid_classes"]],
                columns=[f"预测-{cls}" for cls in metrics["valid_classes"]]
            )
            print(cm_df.to_string())
        else:
            print("（所有预测结果均为-1，无有效混淆矩阵）")
        
        print("-"*80)

# ===================== 主函数 =====================
def main():
    # 1. 加载模型
    if not load_all_models():
        print("❌ 模型加载失败，终止测试")
        return
    
    # 2. 加载测试集
    test_samples = load_test_dataset()
    if not test_samples:
        print("❌ 无测试集数据，终止测试")
        return
    
    # 3. 执行级联测试
    result_stats = run_cascade_test(test_samples)
    
    # 4. 打印结果
    print_final_results(result_stats)
    
    print("\n 级联测试全部完成！")

if __name__ == "__main__":
    main()