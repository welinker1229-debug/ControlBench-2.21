import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from datasets_edge import load_edge_data
from models.edge_classifier import EdgeClassifier

# --- 配置 ---
EPOCHS = 35
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
ALL_TOPICS = ['lgbtq', 'abortion', 'capitalism', 'trump', 'religion']
LABEL_ORDER = ["Oppose", "Neutral", "Support"]


def train_single_topic(topic):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 60}")
    print(f"🚀 启动任务: {topic.upper()} (训练 + 导出 + 绘图)")
    print(f"{'=' * 60}")

    # 1. 加载数据
    try:
        g = load_edge_data(topic)
    except Exception as e:
        print(f"❌ {topic} 数据加载错误: {e}")
        return

    features = g.edges['interacts'].data['feat']
    labels = g.edges['interacts'].data['label']

    input_dim = features.shape[1]

    # 2. 计算 Loss 权重 (逻辑不变)
    labels_np = labels.numpy()
    class_counts = np.bincount(labels_np)
    if len(class_counts) < 3: class_counts = np.pad(class_counts, (0, 3 - len(class_counts)), 'constant')

    print(f"📊 样本分布: 反对({class_counts[0]}) | 中立({class_counts[1]}) | 支持({class_counts[2]})")

    # 反比权重
    weights = [sum(class_counts) / c if c > 0 else 1.0 for c in class_counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"⚖️ Loss权重: {weights}")

    # 3. 数据集
    dataset = TensorDataset(features, labels)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. 模型
    model = EdgeClassifier(input_dim, hidden_dim=256, num_classes=3).to(device)

    # 5. 训练循环
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    best_f1 = 0.0
    print("🔄 Training...")

    for epoch in range(EPOCHS):
        model.train()
        all_preds, all_true = [], []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_true.extend(batch_y.cpu().numpy())

        epoch_f1 = f1_score(all_true, all_preds, average='macro')

        if (epoch + 1) % 5 == 0:
            print(f"   [Epoch {epoch + 1}] F1: {epoch_f1:.4f}")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            torch.save(model.state_dict(), f"saved_model_{topic}.pth")

    print(f"✅ {topic.upper()} 最佳 F1: {best_f1:.4f}")

    # ==========================================
    # 📝 6. 推理 & 生成结果
    # ==========================================
    print(f"📊 正在生成分析报告...")

    # 加载最佳模型
    model.load_state_dict(torch.load(f"saved_model_{topic}.pth", map_location=device))
    model.eval()

    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    results = []
    y_true_labels, y_pred_labels, y_confs = [], [], []
    label_map_rev = {0: "Oppose", 1: "Neutral", 2: "Support"}

    idx = 0
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            trues = batch_y.numpy()

            for i in range(len(preds)):
                t_label = label_map_rev.get(trues[i], "Unknown")
                p_label = label_map_rev.get(preds[i], "Unknown")
                conf = float(probs[i][preds[i]])

                y_true_labels.append(t_label)
                y_pred_labels.append(p_label)
                y_confs.append(conf)

                results.append({
                    "Edge_Index": idx,
                    "True_Label": t_label,
                    "Predicted_Label": p_label,
                    "Correct": bool(trues[i] == preds[i]),
                    "Confidence": conf,
                    "Probabilities": {
                        "Oppose": float(probs[i][0]),
                        "Neutral": float(probs[i][1]),
                        "Support": float(probs[i][2])
                    }
                })
                idx += 1

    # 保存 JSON
    json_filename = f"prediction_results_{topic}.json"
    with open(json_filename, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"   📄 JSON 已保存: {json_filename}")

    # 绘制图表
    print(f"   🎨 正在绘图...")
    acc = accuracy_score(y_true_labels, y_pred_labels)
    final_f1 = f1_score(y_true_labels, y_pred_labels, average='macro')

    fig = plt.figure(figsize=(18, 10))
    plt.suptitle(f"Model Evaluation: {topic.upper()}\nAccuracy: {acc:.2%} | Macro F1: {final_f1:.4f}", fontsize=16,
                 fontweight='bold')

    # 1. 混淆矩阵
    ax1 = plt.subplot(2, 2, 1)
    label_to_idx = {l: i for i, l in enumerate(LABEL_ORDER)}
    y_true_idx = [label_to_idx[l] for l in y_true_labels]
    y_pred_idx = [label_to_idx[l] for l in y_pred_labels]

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=[0, 1, 2])
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax1, cbar=False)
    ax1.set_title('Confusion Matrix (Normalized)', fontsize=14)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    # 2. 分布对比
    ax2 = plt.subplot(2, 2, 2)
    labels, counts_true = np.unique(y_true_labels, return_counts=True)
    _, counts_pred = np.unique(y_pred_labels, return_counts=True)

    def get_counts(labels_found, counts_found):
        mapping = dict(zip(labels_found, counts_found))
        return [mapping.get(l, 0) for l in LABEL_ORDER]

    c_true = get_counts(labels, counts_true)
    c_pred = get_counts(_, counts_pred)

    x = np.arange(len(LABEL_ORDER))
    width = 0.35
    ax2.bar(x - width / 2, c_true, width, label='True Labels', color='#4c72b0', alpha=0.8)
    ax2.bar(x + width / 2, c_pred, width, label='Predicted', color='#dd8452', alpha=0.8)
    ax2.set_title('Label Distribution', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(LABEL_ORDER)
    ax2.legend()

    # 3. 置信度
    ax3 = plt.subplot(2, 1, 2)
    correct_conf = [c for t, p, c in zip(y_true_labels, y_pred_labels, y_confs) if t == p]
    wrong_conf = [c for t, p, c in zip(y_true_labels, y_pred_labels, y_confs) if t != p]

    if correct_conf: ax3.hist(correct_conf, bins=50, range=(0, 1), color='green', alpha=0.5, label='Correct')
    if wrong_conf: ax3.hist(wrong_conf, bins=50, range=(0, 1), color='red', alpha=0.5, label='Wrong')

    ax3.set_title('Prediction Confidence Distribution', fontsize=14)
    ax3.set_xlabel('Confidence')
    ax3.legend()

    img_filename = f"evaluation_report_{topic}.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(img_filename)
    plt.close(fig)  # 释放内存
    print(f"   🖼️ 图表已保存: {img_filename}")


def main():
    print(f"⏰ 开始全自动批量处理: {ALL_TOPICS}")
    start_time = time.time()

    for topic in ALL_TOPICS:
        train_single_topic(topic)

    total_min = (time.time() - start_time) / 60
    print(f"\n🎉 所有任务全部完成！总耗时: {total_min:.2f} 分钟")


if __name__ == "__main__":
    main()