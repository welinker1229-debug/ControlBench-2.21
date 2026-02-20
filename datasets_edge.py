import torch
import dgl
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from tqdm import tqdm

# ==========================================
# 🔧 全局配置 (学术规范版)
# ==========================================
CONFIG = {
    'bart_model': 'facebook/bart-large-mnli',
    'qwen_model': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    'device': 0 if torch.cuda.is_available() else -1,
    'batch_size': 16,

    # 🌟 核心策略: 置信度增强 (Confidence Boosting)
    # 替代硬规则: 如果 BART 极度确信(>0.9)，则视为"铁律"
    'conf_threshold_high': 0.90,
    'weight_boost': 2.0,  # 高置信度样本权重翻倍
    'conf_threshold_low': 0.60,  # 低置信度样本视为噪音
    'weight_noise': 0.1  # 噪音样本给予极低权重
}


def load_edge_data(topic, data_dir='data'):
    print(f"\n{'=' * 70}")
    print(f"🧠 [Dataset] 构建数据集: {topic.upper()}")
    print(f"🔥 策略: 纯数据驱动 (Data-Driven) | 去除人工规则 | 置信度加权")
    print(f"{'=' * 70}")

    # 1. 路径兼容处理
    file_path = os.path.join(data_dir, f'graph_data_{topic}.json')
    if not os.path.exists(file_path):
        possible_path = os.path.join('jzhu1905/controbench/controbench-2bc4d0e4076a80cee47529fd4e3c4e4281ead067/data',
                                     f'graph_data_{topic}.json')
        if os.path.exists(possible_path):
            file_path = possible_path
        else:
            raise FileNotFoundError(f"❌ 找不到文件: {file_path}")

    # 2. 读取数据
    with open(file_path, "r", encoding='utf-8') as f:
        data_json = json.load(f)

    raw_edges = data_json.get('edges', data_json.get('interactions', []))

    # 3. 提取有效交互 & 清洗
    # 建立 Post -> Author 映射
    post_author_map = {e['target']: e['source'] for e in raw_edges if
                       e.get('type') == 'user_publish_post' and 'source' in e}

    valid_interactions = []
    valid_texts = []  # 暂存文本用于批量处理

    print("-> 正在初始化数据并清洗旧标签...")
    for edge in raw_edges:
        src = edge.get('source')
        tgt = edge.get('target')
        etype = edge.get('type')
        content = edge.get('content', "") or edge.get('reply_content', "")

        if not content: continue

        real_target = None
        if etype == 'user_comment_user':
            real_target = tgt
        elif etype == 'user_comment_post' and tgt in post_author_map:
            real_target = post_author_map[tgt]
            if src == real_target: continue

        if src and real_target:
            edge['temp_source_id'] = src
            edge['temp_target_id'] = real_target
            edge['temp_content'] = content

            # 🔥 强制清除旧标签，确保完全由 BART 重新打标
            if 'edge_label' in edge: del edge['edge_label']
            if 'confidence' in edge: del edge['confidence']

            valid_interactions.append(edge)
            valid_texts.append(content[:256])  # 截断防止OOM

    print(f"✅ 待处理数据: {len(valid_interactions)} 条")

    # =================================================
    # 👩‍⚖️ Teacher Model: BART-Large (纯知识蒸馏)
    # =================================================
    print(f"🔨 启动 BART-Large (Teacher)...")
    try:
        classifier = pipeline("zero-shot-classification",
                              model=CONFIG['bart_model'],
                              device=CONFIG['device'])
    except Exception as e:
        print(f"⚠️ GPU加载失败，切换到CPU: {e}")
        classifier = pipeline("zero-shot-classification",
                              model=CONFIG['bart_model'],
                              device=-1)

    # 定义动态标签 (不再需要 check_hard_logic)
    topic_labels_map = {
        "lgbtq": ["anti-lgbtq rights", "neutral statement", "pro-lgbtq rights"],
        "abortion": ["anti-abortion", "neutral statement", "pro-choice"],
        "trump": ["anti-trump", "neutral statement", "pro-trump"],
    }
    # 默认回退
    current_labels = topic_labels_map.get(topic, [f"opposing {topic}", "neutral", f"supporting {topic}"])

    # 标签字符串 -> 数字索引 (0, 1, 2)
    # 注意: BART 输出的 label 顺序是不定的，需要查表
    label_str_to_idx = {
        current_labels[0]: 0,  # Oppose
        current_labels[1]: 1,  # Neutral
        current_labels[2]: 2  # Support
    }

    print(f"-> 正在进行 Knowledge Distillation (Batch Size: {CONFIG['batch_size']})...")

    high_conf_count = 0
    low_conf_count = 0

    for i in tqdm(range(0, len(valid_interactions), CONFIG['batch_size'])):
        batch_texts = valid_texts[i: i + CONFIG['batch_size']]
        batch_edges = valid_interactions[i: i + CONFIG['batch_size']]

        try:
            # Zero-Shot 推理
            results = classifier(batch_texts, current_labels, multi_label=False)
        except:
            # 容错处理
            results = [{'labels': [current_labels[1]], 'scores': [0.5]} for _ in batch_texts]

        for edge, res in zip(batch_edges, results):
            top_label_str = res['labels'][0]
            score = res['scores'][0]

            # 1. 确定标签
            label_idx = label_str_to_idx.get(top_label_str, 1)  # 默认中立
            edge['edge_label'] = label_idx

            # 2. 确定权重 (Confidence Boosting)
            # 这是替代硬规则的关键逻辑
            if score > CONFIG['conf_threshold_high']:
                # 情况 A: Teacher 非常确信 -> 视为"伪金标" -> 权重翻倍
                final_weight = score * CONFIG['weight_boost']
                high_conf_count += 1
            elif score < CONFIG['conf_threshold_low']:
                # 情况 B: Teacher 犹豫不决 -> 视为噪音 -> 权重极低
                final_weight = CONFIG['weight_noise']
                low_conf_count += 1
            else:
                # 情况 C: 普通样本 -> 权重等于置信度
                final_weight = score

            edge['confidence'] = float(final_weight)

    print(f"📊 蒸馏统计: 高置信度强化样本 {high_conf_count} 条 | 低置信度降噪样本 {low_conf_count} 条")

    # 保存处理后的数据 (可选，方便debug)
    # with open(file_path, "w", encoding='utf-8') as f:
    #     json.dump(data_json, f, ensure_ascii=False)

    # =================================================
    # 🚀 Student Model Input: Qwen Embedding
    # =================================================
    print(f"-> 生成 Qwen Embedding (Student Features)...")
    try:
        text_model = SentenceTransformer(CONFIG['qwen_model'], trust_remote_code=True)
    except Exception as e:
        print(f"⚠️ Qwen加载失败，尝试 GTE-Base: {e}")
        text_model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)

    # 构造指令文本
    if topic == 'lgbtq':
        task = "Classify the stance of this text regarding LGBTQ rights."
    elif topic == 'abortion':
        task = "Classify the stance of this text regarding Abortion."
    else:
        task = f"Classify the stance regarding {topic}."

    formatted_texts = [f"Instruct: {task}\nQuery: {t}" for t in valid_texts]

    # 编码
    feat_tensor = text_model.encode(formatted_texts, convert_to_tensor=True, show_progress_bar=True).cpu()

    # =================================================
    # 📦 构建 DGL 图
    # =================================================
    # 映射用户 ID
    all_users = set()
    for edge in valid_interactions:
        all_users.add(edge['temp_source_id'])
        all_users.add(edge['temp_target_id'])

    user_map = {uid: i for i, uid in enumerate(all_users)}
    num_nodes = len(user_map)

    src_ids = [user_map[e['temp_source_id']] for e in valid_interactions]
    dst_ids = [user_map[e['temp_target_id']] for e in valid_interactions]

    # 提取标签和权重
    labels_list = [e['edge_label'] for e in valid_interactions]
    weights_list = [e['confidence'] for e in valid_interactions]

    # 建图
    g = dgl.heterograph({('user', 'interacts', 'user'): (torch.tensor(src_ids), torch.tensor(dst_ids))},
                        num_nodes_dict={'user': num_nodes})

    # 存入数据
    g.edges['interacts'].data['feat'] = feat_tensor
    g.edges['interacts'].data['label'] = torch.tensor(labels_list, dtype=torch.long)
    g.edges['interacts'].data['weight'] = torch.tensor(weights_list, dtype=torch.float)

    print(f"✅ {topic.upper()} 数据集构建完毕! (Edges: {g.num_edges()}, Feat Dim: {feat_tensor.shape[1]})")
    return g