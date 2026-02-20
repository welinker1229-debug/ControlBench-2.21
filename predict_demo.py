import torch
import torch.nn.functional as F
import os
from sentence_transformers import SentenceTransformer
from models.edge_classifier import EdgeClassifier

ALL_TOPICS = ['lgbtq', 'abortion', 'capitalism', 'trump', 'religion']
MODEL_DIR = "."
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_demo_model(topic):
    model_path = os.path.join(MODEL_DIR, f"saved_model_{topic}.pth")
    if not os.path.exists(model_path): return None, None, None

    state_dict = torch.load(model_path, map_location=DEVICE)

    # 简单判断维度
    if 'layer1.weight' in state_dict:
        input_dim = state_dict['layer1.weight'].shape[1]
    else:
        input_dim = 1536  # 默认 Qwen

    if input_dim == 1536:
        name = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
        print("-> 加载 Qwen (1.5B)")
    else:
        name = 'Alibaba-NLP/gte-base-en-v1.5'
        print("-> 加载 GTE-Base")

    try:
        text_model = SentenceTransformer(name, trust_remote_code=True)
    except:
        text_model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)

    # 对应极简版模型结构
    classifier = EdgeClassifier(input_dim, hidden_dim=256, num_classes=3).to(DEVICE)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return text_model, classifier, input_dim


def predict(text, text_model, classifier, topic, input_dim):
    # 指令对齐
    if topic == 'lgbtq':
        task = "Classify the stance regarding LGBTQ rights."
    elif topic == 'abortion':
        task = "Classify the stance regarding Abortion."
    else:
        task = f"Classify the stance regarding {topic}."

    formatted_text = f"Instruct: {task}\nQuery: {text}"

    with torch.no_grad():
        emb = text_model.encode([formatted_text], convert_to_tensor=True).to(DEVICE)

    with torch.no_grad():
        logits = classifier(emb)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()

    return pred, probs[0].cpu().numpy()


def main():
    while True:
        choice = input("\n👉 输入话题 (lgbtq...): ").strip().lower()
        if choice in ALL_TOPICS:
            topic = choice
            break

    text_model, classifier, input_dim = load_demo_model(topic)
    label_map = {0: "🔴 反对", 1: "⚪ 中立", 2: "🟢 支持"}
    print(f"\n✨ {topic.upper()} 就绪")

    while True:
        text = input("\n📝 输入: ").strip()
        if text == 'exit': break

        pred, probs = predict(text, text_model, classifier, topic, input_dim)
        print(f"📊 {label_map[pred]} ({probs[pred] * 100:.2f}%)")
        print(f"   Opp: {probs[0]:.2f} | Neu: {probs[1]:.2f} | Sup: {probs[2]:.2f}")


if __name__ == "__main__":
    main()