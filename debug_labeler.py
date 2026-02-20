import torch
from transformers import pipeline
import time


def main():
    print("=" * 60)
    print("🕵️‍♂️ Zero-Shot 立场检测调试器 ")
    print("   -> 正在加载模型 (valhalla/distilbart-mnli-12-3)...")
    print("=" * 60)

    device = 0 if torch.cuda.is_available() else -1
    try:
        # 使用 distilbart-mnli，速度快且逻辑能力够用
        classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=device)
    except:
        classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=-1)

    # 🔥 核心升级：语义扩容
    # 1. 不再纠结具体议题(如婚姻)，而是上升到"Rights"(权益)或"Ideology"(意识形态)。
    # 2. 中立标签特定化，防止它吸走通用句。
    topic_labels = {
        "lgbtq": [
            "anti-LGBTQ rights",  # 反对
            "neutral regarding LGBTQ",  # 中立 (特指对此话题中立)
            "pro-LGBTQ rights"  # 支持 (覆盖面更广，包含Love is love)
        ],
        "abortion": [
            "pro-life (anti-abortion)",  # 反堕胎 (Pro-life是强语义词)
            "neutral regarding abortion",
            "pro-choice (supporting abortion access)"  # 支持堕胎 (Pro-choice是强语义词)
        ],
        "capitalism": [
            "anti-capitalism",
            "neutral regarding economics",
            "pro-capitalism"
        ],
        "trump": [
            "anti-Trump",
            "neutral regarding Trump",
            "pro-Trump"
        ],
        "religion": [
            "atheist or anti-religion",  # 无神论/反宗教
            "neutral regarding religion",
            "religious or pro-faith"  # 有信仰/支持宗教
        ]
    }

    print("\n✅ 老师已就位！")

    while True:
        topic = input("\n👉 请选择话题 (lgbtq/abortion/trump...): ").strip().lower()
        if topic == 'exit': break

        if topic not in topic_labels:
            print("   (默认使用 lgbtq 标签)")
            topic = "lgbtq"

        labels = topic_labels[topic]
        # 映射显示的中文
        label_map = {labels[0]: "🔴 反对", labels[1]: "⚪ 中立", labels[2]: "🟢 支持"}

        print(f"   🎯 逻辑探针: {labels}")

        text = input("📝 输入测试文本 (English): ").strip()
        if not text: continue
        if text == 'exit': break

        # 预测
        start = time.time()
        # hypothesis_template 默认是 "This example is {}."，对于 stance 任务通常够用
        # 也可以尝试 "The stance of this text is {}."
        result = classifier(text, labels, multi_label=False)
        end = time.time()

        print(f"\n📊 判决结果 (耗时 {end - start:.2f}s):")

        scores = dict(zip(result['labels'], result['scores']))

        # 打印条形图
        for lbl in labels:  # 按反对/中立/支持的顺序打印
            score = scores.get(lbl, 0.0)
            bar = "#" * int(score * 20)
            print(f"   {label_map[lbl]:<5} [{bar:<20}] {score:.4f}  <-- {lbl}")

        top_label = result['labels'][0]
        print(f"\n💡 最终结论: {label_map[top_label]}")


if __name__ == "__main__":
    main()