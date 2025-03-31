from gensim.models import KeyedVectors

def check_oov_rate(file_path, word2vec_model):
    total_sentences = 0
    total_words = 0
    total_oov = 0

    print("高OOV句子示例（>50%）：")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            words = parts[1:]  # 跳过 label
            oov_words = [w for w in words if w not in word2vec_model]
            oov_ratio = len(oov_words) / len(words) if words else 0
            total_sentences += 1
            total_words += len(words)
            total_oov += len(oov_words)
            if oov_ratio > 0.5:
                print(f"OOV比率：{oov_ratio:.2f}，句子：{' '.join(words)}")

    avg_oov_ratio = total_oov / total_words if total_words else 0
    print("\n📊 总结：")
    print(f"句子总数：{total_sentences}")
    print(f"总词数：{total_words}")
    print(f"OOV词总数：{total_oov}")
    print(f"平均OOV比例：{avg_oov_ratio:.2%}")


word2vec_model = KeyedVectors.load_word2vec_format('./word2vec/wiki_word2vec_50.bin', binary=True)

# 假设训练集是 train.txt，格式如：1 不错 质量 好开心
check_oov_rate("data/test.txt", word2vec_model)
