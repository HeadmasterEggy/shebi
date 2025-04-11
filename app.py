import logging
import os
import re

import jieba
import torch
import pandas as pd
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnn_model import TextCNN
from config import Config
from data_Process import build_word2id, build_word2vec, build_id2word, prepare_data, text_to_array_nolabel, Data_set
from eval import CacheManager  # 导入 CacheManager
from lstm_model import LSTM_attention
from data_Process import tokenize, clean_text, process_texts
# 从eval模块导入预测函数

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 预加载jieba词典，提高分词速度
jieba.initialize()
logger.info("Jieba分词器初始化完成")

app = Flask(__name__, static_folder='static')
CORS(app)  # 启用CORS

# 添加安全全局类
torch.serialization.add_safe_globals([
    nn.Embedding,
    nn.LSTM,
    nn.Linear,
    nn.Dropout,
    nn.Sequential,
    nn.Module,
    LSTM_attention,
    TextCNN
])

def pre(word2id, model, seq_length, path):
    """
    给定文本，预测其情感标签。

    参数:
        word2id (dict): 语料文本中包含的词汇字典（词和ID映射关系）。
        model (Object): 情感分析模型（Seq2Seq）。
        seq_length (int): 序列的固定长度。
        path (str): 数据文件路径。

    返回:
        dict: 包含预测结果和详细信息的字典。
    """
    model.cpu()  # 确保模型使用CPU

    # 读取文件中的文本
    with open(path, "r", encoding="utf-8") as file:
        texts = file.readlines()

    predictions = []  # 用于存储预测的标签
    probabilities = []  # 用于存储预测的概率
    word_freq = {}  # 用于存储词频统计
    sentence_results = []  # 用于存储每个句子的分析结果

    with torch.no_grad():  # 禁用梯度计算
        # 将文本转换为索引数字
        input_array = text_to_array_nolabel(word2id, seq_length, path)
        sen_p = torch.tensor(input_array, dtype=torch.long)

        # 获取模型预测的输出
        output_p = model(sen_p)

        # 获取概率分布
        probs = torch.softmax(output_p, dim=1)

        # 获取每个文本的预测类别
        _, pred = torch.max(output_p, 1)

        for i in range(pred.size(0)):
            prediction = pred[i].item()
            prob = probs[i].tolist()
            predictions.append(prediction)
            probabilities.append(prob)

            # 分词并统计词频
            text = texts[i].strip()
            if text:  # 只处理非空文本
                words = jieba.lcut(text)
                for word in words:
                    if len(word) > 1:  # 只统计长度大于1的词
                        word_freq[word] = word_freq.get(word, 0) + 1

                # 在终端输出预测结果
                sentiment = "积极" if prediction == 1 else "消极"
                confidence = max(prob) * 100
                pos_prob = prob[1] * 100
                neg_prob = prob[0] * 100

                logger.info("-" * 50)
                logger.info(f"输入文本: {text}")
                logger.info(f"情感倾向: {sentiment}")
                logger.info(f"预测置信度: {confidence:.2f}%")
                logger.info(f"积极概率: {pos_prob:.2f}%")
                logger.info(f"消极概率: {neg_prob:.2f}%")

                # 保存每个句子的分析结果
                sentence_results.append({
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "probabilities": {
                        "positive": pos_prob,
                        "negative": neg_prob
                    }
                })

    # 计算模型评估指标
    # 读取评估指标日志
    metrics_df = pd.read_csv('metrics_log.csv')
    # 获取最新的测试评估指标
    latest_metrics = metrics_df[metrics_df['type'] == 'validation'].iloc[-1]
    model_metrics = {
        "accuracy": latest_metrics['accuracy'] / 100,  # 转换为小数
        "f1_score": latest_metrics['f1'] / 100,
        "recall": latest_metrics['recall'] / 100
    }

    # 将词频统计转换为列表格式
    word_freq_list = [{"word": word, "count": count} for word, count in word_freq.items()]
    word_freq_list.sort(key=lambda x: x["count"], reverse=True)  # 按词频降序排序

    # 输出高频词统计
    logger.info("-" * 50)
    logger.info("高频词统计 (Top 10):")
    for item in word_freq_list[:10]:
        logger.info(f"词语: {item['word']}, 出现次数: {item['count']}")
    logger.info("-" * 50)

    # 计算整体情感倾向
    total_pos_prob = sum(prob[1] for prob in probabilities) / len(probabilities) * 100
    total_neg_prob = sum(prob[0] for prob in probabilities) / len(probabilities) * 100
    overall_sentiment = "积极" if total_pos_prob > total_neg_prob else "消极"
    overall_confidence = max(total_pos_prob, total_neg_prob)

    return {
        "overall": {
            "sentiment": overall_sentiment,
            "confidence": overall_confidence,
            "probabilities": {
                "positive": total_pos_prob,
                "negative": total_neg_prob
            }
        },
        "sentences": sentence_results,  # 每个句子的分析结果
        "modelMetrics": model_metrics,
        "wordFreq": word_freq_list[:20]  # 只返回前20个高频词
    }


def initialize_data():
    """
    初始化数据、字典和模型。
    """
    # 定义缓存参数
    cache_params = {
        "word2id_path": Config.word2id_path,
        "train_path": Config.train_path,
        "val_path": Config.val_path,
        "test_path": Config.test_path,
        "seq_length": Config.max_sen_len
    }

    # 尝试从缓存加载数据
    logger.info(f"尝试加载数据缓存，参数: {cache_params}")
    cached_data = cache_manager.load("processed_data", cache_params)
    if cached_data is not None:
        logger.info("成功从缓存加载预处理数据")
        return (cached_data["word2id"], cached_data["test_dataloader"],
                cached_data["val_dataloader"], cached_data["train_array"],
                cached_data["train_label"])

    logging.info("初始化数据...")
    word2id = build_word2id(Config.word2id_path)
    id2word = build_id2word(word2id)

    # 准备训练、验证和测试数据
    train_array, train_label, val_array, val_label, test_array, test_label = prepare_data(
        word2id,
        train_path=Config.train_path,
        val_path=Config.val_path,
        test_path=Config.test_path,
        seq_length=Config.max_sen_len,
    )

    # 创建 DataLoader
    test_loader = Data_set(test_array, test_label)
    test_dataloader = DataLoader(test_loader, batch_size=Config.lstm_batch_size, shuffle=True, num_workers=0)

    val_loader = Data_set(val_array, val_label)
    val_dataloader = DataLoader(val_loader, batch_size=Config.lstm_batch_size, shuffle=True, num_workers=0)

    # 保存到缓存
    processed_data = {
        "word2id": word2id,
        "test_dataloader": test_dataloader,
        "val_dataloader": val_dataloader,
        "train_array": train_array,
        "train_label": train_label
    }
    cache_manager.save(processed_data, "processed_data", cache_params)

    return word2id, test_dataloader, val_dataloader, train_array, train_label


def initialize_model(w2vec, model_type=None):
    """
    初始化模型并加载最优模型或初始模型。

    参数:
        w2vec: 词向量
        model_type: 模型类型，可以是'lstm'或'cnn'，默认为None，使用Config.model_name
    """
    # 如果未指定模型类型，则使用配置中的默认值
    model_name = model_type.upper() if model_type else Config.model_name

    # 检查是否有缓存的模型状态
    model_cache_params = {
        "model_name": model_name,
        "vocab_size": Config.vocab_size,
        "embedding_dim": Config.embedding_dim,
        "hidden_dim": Config.hidden_dim,
        "num_layers": Config.num_layers
    }

    bilstm_model = LSTM_attention(
        Config.vocab_size,
        Config.embedding_dim,
        w2vec,
        Config.update_w2v,
        Config.hidden_dim,
        Config.num_layers,
        Config.drop_keep_prob,
        Config.n_class,
        Config.bidirectional,
    )

    cnn_model = TextCNN(
        Config.dropout,
        Config.require_improvement,
        Config.vocab_size,
        Config.cnn_batch_size,
        Config.pad_size,
        Config.filter_sizes,
        Config.num_filters,
        w2vec,  # 修正：传入预训练词向量w2vec而不是embedding_dim
        Config.embedding_dim,
        Config.n_class,
    )
    model = bilstm_model if model_name == "lstm" else cnn_model

    # 选择适合的模型路径
    best_model_path = Config.lstm_best_model_path if model_name == "lstm" else Config.cnn_best_model_path

    logger.info(f"使用 {model_name} 模型")

    # 尝试加载缓存的模型状态
    cache_key = f"{model_name.lower()}_model_state"
    logger.info(f"尝试加载模型缓存 {cache_key}，参数: {model_cache_params}")
    cached_state = cache_manager.load(cache_key, model_cache_params)
    if cached_state is not None:
        logger.info(f"成功从缓存加载 {model_name} 模型状态")
        model.load_state_dict(cached_state)
        return model

    logging.info(f"初始化 {model_name} 模型...")
    # 加载最佳模型
    if os.path.exists(best_model_path):
        model = torch.load(best_model_path, weights_only=False)
    else:
        logging.warning(f"找不到 {model_name} 最佳模型，请确保模型文件存在")

    # 保存模型状态到缓存
    cache_manager.save(model.state_dict(), cache_key, model_cache_params)

    model.eval()  # 设置为评估模式
    return model

# 创建全局缓存管理器实例
cache_manager = CacheManager(cache_dir="./cache")
logger.info("缓存管理器初始化完成，缓存目录: ./cache")
# 读取停用词
stopwords = []
with open("data/stopword.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        stopwords.append(line.strip())

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# 添加新的API端点，获取可用的模型列表
@app.route('/api/models', methods=['GET'])
def get_models():
    """
    获取可用的模型列表
    """
    models = [
        {
            "id": "lstm",
            "name": "Bi-LSTM模型",
            "description": "基于LSTM的情感分析模型，适合处理长文本和序列依赖性强的文本"
        },
        {
            "id": "cnn",
            "name": "TextCNN模型",
            "description": "基于CNN的情感分析模型，适合处理短文本和特征提取"
        }
    ]

    # 检查当前默认模型
    default_model = Config.model_name.lower()

    return jsonify({
        "models": models,
        "default": default_model
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        data = request.get_json()
        text = data.get('text', '')
        model_type = data.get('model_type', '').lower()  # 获取模型类型参数
        if not text:
            return jsonify({'error': '请输入要分析的文本'}), 400

        # 验证模型类型
        if model_type and model_type not in ['lstm', 'cnn']:
            return jsonify({'error': '不支持的模型类型，请选择 lstm 或 cnn'}), 400

        logger.info("=" * 80)
        logger.info("收到新的分析请求")
        logger.info(f"设备: {device}")
        logger.info(f"模型类型: {model_type if model_type else '默认'}")

        # 按回车分割句子
        original_sentences = [s.strip() for s in text.split('\n') if s.strip()]

        # 处理每个句子
        processed_sentences = []
        for sentence in original_sentences:
            # 清洗文本
            cleaned_text = clean_text(sentence)
            # 分词处理
            tokenized_text = tokenize(cleaned_text, stopwords)
            if tokenized_text.strip():  # 只添加非空句子
                processed_sentences.append(tokenized_text)

        # 将处理后的句子写入预测文件
        with open(Config.pre_path, 'w', encoding='utf-8') as file:
            for sentence in processed_sentences:
                file.write(sentence + '\n')

        # 初始化数据（使用缓存）
        word2id, test_dataloader, val_dataloader, train_array, train_label = initialize_data()

        # 生成或加载 word2vec（使用缓存）
        w2vec_cache_params = {
            "pre_word2vec_path": Config.pre_word2vec_path,
            "word2id_size": len(word2id),  # 添加word2id的大小作为参数，确保word2id变化时缓存也会更新
            "model_name": model_type.upper() if model_type else Config.model_name  # 添加模型名称，确保不同模型使用相同的word2vec缓存
        }
        logger.info(f"尝试加载word2vec缓存，参数: {w2vec_cache_params}")
        w2vec = cache_manager.load("w2vec", w2vec_cache_params)
        if w2vec is None:
            logger.info("未找到word2vec缓存，正在生成...")
            w2vec = build_word2vec(Config.pre_word2vec_path, word2id, None)
            w2vec = torch.from_numpy(w2vec).float()
            logger.info("保存word2vec到缓存")
            cache_manager.save(w2vec, "w2vec", w2vec_cache_params)
        else:
            logger.info("成功从缓存加载word2vec")

        # 初始化模型（使用缓存），传入模型类型
        model = initialize_model(w2vec, model_type)

        logger.info("开始进行情感分析...")
        # 获取预测结果
        result = pre(word2id, model, Config.max_sen_len, Config.pre_path)

        # 更新结果中的文本为原始句子
        for i, sentence_result in enumerate(result['sentences']):
            if i < len(original_sentences):  # 确保索引有效
                sentence_result['text'] = original_sentences[i]

        # 添加使用的模型信息到结果中
        used_model = model_type.upper() if model_type else Config.model_name
        result['modelInfo'] = {
            'type': used_model
        }

        logger.info("分析完成")
        logger.info("=" * 80)

        return jsonify(result), 200

    except Exception as e:
        error_msg = f'分析时出错: {str(e)}'
        logger.error(error_msg)
        logger.error("=" * 80)
        return jsonify({'error': error_msg}), 500


if __name__ == '__main__':
    app.run(debug=False, port=5003)
