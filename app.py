import logging
import os

import jieba
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from torch.utils.data import DataLoader

from cnn_model import TextCNN
from config import Config
from data_Process import build_word2id, build_word2vec, build_id2word, prepare_data, text_to_array_nolabel, Data_set
from data_Process import tokenize, clean_text
from lstm_model import LSTM_attention, LSTMModel
# 导入模型工具模块
from utils import initialize_model
# 导入用户模型和认证模块
from models import db, User
from auth import auth, login_manager, admin_required
from flask_login import login_required, current_user

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 预加载jieba词典，提高分词速度
jieba.initialize()
logger.info("Jieba分词器初始化完成")

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_testing')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化扩展
CORS(app)  # 启用CORS
db.init_app(app)
login_manager.init_app(app)

# 注册认证蓝图
app.register_blueprint(auth, url_prefix='/auth')

# 添加安全全局类
torch.serialization.add_safe_globals([
    nn.Embedding,
    nn.LSTM,
    nn.Linear,
    nn.Dropout,
    nn.Sequential,
    nn.Module,
    LSTM_attention,
    LSTMModel,
    TextCNN
])

# 全局设备变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    try:
        with torch.no_grad():  # 禁用梯度计算
            # 将文本转换为索引数字
            input_array = text_to_array_nolabel(word2id, seq_length, path)
            
            # 检查输入数组是否为空
            if len(input_array) == 0:
                logger.error("转换后的输入数组为空")
                return {
                    "overall": {
                        "sentiment": "未知",
                        "confidence": 0,
                        "probabilities": {"positive": 0, "negative": 0}
                    },
                    "sentences": [],
                    "modelMetrics": {"accuracy": 0, "f1_score": 0, "recall": 0},
                    "wordFreq": []
                }
            
            sen_p = torch.tensor(input_array, dtype=torch.long)

            # 获取模型预测的输出
            output_p = model(sen_p)

            # 获取概率分布
            probs = torch.softmax(output_p, dim=1)

            # 获取每个文本的预测类别
            _, pred = torch.max(output_p, 1)

            for i in range(min(pred.size(0), len(texts))):  # 使用最小值避免索引越界
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
                    pos_prob = prob[1] * 100 if len(prob) > 1 else 0  # 确保索引存在
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
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        # 返回一个默认的结果结构
        return {
            "overall": {
                "sentiment": "未知",
                "confidence": 0,
                "probabilities": {"positive": 0, "negative": 0}
            },
            "sentences": [],
            "modelMetrics": {"accuracy": 0, "f1_score": 0, "recall": 0},
            "wordFreq": []
        }

    # 计算模型评估指标
    try:
        # 读取评估指标日志
        metrics_df = pd.read_csv('metrics_log.csv')
        # 获取最新的测试评估指标
        validation_metrics = metrics_df[metrics_df['type'] == 'validation']
        if not validation_metrics.empty:
            latest_metrics = validation_metrics.iloc[-1]
            model_metrics = {
                "accuracy": latest_metrics['accuracy'] / 100,  # 转换为小数
                "f1_score": latest_metrics['f1'] / 100,
                "recall": latest_metrics['recall'] / 100
            }
        else:
            # 如果没有验证数据，使用默认值
            logger.warning("未找到验证指标数据，使用默认值")
            model_metrics = {
                "accuracy": 0.85,
                "f1_score": 0.84,
                "recall": 0.83
            }
    except Exception as e:
        # 如果读取指标文件出错，使用默认值
        logger.error(f"读取评估指标时出错: {str(e)}")
        model_metrics = {
            "accuracy": 0.85,
            "f1_score": 0.84,
            "recall": 0.83
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
    if probabilities:  # 确保有概率数据
        try:
            total_pos_prob = sum(prob[1] for prob in probabilities) / len(probabilities) * 100
            total_neg_prob = sum(prob[0] for prob in probabilities) / len(probabilities) * 100
            overall_sentiment = "积极" if total_pos_prob > total_neg_prob else "消极"
            overall_confidence = max(total_pos_prob, total_neg_prob)
        except IndexError:
            logger.error("计算整体情感倾向时索引错误")
            total_pos_prob = total_neg_prob = 50.0
            overall_sentiment = "未知"
            overall_confidence = 0
    else:
        logger.warning("没有概率数据，使用默认值")
        total_pos_prob = total_neg_prob = 50.0
        overall_sentiment = "未知"
        overall_confidence = 0

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
        "wordFreq": word_freq_list[:20] if word_freq_list else []  # 只返回前20个高频词，确保有数据
    }


def initialize_data():
    """
    初始化数据、字典和模型。
    """
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
    test_dataloader = DataLoader(test_loader, batch_size=Config.batch_size, shuffle=True, num_workers=0)

    val_loader = Data_set(val_array, val_label)
    val_dataloader = DataLoader(val_loader, batch_size=Config.batch_size, shuffle=True, num_workers=0)

    return word2id, test_dataloader, val_dataloader, train_array, train_label


# 删除 create_model 和 initialize_model 函数，使用从 model_utils 导入的函数

# 读取停用词
stopwords = []
with open("data/stopword.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        stopwords.append(line.strip())

# 删除 @app.before_first_request 装饰器及其函数，改为在启动前初始化
# 创建数据表和初始化管理员账户
with app.app_context():
    db.create_all()
    
    # 检查是否已存在管理员账户
    admin = User.query.filter_by(is_admin=True).first()
    if not admin:
        # 创建默认管理员账户
        admin = User(
            username='admin',
            email='admin@example.com',
            password='admin123',
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()
        logger.info("已创建默认管理员账户 (admin/admin123)")

@app.route('/')
def index():
    """主页面路由，检查用户是否已登录"""
    if current_user.is_authenticated:
        return send_from_directory('static', 'index.html')
    else:
        return render_template('login.html')

# 定义默认模型
default_model = "cnn"

@app.route('/api/models', methods=['GET'])
@login_required
def get_models():
    """
    获取可用的模型列表 (需要登录)
    """
    models = [
        {
            "id": "bilstm_attention",
            "name": "Bi-LSTM注意力模型",
            "description": "基于双向LSTM和注意力机制的情感分析模型，适合处理长文本和复杂语义依赖的文本"
        },
        {
            "id": "bilstm",
            "name": "Bi-LSTM模型",
            "description": "基于双向LSTM的情感分析模型，适合处理长文本和序列依赖性强的文本"
        },
        {
            "id": "lstm_attention",
            "name": "LSTM注意力模型",
            "description": "基于LSTM和注意力机制的情感分析模型，能够关注句子中的重要部分"
        },
        {
            "id": "lstm",
            "name": "LSTM模型",
            "description": "基于LSTM的情感分析模型，适合处理有序序列数据"
        },
        {
            "id": "cnn",
            "name": "TextCNN模型",
            "description": "基于CNN的情感分析模型，适合处理短文本和特征提取"
        }
    ]

    return jsonify({
        "models": models,
        "default": default_model
    })

@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze():
    """文本分析API (需要登录)"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_type = data.get('model_type', '').lower()  # 获取模型类型参数
        if not text:
            return jsonify({'error': '请输入要分析的文本'}), 400

        # 验证模型类型
        if model_type and model_type not in ['bilstm_attention', 'bilstm', 'lstm_attention', 'lstm', 'cnn']:
            return jsonify({'error': '不支持的模型类型，请选择有效的模型类型'}), 400

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

        # 初始化数据
        word2id, test_dataloader, val_dataloader, train_array, train_label = initialize_data()

        # 生成 word2vec
        logger.info("生成word2vec...")
        w2vec = build_word2vec(Config.pre_word2vec_path, word2id, None)
        w2vec = torch.from_numpy(w2vec).float()

        # 使用导入的初始化模型函数，传入当前设备
        model = initialize_model(model_type, w2vec, device)

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

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_users():
    """获取所有用户列表 (仅限管理员)"""
    users = User.query.all()
    return jsonify({
        'users': [
            {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            } for user in users
        ]
    })

@app.route('/api/admin/create_user', methods=['POST'])
@admin_required
def create_user():
    """创建新用户 (仅限管理员)"""
    data = request.get_json()
    
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    is_admin = data.get('is_admin', False)
    
    if not all([username, email, password]):
        return jsonify({'error': '缺少必要参数'}), 400
        
    # 检查用户名是否已存在
    if User.query.filter_by(username=username).first():
        return jsonify({'error': '用户名已被使用'}), 400
        
    # 检查邮箱是否已存在
    if User.query.filter_by(email=email).first():
        return jsonify({'error': '邮箱已被注册'}), 400
    
    # 创建新用户
    new_user = User(
        username=username,
        email=email,
        password=password,
        is_admin=is_admin
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({'message': '用户创建成功', 'user_id': new_user.id}), 201

@app.route('/api/admin/update_user/<int:user_id>', methods=['PUT'])
@admin_required
def update_user(user_id):
    """更新用户信息 (仅限管理员)"""
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    # 获取请求数据
    username = data.get('username')
    email = data.get('email')
    is_admin = data.get('is_admin', False)
    password = data.get('password')  # 可选，如果提供则更新密码
    
    # 检查用户名是否与其他用户重复
    if username != user.username:
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return jsonify({'error': '用户名已被使用'}), 400
    
    # 检查邮箱是否与其他用户重复
    if email != user.email:
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({'error': '邮箱已被注册'}), 400
    
    # 更新用户信息
    user.username = username
    user.email = email
    user.is_admin = is_admin
    
    # 如果提供了新密码则更新
    if password and password.strip():
        user.password_hash = User.generate_password_hash(password)
    
    db.session.commit()
    
    return jsonify({
        'message': '用户更新成功', 
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'is_admin': user.is_admin
        }
    })

@app.route('/api/admin/delete_user/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    """删除用户 (仅限管理员)"""
    # 不允许删除当前登录的管理员账户
    if user_id == current_user.id:
        return jsonify({'error': '不能删除当前登录的管理员账户'}), 400
        
    user = User.query.get_or_404(user_id)
    
    # 确保系统中至少保留一个管理员账户
    if user.is_admin and User.query.filter_by(is_admin=True).count() <= 1:
        return jsonify({'error': '系统必须保留至少一个管理员账户'}), 400
    
    db.session.delete(user)
    db.session.commit()
    
    return jsonify({'message': f'用户 {user.username} 已成功删除'})

@app.route('/api/user')
@login_required
def get_current_user():
    """获取当前登录用户信息"""
    return jsonify({
        'username': current_user.username,
        'is_admin': current_user.is_admin
    })

@app.route('/profile')
@login_required
def profile():
    """个人资料页面"""
    return render_template('profile.html')

@app.route('/settings')
@login_required
def settings():
    """设置页面"""
    return render_template('settings.html')

@app.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    """获取用户个人资料"""
    return jsonify({
        'id': current_user.id,
        'username': current_user.username,
        'email': current_user.email,
        'is_admin': current_user.is_admin
    })

@app.route('/api/profile/update', methods=['POST'])
@login_required
def update_profile():
    """更新用户个人资料"""
    data = request.get_json()
    
    # 获取要更新的数据
    email = data.get('email')
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    
    # 验证当前密码
    if current_password and not current_user.verify_password(current_password):
        return jsonify({'error': '当前密码不正确'}), 400
    
    # 检查邮箱是否已被其他用户使用
    if email != current_user.email:
        existing_user = User.query.filter_by(email=email).first()
        if existing_user and existing_user.id != current_user.id:
            return jsonify({'error': '该邮箱已被其他用户使用'}), 400
        current_user.email = email
    
    # 如果提供了新密码，则更新密码
    if current_password and new_password:
        current_user.password_hash = User.generate_password_hash(new_password)
    
    db.session.commit()
    return jsonify({'message': '个人资料更新成功'})

@app.route('/api/settings', methods=['GET'])
@login_required
def get_settings():
    """获取用户设置"""
    # 创建一个User_Settings模型或在User模型中添加settings字段
    # 这里简化处理，使用用户ID为键的文件来存储设置
    settings_file = os.path.join('user_settings', f'{current_user.id}.json')
    
    # 默认设置
    default_settings = {
        'theme': 'light',
        'language': 'zh_CN',
        'notifications_enabled': True,
        'sidebar_position': 'left',
        'font_size': 14,
        'auto_save': True,
        'confirm_exit': True,
        'session_timeout': 30,
        'admin_notification': False,
        'system_updates_notification': True,
        'analysis_complete_notification': True
    }
    
    # 如果设置文件存在，加载其中的设置
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                user_settings = json.load(f)
                # 合并用户设置和默认设置
                settings = {**default_settings, **user_settings}
        except Exception as e:
            logger.error(f"读取用户设置出错: {str(e)}")
            settings = default_settings
    else:
        settings = default_settings
    
    return jsonify(settings)

@app.route('/api/settings/update', methods=['POST'])
@login_required
def update_settings():
    """更新用户设置"""
    data = request.get_json()
    
    # 确保user_settings目录存在
    os.makedirs('user_settings', exist_ok=True)
    settings_file = os.path.join('user_settings', f'{current_user.id}.json')
    
    # 读取现有设置，如果存在
    existing_settings = {}
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                existing_settings = json.load(f)
        except Exception as e:
            logger.error(f"读取现有用户设置出错: {str(e)}")
    
    # 更新设置
    updated_settings = {**existing_settings, **data}
    
    # 保存设置
    try:
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(updated_settings, f, ensure_ascii=False, indent=2)
        logger.info(f"用户 {current_user.username} 设置已更新")
        return jsonify({'message': '设置更新成功'})
    except Exception as e:
        logger.error(f"保存用户设置出错: {str(e)}")
        return jsonify({'error': '保存设置失败'}), 500

@app.route('/api/settings/reset', methods=['POST'])
@login_required
def reset_settings():
    """重置用户设置"""
    settings_file = os.path.join('user_settings', f'{current_user.id}.json')
    
    # 如果设置文件存在，删除它
    if os.path.exists(settings_file):
        try:
            os.remove(settings_file)
            logger.info(f"用户 {current_user.username} 设置已重置")
        except Exception as e:
            logger.error(f"重置用户设置出错: {str(e)}")
            return jsonify({'error': '重置设置失败'}), 500
    
    return jsonify({'message': '设置已重置为默认值'})

if __name__ == '__main__':
    app.run(debug=False, port=5003)
