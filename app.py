import datetime
import logging
import os
import shutil
import time
import uuid

import re

import jieba
import torch
from flask import Flask, jsonify, send_from_directory, render_template, request  # 添加request导入
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from auth import auth, login_manager, admin_required, validate_account_fields, wants_json
from config import Config
from data_Process import clean_text, load_stopwords, sentences_to_array, tokenize
# 导入用户模型和认证模块
from models import db, User
from inference import engine
from metrics_store import load_model_metrics
from utils import ModelNotAvailable
from training_control import (
    TERMINAL_STATUSES,
    atomic_write_json,
    manager as train_manager,
    read_command,
    read_json,
    send_command,
    tail_log,
)

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
_secret = os.environ.get('SECRET_KEY')
if not _secret:
    if os.environ.get('SHEBI_ALLOW_DEV_SECRET') == '1':
        _secret = 'dev_key_for_testing'
        logger.warning('未设置 SECRET_KEY，正在使用开发用默认密钥，请勿用于生产环境')
    else:
        raise RuntimeError(
            '缺少 SECRET_KEY 环境变量。生产环境请设置一个随机值；'
            '本地开发可设置 SHEBI_ALLOW_DEV_SECRET=1 使用默认密钥。'
        )
app.config['SECRET_KEY'] = _secret
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# 请求体大小上限。原来没设，Flask 会把任意大的请求体整个读进内存再解析——
# /api/analyze 的 5 万字上限只在解析完之后才检查，而 /auth/login 连登录都不需要，
# 一个匿名的多 GB 请求就能把服务进程的内存打满。2MB 足够容纳上限内的文本。
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('SHEBI_MAX_REQUEST_BYTES', 2 * 1024 * 1024))



@app.errorhandler(413)
def _request_too_large(_e):
    limit_mb = app.config['MAX_CONTENT_LENGTH'] / 1024 / 1024
    msg = f'请求体过大（上限 {limit_mb:g} MB）'
    if wants_json():
        return jsonify({'error': msg}), 413
    return msg, 413


@app.after_request
def _security_headers(resp):
    """基础安全响应头：禁止 MIME 嗅探、禁止被第三方页面 iframe 嵌套（点击劫持）。"""
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('X-Frame-Options', 'SAMEORIGIN')
    resp.headers.setdefault('Referrer-Policy', 'same-origin')
    return resp


# 初始化扩展
# 不再全局开启 CORS：前端与 API 同源，不需要跨域；原来的 CORS(app) 对任意来源
# 放行所有接口，等于允许任何网站在用户浏览器里调用本站 API。
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['REMEMBER_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_SAMESITE'] = 'Lax'
db.init_app(app)
login_manager.init_app(app)

# 注册认证蓝图
app.register_blueprint(auth, url_prefix='/auth')

# /api/analyze 单次请求的上限：整段文本会被逐句分词并一次性送进模型，
# 不设上限时一个请求就能把服务进程的内存和 CPU 占满
MAX_ANALYZE_CHARS = int(os.environ.get('SHEBI_MAX_ANALYZE_CHARS', 50000))
MAX_ANALYZE_SENTENCES = int(os.environ.get('SHEBI_MAX_ANALYZE_SENTENCES', 2000))

# 用户手动另存的模型副本与用户设置，都属于运行时数据
SAVED_MODELS_DIR = os.path.join(Config.runtime_dir, 'saved_models')
USER_SETTINGS_DIR = os.path.join(Config.runtime_dir, 'user_settings')


def predict_sentences(word2id, model, seq_length, sentences, model_type=None):
    """对已分词的句子做情感判定，返回逐句结果、整体倾向、词频与模型指标。

    原实现先把句子写进临时文件、再读回来转索引：多一次磁盘往返，且模型加载失败时
    临时文件不会被删除。现在全程在内存里完成。
    """
    word_freq = {}
    sentence_results = []
    for sentence in sentences:
        for word in sentence.split():
            if len(word) > 1:
                word_freq[word] = word_freq.get(word, 0) + 1

    input_ids = torch.from_numpy(sentences_to_array(word2id, seq_length, sentences))
    with torch.no_grad():
        probs = torch.softmax(model(input_ids), dim=1).tolist()

    for sentence, (neg, pos) in zip(sentences, probs):
        sentence_results.append({
            "text": sentence,
            "sentiment": "积极" if pos > neg else "消极",
            "confidence": max(pos, neg) * 100,
            "probabilities": {"positive": pos * 100, "negative": neg * 100},
        })

    # 模型评估指标只认 evaluate.py 在测试集上跑出来的、按模型分开存的真实结果；
    # 没有就返回 None，由前端显示「未评测」，绝不编造。
    model_type = model_type or Config.default_model
    model_metrics = load_model_metrics(model_type)
    if model_metrics is None:
        logger.warning("模型 %s 尚无评测记录，请先运行 python evaluate.py --model %s",
                       model_type, model_type)

    total_pos = sum(p for _, p in probs) / len(probs) * 100
    total_neg = 100 - total_pos
    word_freq_list = sorted(({"word": w, "count": c} for w, c in word_freq.items()),
                            key=lambda x: x["count"], reverse=True)
    return {
        "overall": {
            "sentiment": "积极" if total_pos > total_neg else "消极",
            "confidence": max(total_pos, total_neg),
            "probabilities": {"positive": total_pos, "negative": total_neg},
        },
        "sentences": sentence_results,
        "modelMetrics": model_metrics,
        "wordFreq": word_freq_list[:20],
    }


stopwords = load_stopwords()

# 删除 @app.before_first_request 装饰器及其函数，改为在启动前初始化
# 创建数据表和初始化管理员账户
with app.app_context():
    db.create_all()

    # 不再在服务启动时自动创建 admin/admin123。
    # 任何人 clone 这个公开仓库都能读到那组凭据，等于给部署实例留了一道后门。
    # 管理员账户改由 init_db.py 依据环境变量显式创建。
    if not User.query.filter_by(is_admin=True).first():
        logger.warning(
            "数据库中没有管理员账户。请运行："
            "ADMIN_USERNAME=... ADMIN_EMAIL=... ADMIN_PASSWORD=... python init_db.py"
        )


@app.route('/healthz')
def healthz():
    """存活探针，无需登录。

    Dockerfile 的 HEALTHCHECK 原来探测 /api/info——那是个需要登录的调试接口，
    后来被删掉了，容器会因此一直被判为 unhealthy。
    """
    return jsonify({
        'status': 'ok',
        'default_model': Config.default_model,
        'model_loaded': Config.default_model in engine._models,
    })


@app.route('/')
def index():
    """主页面路由，检查用户是否已登录"""
    if current_user.is_authenticated:
        return send_from_directory('static', 'index.html')
    else:
        return render_template('login.html')


# 默认模型统一由 Config 提供
default_model = Config.default_model


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
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': '请求体必须是 JSON 对象'}), 400
    text = data.get('text', '')
    model_type = data.get('model_type') or ''
    if not isinstance(text, str) or not isinstance(model_type, str):
        return jsonify({'error': 'text 与 model_type 必须是字符串'}), 400
    model_type = model_type.lower()
    if not text.strip():
        return jsonify({'error': '请输入要分析的文本'}), 400
    if len(text) > MAX_ANALYZE_CHARS:
        return jsonify({'error': f'文本过长，单次最多 {MAX_ANALYZE_CHARS} 个字符'}), 413

    # 验证模型类型
    if model_type and model_type not in Config.model_choices:
        return jsonify({'error': '不支持的模型类型，请选择有效的模型类型'}), 400

    try:
        # 按回车分割句子
        original_sentences = [s.strip() for s in text.split('\n') if s.strip()]

        if len(original_sentences) > MAX_ANALYZE_SENTENCES:
            return jsonify({'error': f'句子过多，单次最多 {MAX_ANALYZE_SENTENCES} 句'}), 413

        # 处理每个句子。
        #
        # 清洗 + 去停用词之后为空的句子（纯标点、纯表情、纯数字、全是停用词）不送进模型，
        # 但必须记住每条结果对应的是哪一句原文。原实现丢掉空句后按下标回填原文：
        # 输入「！！！\n质量很差」时，「质量很差」的预测结果被贴到了「！！！」头上，
        # 后面每一句都错位一格。
        processed_sentences = []
        kept_indices = []
        skipped_sentences = []
        for idx, sentence in enumerate(original_sentences):
            tokenized_text = tokenize(clean_text(sentence), stopwords)
            if tokenized_text.strip():
                processed_sentences.append(tokenized_text)
                kept_indices.append(idx)
            else:
                skipped_sentences.append(sentence)

        if not processed_sentences:
            return jsonify({
                'error': '输入内容清洗后为空（只有标点、表情、数字或停用词），无法判断情感',
                'skipped': skipped_sentences,
            }), 400

        model = engine.get_model(model_type)
        result = predict_sentences(engine.word2id, model, Config.max_sen_len,
                                   processed_sentences, model_type)

        # 把结果对回各自的原文
        for sentence_result, idx in zip(result['sentences'], kept_indices):
            sentence_result['text'] = original_sentences[idx]
        result['skipped'] = skipped_sentences

        result['modelInfo'] = {'type': (model_type or Config.default_model).upper()}
        return jsonify(result), 200

    except ModelNotAvailable:
        logger.warning('模型 %s 不可用', model_type or Config.default_model, exc_info=True)
        return jsonify({'error': f'{(model_type or Config.default_model).upper()} 模型尚未训练或权重不可用，'
                                 '请先在「模型训练」中训练，或选择其他模型'}), 503
    except Exception:
        # 异常详情只进服务端日志，不回给客户端（可能含文件路径等内部信息）
        logger.exception('分析时出错')
        return jsonify({'error': '分析时出错，详见服务端日志'}), 500


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


def _user_json(user):
    return {'id': user.id, 'username': user.username, 'email': user.email, 'is_admin': user.is_admin}


def _json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


def _str_field(data, key):
    """取字符串字段：缺省 / null 返回 None，非字符串抛 ValueError，其余去掉首尾空白。"""
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'{key} 必须是字符串')
    return value.strip()


def _apply_user_update(user, data):
    """管理员更新用户的公共逻辑。返回 (错误信息, HTTP 状态码)；成功时返回 (None, None)。

    原实现有三个坑：
    - 请求里不带 email 时把邮箱改成 None；不带 is_admin 时默认 False，
      管理员只想改个密码，结果顺手把对方（或自己）降成了普通用户
    - 可以把系统里最后一个管理员降级，之后再没人能进管理后台
    - 用户名 / 邮箱 / 密码不做任何格式校验
    """
    try:
        username = _str_field(data, 'username')
        email = _str_field(data, 'email')
        password = data.get('password')
        if password is not None and not isinstance(password, str):
            raise ValueError('password 必须是字符串')
    except ValueError as e:
        return str(e), 400

    username = username or None
    email = email or None
    password = password if password and password.strip() else None

    error = validate_account_fields(username, email, password)
    if error:
        return error, 400

    if username and username != user.username:
        if User.query.filter(User.username == username, User.id != user.id).first():
            return '用户名已被使用', 400
        user.username = username

    if email and email != user.email:
        if User.query.filter(User.email == email, User.id != user.id).first():
            return '邮箱已被注册', 400
        user.email = email

    if 'is_admin' in data:
        is_admin = bool(data['is_admin'])
        if user.is_admin and not is_admin:
            if user.id == current_user.id:
                return '不能取消自己的管理员权限', 400
            if User.query.filter_by(is_admin=True).count() <= 1:
                return '系统必须保留至少一个管理员账户', 400
        user.is_admin = is_admin

    if password:
        user.set_password(password)
    return None, None


@app.route('/api/admin/create_user', methods=['POST'])
@admin_required
def create_user():
    """创建新用户 (仅限管理员)"""
    data = _json_body()
    if data is None:
        return jsonify({'error': '请求体必须是 JSON 对象'}), 400

    try:
        username = _str_field(data, 'username')
        email = _str_field(data, 'email')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    password = data.get('password')
    if not all([username, email, password]) or not isinstance(password, str):
        return jsonify({'error': '缺少必要参数'}), 400

    error = validate_account_fields(username, email, password)
    if error:
        return jsonify({'error': error}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'error': '用户名已被使用'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': '邮箱已被注册'}), 400

    new_user = User(username=username, email=email, password=password,
                    is_admin=bool(data.get('is_admin', False)))
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': '用户创建成功', 'user_id': new_user.id}), 201


@app.route('/api/admin/update_user/<int:user_id>', methods=['PUT'])
@app.route('/api/admin/user/<int:user_id>', methods=['PUT'])
@admin_required
def update_user(user_id):
    """更新用户信息 (仅限管理员)。字段均可选，没传的字段保持不变。"""
    user = db.get_or_404(User, user_id)
    data = _json_body()
    if data is None:
        return jsonify({'error': '请求体必须是 JSON 对象'}), 400

    error, status = _apply_user_update(user, data)
    if error:
        db.session.rollback()
        return jsonify({'error': error}), status

    db.session.commit()
    return jsonify({'message': '用户更新成功', 'user': _user_json(user)})


@app.route('/api/admin/delete_user/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    """删除用户 (仅限管理员)"""
    # 不允许删除当前登录的管理员账户
    if user_id == current_user.id:
        return jsonify({'error': '不能删除当前登录的管理员账户'}), 400

    user = db.get_or_404(User, user_id)

    # 确保系统中至少保留一个管理员账户
    if user.is_admin and User.query.filter_by(is_admin=True).count() <= 1:
        return jsonify({'error': '系统必须保留至少一个管理员账户'}), 400

    db.session.delete(user)
    db.session.commit()

    return jsonify({'message': f'用户 {user.username} 已成功删除'})


@app.route('/api/admin/user/<int:user_id>', methods=['GET'])
@admin_required
def get_user_by_id(user_id):
    """获取单个用户信息 (仅限管理员)"""
    return jsonify(_user_json(db.get_or_404(User, user_id)))


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
    """更新用户个人资料。改密码需要验证当前密码。

    原实现：修改密码的请求里不带 email 字段，于是 `email != current_user.email`
    成立，邮箱被顺手改成了 None——每改一次密码就丢一次邮箱；只传 new_password
    不传 current_password 时静默忽略，却仍然返回"更新成功"。
    """
    data = _json_body()
    if data is None:
        return jsonify({'error': '请求体必须是 JSON 对象'}), 400
    try:
        email = _str_field(data, 'email')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    current_password = data.get('current_password') or ''
    new_password = data.get('new_password') or ''
    if not isinstance(current_password, str) or not isinstance(new_password, str):
        return jsonify({'error': '密码必须是字符串'}), 400

    changing_email = bool(email) and email != current_user.email
    changing_password = bool(new_password)
    if not (changing_email or changing_password):
        return jsonify({'message': '没有需要更新的内容'})

    if changing_password:
        if not current_password:
            return jsonify({'error': '请输入当前密码'}), 400
        if not current_user.verify_password(current_password):
            return jsonify({'error': '当前密码不正确'}), 400

    error = validate_account_fields(
        email=email if changing_email else None,
        password=new_password if changing_password else None,
    )
    if error:
        return jsonify({'error': error}), 400

    if changing_email:
        if User.query.filter(User.email == email, User.id != current_user.id).first():
            return jsonify({'error': '该邮箱已被其他用户使用'}), 400
        current_user.email = email
    if changing_password:
        current_user.set_password(new_password)

    db.session.commit()
    return jsonify({'message': '个人资料更新成功'})


DEFAULT_SETTINGS = {
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
    'analysis_complete_notification': True,
}
# 单个设置值的大小上限：设置文件按用户落盘，不设限等于给每个账号一块无限大的磁盘
_MAX_SETTING_VALUE_LEN = 1000
_MAX_SETTINGS_KEYS = 64


def _settings_path(user_id):
    return os.path.join(USER_SETTINGS_DIR, f'{int(user_id)}.json')


def _load_user_settings(user_id):
    data = read_json(_settings_path(user_id))
    return data if isinstance(data, dict) else {}


def _validate_settings(data):
    """只接受标量值，键名与长度有上限。返回错误信息或 None。"""
    if len(data) > _MAX_SETTINGS_KEYS:
        return '设置项过多'
    for key, value in data.items():
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_]{1,64}', key):
            return f'非法的设置项名称: {key!r}'
        if value is not None and not isinstance(value, (str, int, float, bool)):
            return f'设置项 {key} 的值必须是字符串、数字或布尔值'
        if isinstance(value, str) and len(value) > _MAX_SETTING_VALUE_LEN:
            return f'设置项 {key} 的值过长'
    return None


@app.route('/api/settings', methods=['GET'])
@login_required
def get_settings():
    """获取用户设置（用户保存的值覆盖默认值）"""
    return jsonify({**DEFAULT_SETTINGS, **_load_user_settings(current_user.id)})


@app.route('/api/settings/update', methods=['POST'])
@login_required
def update_settings():
    """更新用户设置"""
    data = _json_body()
    if data is None:
        return jsonify({'error': '请求体必须是 JSON 对象'}), 400
    error = _validate_settings(data)
    if error:
        return jsonify({'error': error}), 400

    updated = {**_load_user_settings(current_user.id), **data}
    if len(updated) > _MAX_SETTINGS_KEYS:
        return jsonify({'error': '设置项过多'}), 400
    try:
        os.makedirs(USER_SETTINGS_DIR, exist_ok=True)
        atomic_write_json(_settings_path(current_user.id), updated, indent=2)
    except OSError:
        logger.exception('保存用户设置出错')
        return jsonify({'error': '保存设置失败'}), 500
    logger.info(f"用户 {current_user.username} 设置已更新")
    return jsonify({'message': '设置更新成功'})


@app.route('/api/settings/reset', methods=['POST'])
@login_required
def reset_settings():
    """重置用户设置"""
    try:
        os.remove(_settings_path(current_user.id))
        logger.info(f"用户 {current_user.username} 设置已重置")
    except FileNotFoundError:
        pass
    except OSError:
        logger.exception('重置用户设置出错')
        return jsonify({'error': '重置设置失败'}), 500
    return jsonify({'message': '设置已重置为默认值'})


# ---------------------------------------------------------------------------
# 训练调度
# ---------------------------------------------------------------------------
# 训练会占满 CPU 数十分钟并覆盖线上正在使用的权重文件，属于有副作用的操作，
# 与暂停 / 停止一样只对管理员开放（原实现启动接口只要求登录，任何注册用户都能
# 反复触发训练）。agent 层的 train_model 工具同样被标记为"需人工确认"。

def _initial_progress(total_epochs=0, status='initializing', message=None):
    data = {
        'status': status,
        'current_epoch': 0,
        'total_epochs': total_epochs,
        'train_loss': 0,
        'val_loss': 0,
        'val_acc': 0,
        'history': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
    }
    if message:
        data['message'] = message
    return data


def _num(data, key, default, cast, lo, hi):
    """取一个数值参数并做范围校验；不合法时抛 ValueError（由调用方转成 400）。"""
    value = data.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f'参数 {key} 必须是数字')
    try:
        value = cast(value)
    except (TypeError, ValueError):
        raise ValueError(f'参数 {key} 必须是数字')
    if not (lo <= value <= hi):
        raise ValueError(f'参数 {key} 超出范围 [{lo}, {hi}]')
    return value


TRAIN_OPTIMIZERS = ('adam', 'sgd', 'rmsprop')


def parse_training_params(data):
    """校验前端传来的训练参数，返回写入 params.json 的字典。"""
    model_type = str(data.get('model_type', 'cnn')).lower()
    if model_type not in Config.model_choices:
        raise ValueError(f'不支持的模型类型: {model_type}')

    params = {
        'model_type': model_type,
        'batch_size': _num(data, 'batch_size', 128, int, 1, 4096),
        'epochs': _num(data, 'epochs', 10, int, 1, 1000),
        'dropout': _num(data, 'dropout', 0.5, float, 0.0, 0.95),
        'weight_decay': _num(data, 'weight_decay', 1e-4, float, 0.0, 1.0),
        'optimizer': str(data.get('optimizer', 'adam')).lower(),
    }
    if params['optimizer'] not in TRAIN_OPTIMIZERS:
        raise ValueError(f"不支持的优化器: {params['optimizer']}（可选 {', '.join(TRAIN_OPTIMIZERS)}）")
    if 'learning_rate' in data:
        params['learning_rate'] = _num(data, 'learning_rate', Config.lr, float, 1e-7, 1.0)

    if model_type == 'cnn':
        params['num_filters'] = _num(data, 'num_filters', Config.num_filters, int, 1, 1024)
        # 前端以 kernel_sizes 提交卷积核尺寸；原来后端不认这个字段，选什么都按 (2, 3, 4) 训练
        sizes = data.get('kernel_sizes', data.get('filter_sizes'))
        if sizes is not None:
            if (not isinstance(sizes, list) or not sizes
                    or not all(isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= 10 for k in sizes)):
                raise ValueError('kernel_sizes 必须是 1-10 之间整数组成的非空列表')
            params['filter_sizes'] = sorted(set(sizes))
    else:
        params['hidden_dim'] = _num(data, 'hidden_dim', Config.hidden_dim, int, 1, 2048)
        params['num_layers'] = _num(data, 'num_layers', Config.num_layers, int, 1, 8)

    if data.get('early_stopping'):
        params['early_stopping'] = True
        params['patience'] = _num(data, 'patience', 5, int, 1, 1000)
    return params


def _read_progress():
    return read_json(Config.progress_path)


def _sync_progress_with_process(progress):
    """训练进程已退出但状态还停在 running / paused 时（被 kill、OOM、崩溃），
    依据退出码把状态改成终态，否则前端会一直显示"训练中"。"""
    if progress is None or progress.get('status') in TERMINAL_STATUSES:
        return progress
    if not train_manager.has_handle() or train_manager.is_running():
        return progress
    code = train_manager.returncode()
    if code == 0:
        progress['status'] = 'completed'
        progress.setdefault('message', '训练已完成')
    else:
        progress['status'] = 'failed'
        progress['error'] = f'训练进程异常退出，返回码: {code}'
    atomic_write_json(Config.progress_path, progress)
    return progress


@app.route('/api/train/start', methods=['POST'])
@admin_required
def start_training():
    """启动模型训练任务"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': '请求体必须是 JSON 对象'}), 400
    try:
        params = parse_training_params(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    if train_manager.is_running():
        return jsonify({'error': '已有训练任务正在运行，请先停止或等待其结束'}), 409

    try:
        atomic_write_json(Config.params_path, params)
        atomic_write_json(Config.progress_path, _initial_progress(params['epochs']))
        pid = train_manager.start(args=['--params', Config.params_path])
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 409
    except Exception:
        logger.exception('启动训练出错')
        return jsonify({'error': '启动训练失败，详见服务端日志'}), 500

    # 新权重写盘后，推理引擎会按文件修改时间自动重新加载，这里无需手动失效缓存
    logger.info('训练任务已启动 pid=%s params=%s', pid, params)
    return jsonify({
        'success': True,
        'message': '训练任务已启动',
        'pid': pid,
        'training_id': str(uuid.uuid4()),
        'params': params,
    })


@app.route('/api/train/progress', methods=['GET'])
@login_required
def get_training_progress():
    """获取训练进度。

    原文件里这个 URL 注册了两次：第二个（带进程退出检测的那个）永远不会被路由到，
    于是训练进程崩溃后进度一直停在 running。现在合并成一个。
    """
    progress = _read_progress()
    if progress is None:
        if os.path.exists(Config.progress_path):
            # 原子写入后理论上不会出现半截文件，这里兜底手工改坏的情况
            return jsonify({'status': 'error', 'message': '进度文件格式错误',
                            'error': '无法解析进度文件'})
        progress = _initial_progress(message='等待训练开始')
    else:
        progress = _sync_progress_with_process(progress)

    progress['running'] = train_manager.is_running()
    progress['recent_logs'] = get_recent_training_logs()
    progress['timestamp'] = datetime.datetime.now().isoformat()
    return jsonify(progress)


def get_recent_training_logs(max_lines=10):
    """训练子进程输出的末尾几行。

    原实现读 log/training.log，但从来没有任何代码写这个文件——子进程的输出
    进了一个没人读的管道。现在子进程的 stdout 直接重定向到 runtime/training.log。
    """
    return tail_log(max_lines)


@app.route('/api/train/pause', methods=['POST'])
@admin_required
def pause_training():
    """暂停 / 继续训练（切换）。指令写入控制文件，训练进程在 batch 之间响应。"""
    if not train_manager.is_running():
        return jsonify({'error': '没有正在进行的训练任务'}), 404

    if read_command() == 'pause':
        send_command('resume')
        return jsonify({'status': 'resumed'})
    send_command('pause')
    return jsonify({'status': 'paused'})


@app.route('/api/train/stop', methods=['POST'])
@admin_required
def stop_training():
    """停止训练。

    默认优雅停止：训练进程在下一个 batch 前退出，已保存的最佳权重不受影响。
    传 {"force": true} 时直接终止进程（训练进程卡死时使用）。
    """
    if not train_manager.is_running():
        return jsonify({'error': '没有正在进行的训练任务'}), 404

    data = request.get_json(silent=True) or {}
    if isinstance(data, dict) and data.get('force'):
        train_manager.terminate()
        progress = _read_progress() or _initial_progress()
        progress['status'] = 'stopped'
        progress['message'] = '训练已被强制终止'
        atomic_write_json(Config.progress_path, progress)
        return jsonify({'status': 'stopped'})

    send_command('stop')
    return jsonify({'status': 'stopping'})


@app.route('/api/train/save_model', methods=['POST'])
@admin_required
def save_model():
    """把最近一次训练的最佳权重另存一份副本。"""
    data = request.get_json(silent=True) or {}
    raw_name = str(data.get('name') or f'model_{int(time.time())}')
    # name 会被拼进文件路径：原实现直接拼接，传 "../../app" 就能写到仓库任意位置
    name = secure_filename(raw_name)
    if not name:
        return jsonify({'error': '模型名称不合法'}), 400

    params = read_json(Config.params_path)
    if not params:
        return jsonify({'error': '找不到训练参数文件'}), 404

    model_type = str(params.get('model_type', 'cnn')).lower()
    if model_type not in Config.model_choices:
        return jsonify({'error': f'参数文件中的模型类型非法: {model_type}'}), 400
    source_path = os.path.join(Config.model_dir, f"{model_type}_model_best.pkl")
    if not os.path.exists(source_path):
        return jsonify({'error': '找不到模型文件'}), 404

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    target_filename = f"{name}.pkl"
    target_path = os.path.join(SAVED_MODELS_DIR, target_filename)
    try:
        shutil.copy2(source_path, target_path)
    except OSError:
        logger.exception('保存模型出错')
        return jsonify({'error': '保存模型失败'}), 500

    return jsonify({
        'status': 'success',
        'filename': target_filename,
        'path': os.path.relpath(target_path, Config.base_dir),
    })


# ---------------------------------------------------------------------------
# Agent 执行轨迹
# ---------------------------------------------------------------------------
# 轨迹本来就一直在往 runtime/traces/ 落盘，但只有能读 JSON 的人看得懂。
# 把它画成一条时间线之后，"这个 agent 到底做了什么"从一句话变成一屏可点开的
# 证据——每一步的思考、调了哪个工具、传了什么参数、拿回什么、花了多少钱。
@app.route('/traces')
@login_required
def traces_page():
    return render_template('traces.html')


@app.route('/api/traces', methods=['GET'])
@login_required
def api_traces():
    """轨迹列表，最新的在前。"""
    from agent.trace import list_traces, load_trace

    items = []
    for run_id in reversed(list_traces()):
        try:
            data = load_trace(run_id)
        except (OSError, ValueError):
            continue
        summary = data.get('summary', {})
        items.append({
            'run_id': run_id,
            'task': data.get('task', ''),
            'status': data.get('status', 'unknown'),
            'model': data.get('model', ''),
            'started_at': data.get('started_at'),
            'records': summary.get('records'),
            'tool_calls': summary.get('tool_calls'),
            'roles': summary.get('roles') or [],
            'duration_s': summary.get('duration_s'),
            'cost_usd': summary.get('cost_usd'),
            'total_tokens': summary.get('total_tokens'),
            'unsupported_rate_final': summary.get('unsupported_rate_final'),
        })
    return jsonify({'traces': items, 'count': len(items)})


@app.route('/api/traces/<run_id>', methods=['GET'])
@login_required
def api_trace_detail(run_id):
    from agent.trace import load_trace

    # run_id 会被拼进文件路径，只放行十六进制——否则 ../ 就能读到仓库里任何文件
    if not re.fullmatch(r'[0-9a-f]{6,32}', run_id or ''):
        return jsonify({'error': '非法的 run_id'}), 400
    try:
        return jsonify(load_trace(run_id))
    except FileNotFoundError:
        return jsonify({'error': f'找不到轨迹 {run_id}'}), 404


def _warmup():
    """进程启动时预热词表 / 词向量 / 默认模型。"""
    try:
        engine.warmup()
    except Exception as e:  # 预热失败不应阻止服务启动
        logger.warning("推理引擎预热失败（首个请求会变慢）: %s", e)


if __name__ == '__main__':
    _warmup()
    app.run(debug=False, port=int(os.environ.get('PORT', 5003)), host=os.environ.get('HOST', '127.0.0.1'))
