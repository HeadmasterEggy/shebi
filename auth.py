import functools
import re
import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlparse  # Replace werkzeug.urls import

from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from models import db, User

# 创建Blueprint
auth = Blueprint('auth', __name__)

# 初始化LoginManager
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = '请先登录以访问此页面'

USERNAME_RE = re.compile(r'^[\w\u4e00-\u9fff.-]{2,32}$')
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
MIN_PASSWORD_LEN = 8


def validate_account_fields(username=None, email=None, password=None):
    """校验注册 / 管理员建号时的字段，返回错误信息；全部合法时返回 None。

    传 None 表示不校验该字段（例如更新时没有改用户名）。
    """
    if username is not None and not USERNAME_RE.match(username):
        return '用户名须为 2-32 位中文、字母、数字、下划线、点或连字符'
    if email is not None and (len(email) > 120 or not EMAIL_RE.match(email)):
        return '邮箱格式不正确'
    if password is not None and len(password) < MIN_PASSWORD_LEN:
        return f'密码至少 {MIN_PASSWORD_LEN} 位'
    return None


def wants_json():
    """API 请求返回 JSON 错误而不是重定向到登录页 / 首页。

    前端用 fetch 调 /api/*，原实现对未登录 / 非管理员一律 302 到 HTML 页面，
    fetch 跟随重定向后拿到一整页 HTML，再 response.json() 就抛解析异常，
    界面上只剩一句莫名其妙的 "Unexpected token <"。
    """
    return request.path.startswith('/api/') or request.is_json


class LoginThrottle:
    """同一 (IP, 用户名) 在窗口期内失败次数过多时暂时拒绝登录。

    原实现对失败登录不设任何限制，可以无限次猜密码。进程内计数即可：
    服务本来就只跑一个 worker（见 wsgi.py）。
    """

    def __init__(self, max_failures=5, window_s=600):
        self.max_failures = max_failures
        self.window_s = window_s
        self._failures = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key, now):
        q = self._failures[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        if not q:
            self._failures.pop(key, None)
        return q

    def retry_after(self, key):
        """仍被锁定时返回需等待的秒数，否则返回 0。"""
        now = time.monotonic()
        with self._lock:
            q = self._prune(key, now)
            if len(q) < self.max_failures:
                return 0
            return max(1, int(self.window_s - (now - q[0])))

    def failure(self, key):
        with self._lock:
            self._failures[key].append(time.monotonic())

    def reset(self, key):
        with self._lock:
            self._failures.pop(key, None)


login_throttle = LoginThrottle()


@login_manager.unauthorized_handler
def unauthorized():
    if wants_json():
        return jsonify({'error': '请先登录'}), 401
    flash(login_manager.login_message)
    return redirect(url_for('auth.login', next=request.full_path if request.query_string else request.path))


@login_manager.user_loader
def load_user(user_id):
    """加载用户"""
    return User.query.get(int(user_id))


@auth.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录处理"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        password = data.get('password')
        remember = True if data.get('remember') else False

        throttle_key = (request.remote_addr or '', (username or '').strip().lower())
        wait = login_throttle.retry_after(throttle_key)
        if wait:
            flash(f'登录失败次数过多，请 {max(1, wait // 60)} 分钟后再试')
            return render_template('login.html'), 429

        user = User.query.filter_by(username=username).first()

        if not user or not user.verify_password(password or ''):
            login_throttle.failure(throttle_key)
            flash('用户名或密码错误')
            return redirect(url_for('auth.login'))

        login_throttle.reset(throttle_key)
        login_user(user, remember=remember)
        next_page = request.args.get('next')
        # 只允许站内相对路径。仅判断 netloc 不够："\\evil.com" 与 "/\\evil.com"
        # 的 netloc 为空，但浏览器会把它们当成 //evil.com，形成开放重定向。
        if (not next_page or not next_page.startswith('/') or next_page.startswith('//')
                or '\\' in next_page or urlparse(next_page).netloc or urlparse(next_page).scheme):
            next_page = url_for('index')

        return redirect(next_page)

    return render_template('login.html')


@auth.route('/register', methods=['GET', 'POST'])
def register():
    """用户注册处理"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        data = request.form
        username = (data.get('username') or '').strip()
        email = (data.get('email') or '').strip()
        password = data.get('password') or ''

        error = validate_account_fields(username, email, password)
        if error:
            flash(error)
            return redirect(url_for('auth.register'))

        # 检查用户名是否已存在
        user = User.query.filter_by(username=username).first()
        if user:
            flash('用户名已被使用')
            return redirect(url_for('auth.register'))

        # 检查邮箱是否已存在
        user = User.query.filter_by(email=email).first()
        if user:
            flash('邮箱已被注册')
            return redirect(url_for('auth.register'))

        # 创建新用户
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('注册成功，请登录')
        return redirect(url_for('auth.login'))

    return render_template('register.html')


@auth.route('/logout')
@login_required
def logout():
    """用户注销"""
    logout_user()
    return redirect(url_for('index'))


# 管理员权限检查装饰器
def admin_required(f):
    """检查用户是否为管理员的装饰器"""

    @functools.wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            if wants_json():
                return jsonify({'error': '需要管理员权限'}), 403
            flash('需要管理员权限')
            return redirect(url_for('index'))
        return f(*args, **kwargs)

    return decorated_function
