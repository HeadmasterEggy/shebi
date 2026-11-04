import hashlib

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model, UserMixin):
    """用户模型，支持普通用户和管理员角色"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True)
    email = db.Column(db.String(120), unique=True, index=True)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)

    def __init__(self, username, email, password, is_admin=False):
        self.username = username
        self.email = email
        self.password_hash = generate_password_hash(password)
        self.is_admin = is_admin

    @property
    def session_token(self):
        """密码哈希的指纹。改密码后指纹随之变化。"""
        return hashlib.sha256((self.password_hash or "").encode()).hexdigest()[:16]

    def get_id(self):
        """写进会话与"记住我" Cookie 的标识：用户 id + 密码指纹。

        Flask-Login 默认只存用户 id，于是改密码不会让任何已登录的会话失效——账号被盗后
        改密码，对方的会话和 365 天的"记住我" Cookie 照样能用。带上密码指纹后，
        改密码即作废此前签发的所有会话（校验见 auth.load_user）。
        """
        return f"{self.id}:{self.session_token}"

    def verify_password(self, password):
        """验证密码"""
        return check_password_hash(self.password_hash, password)

    def set_password(self, password):
        """设置密码散列"""
        self.password_hash = generate_password_hash(password)

    def __repr__(self):
        return f'<User {self.username}>'
