from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from urllib.parse import urlparse  # Replace werkzeug.urls import
from models import db, User

# 创建Blueprint
auth = Blueprint('auth', __name__)

# 初始化LoginManager
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = '请先登录以访问此页面'

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
        
        user = User.query.filter_by(username=username).first()
        
        if not user or not user.verify_password(password):
            flash('用户名或密码错误')
            return redirect(url_for('auth.login'))
            
        login_user(user, remember=remember)
        next_page = request.args.get('next')
        if not next_page or urlparse(next_page).netloc != '':  # Updated to use urlparse
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
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
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

@auth.route('/api/user')
@login_required
def get_user():
    """获取当前用户信息API"""
    return jsonify({
        'username': current_user.username,
        'is_admin': current_user.is_admin
    })

# 管理员权限检查装饰器
def admin_required(f):
    """检查用户是否为管理员的装饰器"""
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            flash('需要管理员权限')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function
