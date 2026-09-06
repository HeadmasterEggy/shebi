import os

from flask import Flask

from models import db, User

# Create a minimal Flask app for the context
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'init-only-not-used')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database with the app
db.init_app(app)

if __name__ == "__main__":
    with app.app_context():
        print("创建数据库表...")
        db.create_all()

        # 管理员账户。密码只从环境变量读取——不再在源码里写死 admin123，
        # 否则任何人 clone 下来都知道你部署实例的管理员密码。
        admin = User.query.filter_by(is_admin=True).first()
        if admin:
            print(f"管理员账户已存在: {admin.username}")
        else:
            username = os.environ.get('ADMIN_USERNAME')
            email = os.environ.get('ADMIN_EMAIL')
            password = os.environ.get('ADMIN_PASSWORD')
            if not (username and email and password):
                print(
                    "未创建管理员账户。请设置环境变量后重新运行：\n"
                    "  ADMIN_USERNAME=<用户名> ADMIN_EMAIL=<邮箱> ADMIN_PASSWORD=<密码> \\\n"
                    "  python init_db.py"
                )
            elif len(password) < 8:
                print("ADMIN_PASSWORD 至少需要 8 位。")
            else:
                db.session.add(User(username=username, email=email,
                                    password=password, is_admin=True))
                db.session.commit()
                print(f"已创建管理员账户: {username}")

        print("数据库初始化完成!")
