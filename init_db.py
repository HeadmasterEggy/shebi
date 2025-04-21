import os
from flask import Flask
from models import db, User

# Create a minimal Flask app for the context
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_testing')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database with the app
db.init_app(app)

if __name__ == "__main__":
    with app.app_context():
        print("创建数据库表...")
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
            print("已创建默认管理员账户 (admin/admin123)")
        else:
            print("管理员账户已存在")
        
        print("数据库初始化完成!")
