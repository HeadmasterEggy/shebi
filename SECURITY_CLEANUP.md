# 敏感数据清理说明

## 问题

`instance/users.db` 被提交进了这个**公开仓库**，里面有 6 个真实账号：用户名、邮箱
（含仓库所有者的 gmail）和 werkzeug 密码哈希。同时 `app.py` 与 `init_db.py` 在源码里
写死了默认管理员 `admin / admin123`，任何人 clone 下来都知道线上实例的管理员凭据。

已在代码层修复：

- `instance/` 与 `*.db` 加入 `.gitignore`，`users.db` 已从索引移除
- `SECRET_KEY` 必须由环境变量提供（本地开发可设 `SHEBI_ALLOW_DEV_SECRET=1`）
- 管理员账户改由 `init_db.py` 依据 `ADMIN_USERNAME / ADMIN_EMAIL / ADMIN_PASSWORD` 创建
- 服务启动不再自动建号，只在缺少管理员时打一条警告

## 还需要手动做的：清理 git 历史

从索引里删掉文件**不会**让它从历史中消失——任何人 `git log` 都还能拿到。
必须重写历史：

```bash
pip install git-filter-repo

# 在仓库的一个全新 clone 上操作
git clone https://github.com/HeadmasterEggy/shebi.git shebi-clean
cd shebi-clean

git filter-repo --invert-paths --path instance/users.db

git remote add origin https://github.com/HeadmasterEggy/shebi.git
git push origin --force --all
git push origin --force --tags
```

注意事项：

1. 强推会改写所有 commit 的 SHA，本地其他 clone 需要重新 clone
2. GitHub 上已有的 fork 和缓存的 commit 视图不会自动清理，
   需要在仓库 Settings 里提 support 请求，或直接把仓库删掉重建
3. **把那 6 个账号用过的密码在其他站点也换掉**——密码哈希已经公开暴露过

## 检查历史里还有没有别的东西

```bash
git log --all --oneline --name-only | grep -iE '\.db$|\.env$|\.pem$|secret|credential|password'
```
