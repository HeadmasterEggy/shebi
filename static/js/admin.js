/**
 * 管理员功能模块
 */

let adminInitialized = false;
let users = []; // 存储用户列表数据

/**
 * 获取所有用户列表
 */
async function fetchUsers() {
    try {
        // 减少调试日志，保留必要信息
        const response = await fetch('/api/admin/users');
        if (response.ok) {
            const data = await response.json();
            users = data.users; // 存储用户数据
            displayUsers(data.users);
        } else if (response.status === 401) {
            showError('未授权：请先登录');
        } else if (response.status === 403) {
            showError('权限不足：需要管理员权限');
        } else {
            const error = await response.json();
            showError('获取用户列表失败：' + (error.error || '未知错误'));
        }
    } catch (error) {
        console.error('获取用户列表出错:', error);
        showError('获取用户列表出错：' + error.message);
    }
}

/**
 * 显示用户列表
 */
function displayUsers(users) {
    const tableBody = document.getElementById('usersTableBody');
    if (!tableBody) {
        console.error('找不到用户表格主体元素');
        return;
    }
    
    if (users && users.length > 0) {
        let html = '';
        users.forEach(user => {
            html += `
                <tr>
                    <td>${user.id}</td>
                    <td>${user.username}</td>
                    <td>${user.email}</td>
                    <td>
                        <span class="badge ${user.is_admin ? 'bg-danger' : 'bg-primary'}">
                            ${user.is_admin ? '管理员' : '用户'}
                        </span>
                    </td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary me-1" onclick="editUser(${user.id})">
                            <i class="bi bi-pencil"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-danger" onclick="deleteUser(${user.id})">
                            <i class="bi bi-trash"></i>
                        </button>
                    </td>
                </tr>
            `;
        });
        tableBody.innerHTML = html;
    } else {
        tableBody.innerHTML = '<tr><td colspan="5" class="text-center">没有用户数据</td></tr>';
    }
}

/**
 * 创建新用户
 */
async function createUser() {
    const username = document.getElementById('newUsername').value;
    const email = document.getElementById('newEmail').value;
    const password = document.getElementById('newPassword').value;
    const isAdmin = document.getElementById('isAdmin').checked;
    
    if (!username || !email || !password) {
        alert('请填写所有必填字段');
        return;
    }
    
    try {
        const response = await fetch('/api/admin/create_user', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: username,
                email: email,
                password: password,
                is_admin: isAdmin
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            alert('用户创建成功');
            // 关闭模态框并重新加载用户列表
            const modal = bootstrap.Modal.getInstance(document.getElementById('createUserModal'));
            modal.hide();
            fetchUsers();
            
            // 清空表单
            document.getElementById('createUserForm').reset();
        } else {
            alert('创建用户失败：' + (data.error || '未知错误'));
        }
    } catch (error) {
        console.error('创建用户出错:', error);
        alert('创建用户出错：' + error.message);
    }
}

/**
 * 编辑用户
 * @param {number} userId - 用户ID
 */
function editUser(userId) {
    // 从用户列表中查找当前用户数据
    const user = users.find(u => u.id === userId);
    if (!user) {
        showError(`找不到ID为${userId}的用户`);
        return;
    }
    
    // 填充表单
    document.getElementById('editUserId').value = user.id;
    document.getElementById('editUsername').value = user.username;
    document.getElementById('editEmail').value = user.email;
    document.getElementById('editIsAdmin').checked = user.is_admin;
    document.getElementById('editPassword').value = ''; // 密码框清空
    
    // 显示编辑模态框
    const modal = new bootstrap.Modal(document.getElementById('editUserModal'));
    modal.show();
}

/**
 * 更新用户信息
 */
async function updateUser() {
    const userId = document.getElementById('editUserId').value;
    const username = document.getElementById('editUsername').value;
    const email = document.getElementById('editEmail').value;
    const password = document.getElementById('editPassword').value;
    const isAdmin = document.getElementById('editIsAdmin').checked;
    
    if (!username || !email) {
        alert('用户名和邮箱不能为空');
        return;
    }
    
    try {
        const response = await fetch(`/api/admin/update_user/${userId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: username,
                email: email,
                password: password, // 如果为空，后端将不更新密码
                is_admin: isAdmin
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            alert('用户更新成功');
            // 关闭模态框并重新加载用户列表
            const modal = bootstrap.Modal.getInstance(document.getElementById('editUserModal'));
            modal.hide();
            fetchUsers();
        } else {
            alert('更新用户失败：' + (data.error || '未知错误'));
        }
    } catch (error) {
        console.error('更新用户出错:', error);
        alert('更新用户出错：' + error.message);
    }
}

/**
 * 删除用户
 * @param {number} userId - 用户ID
 */
function deleteUser(userId) {
    // 从用户列表中查找当前用户数据
    const user = users.find(u => u.id === userId);
    if (!user) {
        showError(`找不到ID为${userId}的用户`);
        return;
    }
    
    // 填充确认对话框
    document.getElementById('deleteUserId').value = user.id;
    document.getElementById('deleteUserName').textContent = user.username;
    
    // 显示确认对话框
    const modal = new bootstrap.Modal(document.getElementById('deleteConfirmModal'));
    modal.show();
}

/**
 * 确认删除用户
 */
async function confirmDeleteUser() {
    const userId = document.getElementById('deleteUserId').value;
    
    try {
        const response = await fetch(`/api/admin/delete_user/${userId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            const data = await response.json();
            alert(data.message || '用户删除成功');
            
            // 关闭模态框并重新加载用户列表
            const modal = bootstrap.Modal.getInstance(document.getElementById('deleteConfirmModal'));
            modal.hide();
            fetchUsers();
        } else {
            const error = await response.json();
            alert('删除用户失败：' + (error.error || '未知错误'));
        }
    } catch (error) {
        console.error('删除用户出错:', error);
        alert('删除用户出错：' + error.message);
    }
}

/**
 * 初始化管理员功能
 */
function initAdminFeatures() {
    if (adminInitialized) {
        return;
    }
    
    // 确保管理员区域显示
    const adminSection = document.getElementById('admin-section');
    if (adminSection) {
        adminSection.style.display = 'flex';
    }
    
    // 绑定创建用户按钮事件
    const createUserBtn = document.getElementById('createUserBtn');
    if (createUserBtn) {
        createUserBtn.addEventListener('click', function() {
            const modal = new bootstrap.Modal(document.getElementById('createUserModal'));
            modal.show();
        });
    } else {
        console.error('找不到创建用户按钮');
    }
    
    // 绑定保存用户按钮事件
    const saveUserBtn = document.getElementById('saveUserBtn');
    if (saveUserBtn) {
        saveUserBtn.addEventListener('click', createUser);
    } else {
        console.error('找不到保存用户按钮');
    }
    
    // 绑定更新用户按钮事件
    const updateUserBtn = document.getElementById('updateUserBtn');
    if (updateUserBtn) {
        updateUserBtn.addEventListener('click', updateUser);
    } else {
        console.error('找不到更新用户按钮');
    }
    
    // 绑定确认删除用户按钮事件
    const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
    if (confirmDeleteBtn) {
        confirmDeleteBtn.addEventListener('click', confirmDeleteUser);
    } else {
        console.error('找不到确认删除按钮');
    }
    
    // 当切换到管理员页面时自动加载用户列表
    const adminMenuItem = document.querySelector('.menu-item[data-section="admin-section"]');
    if (adminMenuItem) {
        adminMenuItem.addEventListener('click', function() {
            fetchUsers();
        });
    } else {
        console.error('找不到管理员菜单项');
    }
    
    adminInitialized = true;
}

// 使用新增的用户信息加载回调来确保初始化发生在正确的时机
document.addEventListener('DOMContentLoaded', function() {
    // 使用auth.js提供的回调机制
    if (typeof onUserInfoLoaded === 'function') {
        onUserInfoLoaded(function(username, isAdmin) {
            if (isAdmin) {
                initAdminFeatures();
            }
        });
    } else {
        // 备用方案：如果auth.js未提供回调机制，尝试延迟检查
        setTimeout(function() {
            if (typeof isAdmin !== 'undefined' && isAdmin) {
                initAdminFeatures();
            }
        }, 1000); // 给fetchUserInfo一点时间完成
    }
});

// 添加全局错误显示函数，如果shared.js中没有提供
if (typeof showError !== 'function') {
    window.showError = function(message) {
        const errorElement = document.getElementById('errorMessage');
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';
            setTimeout(() => {
                errorElement.style.display = 'none';
            }, 5000);
        } else {
            alert(message);
        }
    };
}
