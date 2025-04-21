/**
 * 个人资料页面功能
 */

document.addEventListener('DOMContentLoaded', function() {
    // 加载个人资料信息
    loadProfileData();
    
    // 表单提交处理
    const profileForm = document.getElementById('profileForm');
    if (profileForm) {
        profileForm.addEventListener('submit', function(e) {
            e.preventDefault();
            updateProfile();
        });
    }
    
    // 密码更改处理
    const changePasswordBtn = document.getElementById('changePasswordBtn');
    if (changePasswordBtn) {
        changePasswordBtn.addEventListener('click', function() {
            changePassword();
        });
    }
});

/**
 * 加载用户个人资料数据
 */
async function loadProfileData() {
    try {
        const response = await fetch('/api/profile');
        if (response.ok) {
            const data = await response.json();
            
            // 更新页面上的用户信息
            document.getElementById('profileUsername').textContent = data.username;
            document.getElementById('username').value = data.username;
            document.getElementById('profileEmail').textContent = data.email;
            document.getElementById('email').value = data.email;
            document.getElementById('profileRole').textContent = data.is_admin ? '管理员' : '普通用户';
            
            // 更新顶部导航栏
            document.getElementById('navbarUsername').textContent = data.username;
            
            const navbarUserRole = document.getElementById('navbarUserRole');
            if (navbarUserRole) {
                navbarUserRole.textContent = data.is_admin ? '管理员' : '用户';
                navbarUserRole.className = data.is_admin ? 'badge bg-danger ms-1' : 'badge bg-primary ms-1';
            }
        } else {
            showMessage('profileMessage', '加载个人资料失败', 'danger');
        }
    } catch (error) {
        console.error('获取个人资料出错:', error);
        showMessage('profileMessage', '获取个人资料时发生错误', 'danger');
    }
}

/**
 * 更新用户个人资料
 */
async function updateProfile() {
    const email = document.getElementById('email').value;
    
    try {
        const response = await fetch('/api/profile/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                email: email
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showMessage('profileMessage', '个人资料更新成功', 'success');
            loadProfileData(); // 重新加载数据
        } else {
            showMessage('profileMessage', data.error || '更新个人资料失败', 'danger');
        }
    } catch (error) {
        console.error('更新个人资料出错:', error);
        showMessage('profileMessage', '更新个人资料时发生错误', 'danger');
    }
}

/**
 * 修改密码
 */
async function changePassword() {
    const currentPassword = document.getElementById('currentPassword').value;
    const newPassword = document.getElementById('newPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    
    // 检查密码确认是否匹配
    if (newPassword !== confirmPassword) {
        showMessage('passwordMessage', '新密码和确认密码不匹配', 'danger');
        return;
    }
    
    try {
        const response = await fetch('/api/profile/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                current_password: currentPassword,
                new_password: newPassword
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showMessage('passwordMessage', '密码修改成功', 'success');
            setTimeout(() => {
                const modal = bootstrap.Modal.getInstance(document.getElementById('changePasswordModal'));
                if (modal) {
                    modal.hide();
                    // 清空密码字段
                    document.getElementById('currentPassword').value = '';
                    document.getElementById('newPassword').value = '';
                    document.getElementById('confirmPassword').value = '';
                    // 隐藏消息
                    document.getElementById('passwordMessage').classList.add('d-none');
                }
            }, 1500);
        } else {
            showMessage('passwordMessage', data.error || '密码修改失败', 'danger');
        }
    } catch (error) {
        console.error('修改密码出错:', error);
        showMessage('passwordMessage', '修改密码时发生错误', 'danger');
    }
}

/**
 * 显示消息
 * @param {string} elementId - 消息元素ID
 * @param {string} message - 显示的消息
 * @param {string} type - 消息类型 (success, danger, warning, info)
 */
function showMessage(elementId, message, type) {
    const messageElement = document.getElementById(elementId);
    if (messageElement) {
        messageElement.textContent = message;
        messageElement.className = `alert alert-${type}`;
        messageElement.classList.remove('d-none');
        
        // 成功消息3秒后自动隐藏
        if (type === 'success') {
            setTimeout(() => {
                messageElement.classList.add('d-none');
            }, 3000);
        }
    }
}
