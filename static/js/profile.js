/**
 * 个人资料页面功能
 */

document.addEventListener('DOMContentLoaded', function () {
    // 加载个人资料信息
    loadProfileData();
    
    // 加载用户活动记录
    loadUserActivities();

    // 表单提交处理
    const profileForm = document.getElementById('profileForm');
    if (profileForm) {
        profileForm.addEventListener('submit', function (e) {
            e.preventDefault();
            updateProfile();
        });
    }

    // 密码更改处理
    const changePasswordBtn = document.getElementById('changePasswordBtn');
    if (changePasswordBtn) {
        changePasswordBtn.addEventListener('click', function () {
            changePassword();
        });
    }
    
    // 添加查看密码功能
    const togglePasswordBtns = document.querySelectorAll('.toggle-password');
    if (togglePasswordBtns) {
        togglePasswordBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const input = this.previousElementSibling;
                const type = input.getAttribute('type') === 'password' ? 'text' : 'password';
                input.setAttribute('type', type);
                this.querySelector('i').classList.toggle('bi-eye');
                this.querySelector('i').classList.toggle('bi-eye-slash');
            });
        });
    }
    
    // 添加密码强度检测
    const newPassword = document.getElementById('newPassword');
    if (newPassword) {
        newPassword.addEventListener('input', checkPasswordStrength);
    }
    
    // 注销链接处理
    const logoutLink = document.getElementById('logoutLink');
    if (logoutLink) {
        logoutLink.addEventListener('click', function(e) {
            e.preventDefault();
            if (confirm('确定要退出登录吗？')) {
                if (typeof logout === 'function') {
                    logout();
                } else {
                    window.location.href = '/auth/logout';
                }
            }
        });
    }
    
    // 表单输入动画效果
    const formInputs = document.querySelectorAll('.form-control');
    formInputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
    });
});

/**
 * 检查密码强度
 */
function checkPasswordStrength() {
    const password = document.getElementById('newPassword').value;
    const strengthBar = document.querySelector('.strength-bar-fill');
    const strengthText = document.querySelector('.strength-text');
    
    // 重置提示样式
    document.querySelectorAll('.tips-list li').forEach(li => {
        li.className = 'text-muted';
    });
    
    if (!password) {
        strengthBar.style.width = '0%';
        strengthBar.className = 'strength-bar-fill';
        strengthText.textContent = '密码强度: 未输入';
        return;
    }
    
    // 检查各项规则
    const hasLength = password.length >= 8;
    const hasUpper = /[A-Z]/.test(password);
    const hasNumber = /[0-9]/.test(password);
    const hasSpecial = /[!@#$%^&*(),.?":{}|<>]/.test(password);
    
    // 更新提示样式
    if (hasLength) document.getElementById('lengthTip').className = 'valid';
    if (hasUpper) document.getElementById('upperTip').className = 'valid';
    if (hasNumber) document.getElementById('numberTip').className = 'valid';
    if (hasSpecial) document.getElementById('specialTip').className = 'valid';
    
    // 计算强度
    let strength = 0;
    if (hasLength) strength += 25;
    if (hasUpper) strength += 25;
    if (hasNumber) strength += 25;
    if (hasSpecial) strength += 25;
    
    // 更新强度条
    strengthBar.style.width = `${strength}%`;
    
    // 更新强度类和文本
    if (strength < 25) {
        strengthBar.className = 'strength-bar-fill bg-danger';
        strengthText.textContent = '密码强度: 非常弱';
    } else if (strength < 50) {
        strengthBar.className = 'strength-bar-fill bg-warning';
        strengthText.textContent = '密码强度: 弱';
    } else if (strength < 75) {
        strengthBar.className = 'strength-bar-fill bg-info';
        strengthText.textContent = '密码强度: 中等';
    } else {
        strengthBar.className = 'strength-bar-fill bg-success';
        strengthText.textContent = '密码强度: 强';
    }
}

/**
 * 加载用户个人资料数据
 */
async function loadProfileData() {
    try {
        // 添加加载状态
        document.getElementById('profileUsername').innerHTML = '<div class="spinner-border spinner-border-sm text-primary" role="status"><span class="visually-hidden">加载中...</span></div>';
        
        const response = await fetch('/api/profile');
        if (response.ok) {
            const data = await response.json();

            // 更新页面上的用户信息
            document.getElementById('profileUsername').textContent = data.username;
            document.getElementById('username').value = data.username;
            document.getElementById('profileEmail').textContent = data.email;
            document.getElementById('email').value = data.email;
            
            // 如果有显示名称则设置
            if (data.display_name) {
                document.getElementById('displayName').value = data.display_name;
            }
            
            // 设置用户角色样式
            const profileRole = document.getElementById('profileRole');
            if (profileRole) {
                profileRole.textContent = data.is_admin ? '管理员' : '普通用户';
                if (data.is_admin) {
                    profileRole.classList.add('admin-badge');
                }
            }

            // 更新顶部导航栏
            updateNavbarInfo(data);
            
            // 添加淡入效果
            document.querySelectorAll('.card').forEach(card => {
                card.classList.add('fade-in-up');
            });
            
            // 延迟加载活动记录，确保用户名已更新
            setTimeout(loadUserActivities, 300);
            
        } else {
            showMessage('profileMessage', '加载个人资料失败', 'danger');
        }
    } catch (error) {
        console.error('获取个人资料出错:', error);
        showMessage('profileMessage', '获取个人资料时发生错误', 'danger');
    }
}

/**
 * 更新导航栏信息
 */
function updateNavbarInfo(userData) {
    const navbarUsername = document.getElementById('navbarUsername');
    if (navbarUsername) {
        navbarUsername.textContent = userData.username;
    }

    const navbarUserRole = document.getElementById('navbarUserRole');
    if (navbarUserRole) {
        navbarUserRole.textContent = userData.is_admin ? '管理员' : '用户';
        navbarUserRole.className = userData.is_admin ? 'badge bg-danger ms-1' : 'badge bg-primary ms-1';
    }
}

/**
 * 更新用户个人资料
 */
async function updateProfile() {
    try {
        // 显示保存按钮加载状态
        const saveBtn = document.querySelector('.save-btn');
        const originalBtnHtml = saveBtn.innerHTML;
        saveBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 保存中...';
        saveBtn.disabled = true;
        
        const email = document.getElementById('email').value;
        const displayName = document.getElementById('displayName')?.value || '';
        const emailNotifications = document.getElementById('emailNotifications')?.checked || false;
        const systemNotifications = document.getElementById('systemNotifications')?.checked || false;

        const response = await fetch('/api/profile/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                email,
                display_name: displayName,
                preferences: {
                    email_notifications: emailNotifications,
                    system_notifications: systemNotifications
                }
            })
        });

        const data = await response.json();

        if (response.ok) {
            showMessage('profileMessage', '个人资料更新成功', 'success');
            
            // 使用处理程序
            handleProfileUpdated();
        } else {
            showMessage('profileMessage', data.error || '更新个人资料失败', 'danger');
        }
        
        // 恢复按钮状态
        setTimeout(() => {
            saveBtn.innerHTML = originalBtnHtml;
            saveBtn.disabled = false;
        }, 1000);
        
    } catch (error) {
        console.error('更新个人资料出错:', error);
        showMessage('profileMessage', '更新个人资料时发生错误', 'danger');
        
        // 恢复按钮状态
        const saveBtn = document.querySelector('.save-btn');
        if (saveBtn) {
            saveBtn.innerHTML = '<i class="bi bi-save me-1"></i> 保存更改';
            saveBtn.disabled = false;
        }
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
    
    // 检查密码强度
    const strengthBar = document.querySelector('.strength-bar-fill');
    const strengthWidth = parseFloat(strengthBar.style.width);
    if (strengthWidth < 50) {
        showMessage('passwordMessage', '密码强度不足，请创建更强的密码', 'warning');
        return;
    }

    try {
        // 显示按钮加载状态
        const btn = document.getElementById('changePasswordBtn');
        const originalBtnHtml = btn.innerHTML;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 处理中...';
        btn.disabled = true;
        
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
            
            // 添加成功动画
            document.querySelector('.modal-content').classList.add('password-success');
            
            setTimeout(() => {
                // 关闭模态框
                const modal = bootstrap.Modal.getInstance(document.getElementById('changePasswordModal'));
                if (modal) {
                    modal.hide();
                    
                    // 清空密码字段
                    document.getElementById('currentPassword').value = '';
                    document.getElementById('newPassword').value = '';
                    document.getElementById('confirmPassword').value = '';
                    
                    // 隐藏消息和重置强度条
                    document.getElementById('passwordMessage').classList.add('d-none');
                    document.querySelector('.strength-bar-fill').style.width = '0%';
                    document.querySelector('.strength-text').textContent = '密码强度: 未输入';
                    
                    // 重置提示样式
                    document.querySelectorAll('.tips-list li').forEach(li => {
                        li.className = 'text-muted';
                    });
                    
                    // 使用处理程序
                    handlePasswordChanged();
                }
            }, 1500);
            
        } else {
            showMessage('passwordMessage', data.error || '密码修改失败', 'danger');
        }
        
        // 恢复按钮状态
        setTimeout(() => {
            btn.innerHTML = originalBtnHtml;
            btn.disabled = false;
        }, 1000);
        
    } catch (error) {
        console.error('修改密码出错:', error);
        showMessage('passwordMessage', '修改密码时发生错误', 'danger');
        
        // 恢复按钮状态
        const btn = document.getElementById('changePasswordBtn');
        if (btn) {
            btn.innerHTML = '<i class="bi bi-check-lg me-1"></i> 确认修改';
            btn.disabled = false;
        }
    }
}

/**
 * 添加活动到时间线
 */
function addTimelineActivity(title, time, description) {
    const timeline = document.querySelector('.timeline');
    if (!timeline) return;
    
    // 移除空活动提示
    const emptyNotice = document.querySelector('.timeline-empty');
    if (emptyNotice) {
        emptyNotice.remove();
    }
    
    // 创建新的时间线项
    const timelineItem = document.createElement('div');
    timelineItem.className = 'timeline-item';
    
    // 随机选择一个颜色类
    const colorClasses = ['bg-primary', 'bg-success', 'bg-info', 'bg-warning'];
    const randomColor = colorClasses[Math.floor(Math.random() * colorClasses.length)];
    
    // 根据活动类型选择图标
    let icon = 'bi-bell';
    if (title.includes('更新')) icon = 'bi-pencil';
    if (title.includes('密码')) icon = 'bi-shield-lock';
    if (title.includes('登录')) icon = 'bi-box-arrow-in-right';
    
    // 构建HTML
    timelineItem.innerHTML = `
        <div class="timeline-marker ${randomColor}">
            <i class="bi ${icon}"></i>
        </div>
        <div class="timeline-content">
            <div class="timeline-header">
                <div class="timeline-title">${title}</div>
                <div class="timeline-time">${time}</div>
            </div>
            <div class="timeline-body">
                ${description}
            </div>
        </div>
    `;
    
    // 添加到时间线开头
    timeline.insertBefore(timelineItem, timeline.firstChild);
    
    // 添加动画
    timelineItem.classList.add('timeline-item-new');
    setTimeout(() => {
        timelineItem.classList.remove('timeline-item-new');
    }, 500);
    
    // 保存到本地存储
    saveUserActivity({
        title,
        time,
        description,
        icon,
        color: randomColor,
        timestamp: new Date().getTime()
    });
}

/**
 * 保存用户活动到本地存储
 */
function saveUserActivity(activity) {
    try {
        // 获取现有活动
        const username = document.getElementById('profileUsername').textContent;
        const storageKey = `user_activities_${username}`;
        const existingActivities = JSON.parse(localStorage.getItem(storageKey)) || [];
        
        // 添加新活动并限制最大数量为10
        existingActivities.unshift(activity);
        if (existingActivities.length > 10) {
            existingActivities.pop();
        }
        
        // 更新存储
        localStorage.setItem(storageKey, JSON.stringify(existingActivities));
    } catch (error) {
        console.error('保存活动记录出错:', error);
    }
}

/**
 * 加载用户活动记录
 */
function loadUserActivities() {
    try {
        const timeline = document.getElementById('userTimeline');
        if (!timeline) return;
        
        // 清除现有内容
        timeline.innerHTML = '';
        
        // 从localStorage获取活动
        const username = document.getElementById('profileUsername').textContent;
        if (username && username !== '加载中...') {
            const storageKey = `user_activities_${username}`;
            const activities = JSON.parse(localStorage.getItem(storageKey)) || [];
            
            if (activities.length === 0) {
                // 显示空活动提示
                timeline.innerHTML = `
                    <div class="timeline-empty text-center py-4 text-muted">
                        <i class="bi bi-clock-history d-block mb-2" style="font-size: 2rem;"></i>
                        暂无活动记录
                    </div>
                `;
                return;
            }
            
            // 显示活动记录
            activities.forEach(activity => {
                const timelineItem = document.createElement('div');
                timelineItem.className = 'timeline-item';
                
                timelineItem.innerHTML = `
                    <div class="timeline-marker ${activity.color}">
                        <i class="bi ${activity.icon}"></i>
                    </div>
                    <div class="timeline-content">
                        <div class="timeline-header">
                            <div class="timeline-title">${activity.title}</div>
                            <div class="timeline-time">${activity.time}</div>
                        </div>
                        <div class="timeline-body">
                            ${activity.description}
                        </div>
                    </div>
                `;
                
                timeline.appendChild(timelineItem);
            });
        } else {
            // 如果用户名还在加载中，延迟再次尝试
            setTimeout(loadUserActivities, 500);
        }
    } catch (error) {
        console.error('加载活动记录出错:', error);
    }
}

/**
 * 更新用户个人资料后处理
 */
function handleProfileUpdated() {
    // 添加新活动到时间线
    addTimelineActivity('个人资料更新', '刚刚', '您更新了个人资料信息');
    
    // 重新加载个人资料数据
    loadProfileData();
}

/**
 * 修改密码成功后处理
 */
function handlePasswordChanged() {
    // 添加新活动到时间线
    addTimelineActivity('密码修改', '刚刚', '您成功修改了账户密码');
    
    // 其他处理逻辑...
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

        // 添加动画效果
        messageElement.classList.add('alert-animated');
        
        // 成功消息3秒后自动隐藏
        if (type === 'success') {
            setTimeout(() => {
                messageElement.classList.add('alert-fade');
                setTimeout(() => {
                    messageElement.classList.add('d-none');
                    messageElement.classList.remove('alert-fade', 'alert-animated');
                }, 300);
            }, 3000);
        }
    }
}

// 添加CSS动画
document.head.insertAdjacentHTML('beforeend', `
<style>
    .alert-animated {
        animation: alertIn 0.3s ease-out forwards;
    }
    
    .alert-fade {
        animation: alertOut 0.3s ease-in forwards;
    }
    
    .fade-in-up {
        animation: fadeInUp 0.5s ease-out forwards;
    }
    
    .timeline-item-new {
        animation: fadeInLeft 0.5s ease-out forwards;
    }
    
    .focused {
        box-shadow: 0 0 0 0.25rem rgba(78, 115, 223, 0.25) !important;
    }
    
    .password-success {
        animation: successPulse 0.5s ease-in-out;
    }
    
    @keyframes alertIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes alertOut {
        from { opacity: 1; transform: translateY(0); }
        to { opacity: 0; transform: translateY(-10px); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes successPulse {
        0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.5); }
        50% { box-shadow: 0 0 0 15px rgba(40, 167, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
    }
</style>
`);
