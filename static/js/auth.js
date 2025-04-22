/**
 * 认证相关功能
 */

// 存储当前用户信息的全局变量
let currentUser = '';
let isAdmin = false;
let userInfoCallbacks = [];

/**
 * 注册用户信息加载完成的回调函数
 * @param {Function} callback - 回调函数，接收用户名和管理员状态作为参数
 */
function onUserInfoLoaded(callback) {
    if (typeof callback === 'function') {
        if (currentUser) {
            // 如果用户信息已加载，直接调用回调
            callback(currentUser, isAdmin);
        } else {
            // 否则添加到回调列表
            userInfoCallbacks.push(callback);
        }
    }
}

/**
 * 获取当前用户信息
 */
async function fetchUserInfo() {
    try {
        const response = await fetch('/api/user');
        if (response.ok) {
            const data = await response.json();
            currentUser = data.username;
            isAdmin = !!data.is_admin; // 确保布尔值

            // 更新界面
            updateUserInterface();

            // 触发所有回调
            userInfoCallbacks.forEach(callback => callback(currentUser, isAdmin));
            userInfoCallbacks = []; // 清空回调列表

            return data;
        } else {
            console.error('获取用户信息失败，可能需要重新登录');
            window.location.href = '/auth/login';
        }
    } catch (error) {
        console.error('获取用户信息出错:', error);
    }
}

/**
 * 更新界面以反映用户登录状态
 */
function updateUserInterface() {
    // 更新顶部导航栏用户信息
    const navbarUsername = document.getElementById('navbarUsername');
    if (navbarUsername) {
        navbarUsername.textContent = currentUser;
    }

    // 更新导航栏用户角色标识
    const navbarUserRole = document.getElementById('navbarUserRole');
    if (navbarUserRole) {
        navbarUserRole.textContent = isAdmin ? '管理员' : '用户';
        navbarUserRole.className = isAdmin ? 'badge bg-danger ms-1' : 'badge bg-primary ms-1';
    }

    // 处理管理员专用菜单项 - 添加visible类而不是直接修改style
    document.querySelectorAll('.menu-item.admin-only').forEach(item => {
        if (isAdmin) {
            item.classList.add('visible');
        } else {
            item.classList.remove('visible');
        }
    });

    // 确保窗口对象可以访问用户状态
    window.isAdmin = isAdmin;
    console.log('用户权限已更新，管理员:', isAdmin);
}

/**
 * 注销功能
 */
async function logout() {
    try {
        const response = await fetch('/auth/logout');
        if (response.ok) {
            window.location.href = '/auth/login';
        } else {
            console.error('注销失败');
        }
    } catch (error) {
        console.error('注销请求出错:', error);
    }
}

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function () {
    // 获取用户信息
    fetchUserInfo();

    // 绑定注销按钮事件
    const logoutBtn = document.getElementById('navbarLogoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', logout);
    }

    // 绑定导航栏个人资料和设置链接
    const navbarProfile = document.getElementById('navbarProfile');
    if (navbarProfile) {
        navbarProfile.addEventListener('click', function (e) {
            e.preventDefault();
            window.location.href = '/profile';
        });
    }

    const navbarSettings = document.getElementById('navbarSettings');
    if (navbarSettings) {
        navbarSettings.addEventListener('click', function (e) {
            e.preventDefault();
            window.location.href = '/settings';
        });
    }
});
