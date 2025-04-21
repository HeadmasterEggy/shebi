/**
 * 认证相关功能
 */

// 用户相关全局变量
let currentUser = null;
let isAdmin = false;

// 添加回调数组，用于在用户信息加载后执行
const userInfoCallbacks = [];

/**
 * 添加用户信息加载后的回调函数
 * @param {Function} callback - 回调函数
 */
function onUserInfoLoaded(callback) {
    if (currentUser !== null) {
        // 如果用户信息已加载，直接执行回调
        callback(currentUser, isAdmin);
    } else {
        // 否则添加到回调数组中
        userInfoCallbacks.push(callback);
    }
}

/**
 * 获取当前登录用户信息
 */
async function fetchUserInfo() {
    try {
        console.log('正在获取用户信息...');
        const response = await fetch('/api/user');
        if (response.ok) {
            const data = await response.json();
            currentUser = data.username;
            isAdmin = data.is_admin;
            console.log('用户信息加载成功:', currentUser, '管理员:', isAdmin);
            updateUserInterface();
            
            // 执行所有注册的回调
            userInfoCallbacks.forEach(callback => callback(currentUser, isAdmin));
        } else if (response.status === 401) {
            console.error('用户未登录，将重定向到登录页面');
            // 用户未登录，重定向到登录页面
            window.location.href = '/auth/login';
        } else {
            console.error('获取用户信息失败，状态码:', response.status);
        }
    } catch (error) {
        console.error('获取用户信息失败:', error);
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
    
    // 根据用户角色显示或隐藏管理员功能
    const adminElements = document.querySelectorAll('.admin-only');
    adminElements.forEach(el => {
        el.style.display = isAdmin ? 'block' : 'none';
    });
    
    // 如果是管理员，初始化管理员功能
    if (isAdmin && typeof initAdminFeatures === 'function') {
        initAdminFeatures();
    }
}

/**
 * 处理注销操作
 */
function logout() {
    window.location.href = '/auth/logout';
}

/**
 * 页面加载时获取用户信息和绑定事件
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM加载完成，获取用户信息');
    fetchUserInfo();
    
    // 绑定顶部导航栏按钮事件
    const navbarLogoutBtn = document.getElementById('navbarLogoutBtn');
    if (navbarLogoutBtn) {
        navbarLogoutBtn.addEventListener('click', logout);
    }
    
    // 绑定个人资料和设置菜单项
    const profileMenuItem = document.getElementById('navbarProfile');
    if (profileMenuItem) {
        profileMenuItem.addEventListener('click', function(e) {
            e.preventDefault();
            window.location.href = '/profile';
        });
    }
    
    const settingsMenuItem = document.getElementById('navbarSettings');
    if (settingsMenuItem) {
        settingsMenuItem.addEventListener('click', function(e) {
            e.preventDefault();
            window.location.href = '/settings';
        });
    }
});
