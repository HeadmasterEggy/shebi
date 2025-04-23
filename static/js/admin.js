/**
 * 管理员功能模块
 */

// 操作历史记录栈 - 用于撤销功能
let operationHistory = [];
const MAX_HISTORY_SIZE = 10; // 限制历史记录最大数量

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function () {
    console.log('管理员模块已加载');

    // 检查用户是否为管理员
    if (typeof onUserInfoLoaded === 'function') {
        onUserInfoLoaded(function (username, isAdmin) {
            if (isAdmin) {
                console.log('管理员用户已登录，初始化管理功能');
                initAdminFunctions();
            } else {
                console.log('非管理员用户，不初始化管理功能');
            }
        });
    }
});

/**
 * 初始化管理员功能
 */
function initAdminFunctions() {
    // 初始化用户管理界面
    setupUserManagement();

    // 立即加载用户列表
    fetchUsers();
}

/**
 * 设置用户管理界面
 */
function setupUserManagement() {
    const adminSection = document.getElementById('admin-section');
    if (!adminSection) {
        console.error('找不到admin-section元素');
        return;
    }

    console.log('正在设置用户管理界面');

    // 添加用户管理卡片
    const userManagementCard = adminSection.querySelector('.card');
    if (userManagementCard) {
        userManagementCard.innerHTML = `
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0">用户管理</h5>
                <div>
                    <button class="btn btn-outline-secondary btn-sm me-2" id="undoOperationBtn" disabled>
                        <i class="bi bi-arrow-counterclockwise"></i> 撤销操作
                    </button>
                    <button class="btn btn-primary btn-sm" id="createUserBtn">
                        <i class="bi bi-person-plus"></i> 添加用户
                    </button>
                </div>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>用户名</th>
                                <th>邮箱</th>
                                <th>角色</th>
                                <th>操作</th>
                            </tr>
                        </thead>
                        <tbody id="usersTableBody">
                            <tr>
                                <td colspan="5" class="text-center">加载中...</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    } else {
        console.error('找不到用户管理卡片元素');
    }

    // 添加创建用户表单
    const createUserForm = document.getElementById('createUserForm');
    if (createUserForm) {
        createUserForm.innerHTML = `
            <div class="mb-3">
                <label for="newUsername" class="form-label">用户名</label>
                <input type="text" class="form-control" id="newUsername" required>
            </div>
            <div class="mb-3">
                <label for="newEmail" class="form-label">电子邮箱</label>
                <input type="email" class="form-control" id="newEmail" required>
            </div>
            <div class="mb-3">
                <label for="newPassword" class="form-label">密码</label>
                <input type="password" class="form-control" id="newPassword" required>
            </div>
            <div class="mb-3 form-check">
                <input type="checkbox" class="form-check-input" id="newIsAdmin">
                <label class="form-check-label" for="newIsAdmin">管理员权限</label>
            </div>
        `;
    } else {
        console.error('找不到createUserForm元素');
    }

    // 添加编辑用户表单
    const editUserForm = document.getElementById('editUserForm');
    if (editUserForm) {
        editUserForm.innerHTML = `
            <input type="hidden" id="editUserId">
            <input type="hidden" id="editOriginalData">
            <div class="mb-3">
                <label for="editUsername" class="form-label">用户名</label>
                <input type="text" class="form-control" id="editUsername" disabled>
            </div>
            <div class="mb-3">
                <label for="editEmail" class="form-label">电子邮箱</label>
                <input type="email" class="form-control" id="editEmail" required>
            </div>
            <div class="mb-3">
                <label for="editPassword" class="form-label">新密码（留空表示不修改）</label>
                <input type="password" class="form-control" id="editPassword">
            </div>
            <div class="mb-3 form-check">
                <input type="checkbox" class="form-check-input" id="editIsAdmin">
                <label class="form-check-label" for="editIsAdmin">管理员权限</label>
            </div>
        `;
    } else {
        console.error('找不到editUserForm元素');
    }

    // 一定要在创建完DOM元素后再绑定事件
    setTimeout(() => {
        bindAdminEvents();
    }, 100);
}

/**
 * 绑定管理员界面的所有事件
 */
function bindAdminEvents() {
    console.log('正在绑定管理员界面事件');

    // 绑定创建用户按钮事件
    const createUserBtn = document.getElementById('createUserBtn');
    if (createUserBtn) {
        console.log('找到创建用户按钮，添加点击事件');
        createUserBtn.addEventListener('click', function () {
            showCreateUserModal();
        });
    } else {
        console.error('找不到创建用户按钮');
    }

    // 绑定保存用户按钮事件
    const saveUserBtn = document.getElementById('saveUserBtn');
    if (saveUserBtn) {
        console.log('找到保存用户按钮，添加点击事件');
        saveUserBtn.addEventListener('click', function () {
            createUser();
        });
    } else {
        console.error('找不到保存用户按钮');
    }

    // 绑定更新用户按钮事件
    const updateUserBtn = document.getElementById('updateUserBtn');
    if (updateUserBtn) {
        console.log('找到更新用户按钮，添加点击事件');
        updateUserBtn.addEventListener('click', function () {
            updateUser();
        });
    } else {
        console.error('找不到更新用户按钮');
    }

    // 绑定确认删除按钮事件
    const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
    if (confirmDeleteBtn) {
        console.log('找到确认删除按钮，添加点击事件');
        confirmDeleteBtn.addEventListener('click', function () {
            deleteUser();
        });
    } else {
        console.error('找不到确认删除按钮');
    }

    // 绑定撤销操作按钮
    const undoBtn = document.getElementById('undoOperationBtn');
    if (undoBtn) {
        console.log('找到撤销操作按钮，添加点击事件');
        undoBtn.addEventListener('click', function() {
            undoLastOperation();
        });
    } else {
        console.error('找不到撤销操作按钮');
    }
}

/**
 * 获取所有用户列表
 */
async function fetchUsers() {
    console.log('正在获取用户列表...');
    try {
        const response = await fetch('/api/admin/users');
        if (response.ok) {
            const data = await response.json();
            console.log('用户列表获取成功:', data);
            displayUsers(data.users || []);
        } else {
            console.error('获取用户列表失败, 状态码:', response.status);
            try {
                const errorData = await response.json();
                console.error('错误详情:', errorData);
            } catch (e) {
                console.error('无法解析错误响应');
            }

            const tableBody = document.getElementById('usersTableBody');
            if (tableBody) {
                tableBody.innerHTML = `
                    <tr>
                        <td colspan="5" class="text-center text-danger">获取用户列表失败 (${response.status})</td>
                    </tr>
                `;
            }
        }
    } catch (error) {
        console.error('获取用户列表出错:', error);
        const tableBody = document.getElementById('usersTableBody');
        if (tableBody) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="5" class="text-center text-danger">获取用户列表出错: ${error.message}</td>
                </tr>
            `;
        }
    }
}

/**
 * 显示用户列表
 */
function displayUsers(users) {
    const tableBody = document.getElementById('usersTableBody');
    if (!tableBody) return;

    if (users.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="5" class="text-center">暂无用户数据</td>
            </tr>
        `;
        return;
    }

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
                    <div class="btn-group btn-group-sm" role="group">
                        <button type="button" class="btn btn-outline-primary" onclick="editUser(${user.id})">
                            <i class="bi bi-pencil"></i>
                        </button>
                        <button type="button" class="btn btn-outline-danger" onclick="confirmDeleteUser(${user.id}, '${user.username}')">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `;
    });

    tableBody.innerHTML = html;
}

/**
 * 显示创建用户模态框
 */
function showCreateUserModal() {
    console.log('显示创建用户模态框');
    try {
        // 清空表单
        const form = document.getElementById('createUserForm');
        if (form) {
            form.reset();
        } else {
            console.error('找不到创建用户表单');
        }

        // 获取模态框元素
        const modalElement = document.getElementById('createUserModal');
        if (!modalElement) {
            console.error('找不到模态框元素 #createUserModal');
            alert('界面错误: 找不到创建用户的对话框');
            return;
        }

        // 显示模态框
        try {
            // 尝试使用不同方法显示模态框
            if (typeof bootstrap !== 'undefined') {
                const modal = new bootstrap.Modal(modalElement);
                modal.show();
                console.log('模态框已显示 (bootstrap)');
            } else if ($(modalElement).modal) {
                $(modalElement).modal('show');
                console.log('模态框已显示 (jQuery)');
            } else {
                // 回退方案：直接设置样式
                modalElement.style.display = 'block';
                modalElement.classList.add('show');
                document.body.classList.add('modal-open');
                console.log('模态框已显示 (手动)');
            }
        } catch (modalError) {
            console.error('显示模态框失败:', modalError);
            alert('无法显示创建用户对话框: ' + modalError.message);
        }
    } catch (error) {
        console.error('显示创建用户模态框出错:', error);
        alert('无法打开创建用户窗口: ' + error.message);
    }
}

/**
 * 创建用户
 */
async function createUser() {
    try {
        const username = document.getElementById('newUsername').value.trim();
        const email = document.getElementById('newEmail').value.trim();
        const password = document.getElementById('newPassword').value;
        const isAdmin = document.getElementById('newIsAdmin').checked;

        // 表单验证
        if (!username || !email || !password) {
            alert('请填写所有必填字段');
            return;
        }

        // 发送创建用户请求
        const response = await fetch('/api/admin/create_user', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username,
                email,
                password,
                is_admin: isAdmin
            })
        });

        const data = await response.json();

        if (response.ok) {
            // 记录操作以便撤销
            recordOperation('create', {
                userId: data.user_id,
                username: username
            });
            
            // 关闭模态框
            const modalElement = document.getElementById('createUserModal');
            const modal = bootstrap.Modal.getInstance(modalElement);
            modal.hide();

            // 提示创建成功
            alert('用户创建成功！');

            // 重新加载用户列表
            fetchUsers();
        } else {
            alert('创建用户失败: ' + (data.error || '未知错误'));
        }
    } catch (error) {
        console.error('创建用户出错:', error);
        alert('创建用户时发生错误: ' + error.message);
    }
}

/**
 * 编辑用户
 */
async function editUser(userId) {
    console.log('编辑用户:', userId);
    try {
        // 尝试多种可能的API端点路径，找到正确的一个
        const possibleEndpoints = [
            `/api/admin/user/${userId}`,
            `/api/admin/users/${userId}`,
            `/api/users/${userId}`,
            `/api/admin/get_user/${userId}`
        ];

        console.log('尝试以下API端点:');
        console.log(possibleEndpoints);

        // 依次尝试各个端点
        let response = null;
        let endpointUsed = '';

        for (const endpoint of possibleEndpoints) {
            console.log(`尝试API端点: ${endpoint}`);
            try {
                const resp = await fetch(endpoint, {
                    headers: {
                        'Accept': 'application/json',
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                });

                console.log(`端点 ${endpoint} 返回状态: ${resp.status}`);

                if (resp.ok) {
                    response = resp;
                    endpointUsed = endpoint;
                    console.log(`成功: 使用端点 ${endpoint}`);
                    break;
                }
            } catch (e) {
                console.log(`端点 ${endpoint} 请求失败:`, e);
            }
        }

        // 如果所有端点都失败，使用第一个端点的响应，以便显示错误信息
        if (!response) {
            response = await fetch(possibleEndpoints[0], {
                headers: {
                    'Accept': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });
        }

        console.log('最终使用的API端点:', endpointUsed || possibleEndpoints[0]);
        console.log('获取用户信息响应状态:', response.status);

        if (response.ok) {
            const user = await response.json();
            console.log('获取到用户数据:', user);

            // 存储原始用户数据用于可能的撤销
            document.getElementById('editOriginalData').value = JSON.stringify(user);
            
            // 填充表单
            document.getElementById('editUserId').value = user.id;
            document.getElementById('editUsername').value = user.username;
            document.getElementById('editEmail').value = user.email || '';
            document.getElementById('editPassword').value = '';  // 不显示密码
            document.getElementById('editIsAdmin').checked = !!user.is_admin;

            // 显示模态框
            const modalElement = document.getElementById('editUserModal');
            if (!modalElement) {
                console.error('找不到编辑用户模态框');
                alert('界面错误: 找不到编辑用户对话框');
                return;
            }

            try {
                // 尝试不同方法显示模态框
                if (typeof bootstrap !== 'undefined') {
                    const modal = new bootstrap.Modal(modalElement);
                    modal.show();
                    console.log('编辑模态框已显示 (bootstrap)');
                } else {
                    // 回退方案
                    modalElement.style.display = 'block';
                    modalElement.classList.add('show');
                    document.body.classList.add('modal-open');
                    console.log('编辑模态框已显示 (手动)');
                }
            } catch (modalError) {
                console.error('显示编辑模态框失败:', modalError);
                alert('无法显示编辑用户对话框');
            }
        } else {
            // 增强错误处理，显示可能的API路径
            console.error('获取用户数据失败:', response.status);
            console.error('尝试使用的API端点:', endpointUsed || possibleEndpoints[0]);
            const responseText = await response.text();
            console.error('错误响应内容:', responseText);

            alert(`获取用户信息失败: 服务器返回状态 ${response.status}。请联系管理员检查API端点。`);
            alert(`尝试使用API端点: ${possibleEndpoints.join(', ')}`);

            // 获取API信息以辅助调试
            try {
                const infoResponse = await fetch('/api/info');
                if (infoResponse.ok) {
                    const apiInfo = await infoResponse.json();
                    console.log('API信息:', apiInfo);
                } else {
                    console.log('无法获取API信息, 状态码:', infoResponse.status);
                }
            } catch (e) {
                console.log('获取API信息时出错:', e);
            }
        }
    } catch (error) {
        console.error('编辑用户出错:', error);
        alert('获取用户信息时发生错误: ' + error.message);
    }
}

/**
 * 确认删除用户
 */
function confirmDeleteUser(userId, username) {
    console.log('确认删除用户:', userId, username);

    // 先获取用户完整数据用于可能的恢复
    fetch(`/api/admin/user/${userId}`)
        .then(response => response.json())
        .then(userData => {
            // 存储用户数据到删除确认框
            const userData_element = document.getElementById('deleteUserData');
            if (userData_element) {
                userData_element.value = JSON.stringify(userData);
            }
            
            // 设置要删除的用户ID
            document.getElementById('deleteUserId').value = userId;

            // 设置确认消息
            const confirmMsg = document.querySelector('#deleteConfirmModal .modal-body p');
            if (confirmMsg) {
                confirmMsg.textContent = `确定要删除用户 "${username}" 吗？此操作可以通过撤销按钮恢复。`;
            }

            // 显示确认对话框
            const modalElement = document.getElementById('deleteConfirmModal');
            const modal = new bootstrap.Modal(modalElement);
            modal.show();
        })
        .catch(error => {
            console.error('获取用户数据失败:', error);
            alert('获取用户数据失败，无法安全删除');
        });
}

/**
 * 删除用户
 */
async function deleteUser() {
    try {
        const userId = document.getElementById('deleteUserId').value;
        const userDataStr = document.getElementById('deleteUserData').value;
        let userData = {};
        
        try {
            userData = JSON.parse(userDataStr);
        } catch (e) {
            console.error('解析用户数据失败:', e);
        }

        // 尝试多种可能的API端点路径
        const possibleEndpoints = [
            `/api/admin/delete_user/${userId}`,
            `/api/admin/users/${userId}`,
            `/api/admin/user/${userId}`
        ];

        console.log('尝试以下删除API端点:');
        console.log(possibleEndpoints);

        // 依次尝试各个端点
        let response = null;
        let endpointUsed = '';

        for (const endpoint of possibleEndpoints) {
            console.log(`尝试删除API端点: ${endpoint}`);
            try {
                const resp = await fetch(endpoint, {
                    method: 'DELETE',
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                });

                console.log(`删除端点 ${endpoint} 返回状态: ${resp.status}`);

                if (resp.ok) {
                    response = resp;
                    endpointUsed = endpoint;
                    console.log(`成功: 使用删除端点 ${endpoint}`);
                    break;
                }
            } catch (e) {
                console.log(`删除端点 ${endpoint} 请求失败:`, e);
            }
        }

        // 如果所有端点都失败，使用第一个端点的响应
        if (!response) {
            response = await fetch(possibleEndpoints[0], {
                method: 'DELETE',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });
        }

        console.log('最终使用的删除API端点:', endpointUsed || possibleEndpoints[0]);
        console.log('删除用户响应状态:', response.status);

        if (response.ok) {
            // 记录操作用于撤销
            recordOperation('delete', {
                userId: userId,
                userData: {
                    username: userData.username,
                    email: userData.email,
                    password: "tempPassword123", // 临时密码
                    is_admin: userData.is_admin
                }
            });
            
            // 关闭模态框
            const modalElement = document.getElementById('deleteConfirmModal');
            const modal = bootstrap.Modal.getInstance(modalElement);
            if (modal) {
                modal.hide();
            } else {
                modalElement.classList.remove('show');
                modalElement.style.display = 'none';
                document.body.classList.remove('modal-open');
            }

            // 提示删除成功
            alert('用户已成功删除！');

            // 重新加载用户列表
            fetchUsers();
        } else {
            const responseText = await response.text();
            console.error('删除用户失败响应:', responseText);
            let data = {};
            try {
                data = responseText ? JSON.parse(responseText) : {};
            } catch (e) {
            }
            alert('删除用户失败: ' + (data.error || `服务器返回状态 ${response.status}`));
        }
    } catch (error) {
        console.error('删除用户出错:', error);
        alert('删除用户时发生错误: ' + error.message);
    }
}

/**
 * 更新用户信息
 */
async function updateUser() {
    try {
        const userId = document.getElementById('editUserId').value;
        const username = document.getElementById('editUsername').value;
        const email = document.getElementById('editEmail').value.trim();
        const password = document.getElementById('editPassword').value;
        const isAdmin = document.getElementById('editIsAdmin').checked;
        
        // 获取原始数据用于撤销
        const originalDataStr = document.getElementById('editOriginalData').value;
        let originalData = {};
        
        try {
            originalData = JSON.parse(originalDataStr);
        } catch (e) {
            console.error('解析原始用户数据失败:', e);
        }

        // 表单验证
        if (!email) {
            alert('请填写电子邮箱');
            return;
        }

        // 准备请求体
        const requestBody = {
            email,
            is_admin: isAdmin
        };

        // 如果密码字段不为空，则添加到请求体
        if (password) {
            requestBody.password = password;
        }

        console.log('更新用户请求体:', JSON.stringify(requestBody));

        // 尝试多种可能的API端点路径
        const possibleEndpoints = [
            `/api/admin/update_user/${userId}`,
            `/api/admin/users/${userId}`,
            `/api/admin/user/${userId}`
        ];

        console.log('尝试以下更新API端点:');
        console.log(possibleEndpoints);

        // 依次尝试各个端点
        let response = null;
        let endpointUsed = '';

        for (const endpoint of possibleEndpoints) {
            console.log(`尝试更新API端点: ${endpoint}`);
            try {
                const resp = await fetch(endpoint, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Requested-With': 'XMLHttpRequest'
                    },
                    body: JSON.stringify(requestBody)
                });

                console.log(`更新端点 ${endpoint} 返回状态: ${resp.status}`);

                if (resp.ok) {
                    response = resp;
                    endpointUsed = endpoint;
                    console.log(`成功: 使用更新端点 ${endpoint}`);
                    break;
                }
            } catch (e) {
                console.log(`更新端点 ${endpoint} 请求失败:`, e);
            }
        }

        // 如果所有端点都失败，使用第一个端点的响应
        if (!response) {
            response = await fetch(possibleEndpoints[0], {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify(requestBody)
            });
        }

        console.log('最终使用的更新API端点:', endpointUsed || possibleEndpoints[0]);
        console.log('更新用户响应状态:', response.status);

        const responseText = await response.text();
        console.log('更新用户响应内容:', responseText);

        let data;
        try {
            data = responseText ? JSON.parse(responseText) : {};
        } catch (e) {
            console.error('解析响应JSON出错:', e);
            data = {};
        }

        if (response.ok) {
            // 记录操作用于撤销
            recordOperation('update', {
                userId: userId,
                username: username,
                previousData: {
                    email: originalData.email,
                    is_admin: originalData.is_admin
                }
            });
            
            // 关闭模态框
            const modalElement = document.getElementById('editUserModal');
            const modal = bootstrap.Modal.getInstance(modalElement);
            if (modal) {
                modal.hide();
            } else {
                modalElement.classList.remove('show');
                modalElement.style.display = 'none';
                document.body.classList.remove('modal-open');
            }

            // 提示更新成功
            alert('用户信息已更新！');

            // 重新加载用户列表
            fetchUsers();
        } else {
            alert('更新用户失败: ' + (data.error || `服务器返回状态 ${response.status}`));
        }
    } catch (error) {
        console.error('更新用户出错:', error);
        alert('更新用户信息时发生错误: ' + error.message);
    }
}

/**
 * 记录操作到历史
 * @param {string} type - 操作类型 ('create', 'update', 或 'delete')
 * @param {Object} data - 操作相关数据
 */
function recordOperation(type, data) {
    // 添加到历史记录开头
    operationHistory.unshift({
        type: type,
        data: data,
        timestamp: new Date().toISOString()
    });
    
    // 限制历史记录大小
    if (operationHistory.length > MAX_HISTORY_SIZE) {
        operationHistory.pop();
    }
    
    // 启用撤销按钮
    const undoBtn = document.getElementById('undoOperationBtn');
    if (undoBtn) {
        undoBtn.disabled = false;
    }
    
    console.log(`已记录操作: ${type}`, data);
    console.log('当前历史记录:', operationHistory);
}

/**
 * 撤销最后一次操作
 */
async function undoLastOperation() {
    if (operationHistory.length === 0) {
        console.log('没有可撤销的操作');
        return;
    }
    
    try {
        const operation = operationHistory[0];
        console.log('正在撤销操作:', operation);
        
        switch (operation.type) {
            case 'create':
                await undoCreate(operation.data);
                break;
                
            case 'update':
                await undoUpdate(operation.data);
                break;
                
            case 'delete':
                await undoDelete(operation.data);
                break;
                
            default:
                console.error('未知的操作类型:', operation.type);
                return;
        }
        
        // 从历史中移除已撤销的操作
        operationHistory.shift();
        
        // 如果没有更多操作，禁用撤销按钮
        if (operationHistory.length === 0) {
            const undoBtn = document.getElementById('undoOperationBtn');
            if (undoBtn) {
                undoBtn.disabled = true;
            }
        }
        
        // 刷新用户列表
        fetchUsers();
        
    } catch (error) {
        console.error('撤销操作失败:', error);
        alert('撤销操作失败: ' + error.message);
    }
}

/**
 * 撤销创建用户操作
 */
async function undoCreate(data) {
    if (!data.userId) {
        throw new Error('缺少用户ID，无法撤销创建');
    }
    
    const response = await fetch(`/api/admin/delete_user/${data.userId}`, {
        method: 'DELETE'
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '删除失败');
    }
    
    alert(`已撤销: 创建用户 "${data.username}"`);
}

/**
 * 撤销更新用户操作
 */
async function undoUpdate(data) {
    if (!data.previousData || !data.userId) {
        throw new Error('缺少原始数据，无法撤销修改');
    }
    
    // 使用原始数据恢复用户状态
    const response = await fetch(`/api/admin/update_user/${data.userId}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data.previousData)
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '恢复失败');
    }
    
    alert(`已撤销: 修改用户 "${data.username}" 的信息`);
}

/**
 * 撤销删除用户操作
 */
async function undoDelete(data) {
    if (!data.userData) {
        throw new Error('缺少用户数据，无法恢复删除的用户');
    }
    
    // 重新创建已删除的用户
    const response = await fetch('/api/admin/create_user', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data.userData)
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '恢复失败');
    }
    
    alert(`已恢复: 删除的用户 "${data.userData.username}"`);
}

// 确保全局可访问
window.fetchUsers = fetchUsers;
window.editUser = editUser;
window.confirmDeleteUser = confirmDeleteUser;
window.createUser = createUser;
window.updateUser = updateUser;
window.deleteUser = deleteUser;
window.undoLastOperation = undoLastOperation;
