/**
 * 文件上传相关功能
 */

// 当前上传的文件内容
let currentFileContent = '';
let currentFileName = '';

// 历史文件记录
const MAX_FILE_HISTORY = 10;

/**
 * 初始化文件上传相关事件监听
 */
function initFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const fileViewBtn = document.getElementById('fileViewBtn');
    const fileInfo = document.getElementById('fileInfo');
    const filePreview = document.getElementById('filePreview');
    const fileContent = document.getElementById('fileContent');
    const closePreview = document.getElementById('closePreview');
    
    // 添加历史文件按钮
    const fileHistoryBtn = document.createElement('button');
    fileHistoryBtn.type = 'button';
    fileHistoryBtn.className = 'btn btn-outline-secondary ms-2';
    fileHistoryBtn.innerHTML = '<i class="bi bi-clock-history me-1"></i> 历史文件';
    fileHistoryBtn.addEventListener('click', showFileHistory);
    
    // 将历史文件按钮添加到界面
    const fileControls = document.querySelector('.file-controls');
    if (fileControls) {
        fileControls.appendChild(fileHistoryBtn);
    } else {
        const fileInputGroup = fileInput.parentElement;
        if (fileInputGroup && fileInputGroup.parentElement) {
            // 创建控制按钮区域
            const controlsDiv = document.createElement('div');
            controlsDiv.className = 'file-controls d-flex mt-2';
            controlsDiv.appendChild(fileHistoryBtn);
            fileInputGroup.parentElement.appendChild(controlsDiv);
        }
    }

    // 监听文件选择事件
    fileInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (!file) {
            fileViewBtn.disabled = true;
            fileInfo.textContent = '';
            currentFileContent = '';
            currentFileName = '';
            return;
        }

        if (!file.name.toLowerCase().endsWith('.txt')) {
            showError('请上传 .txt 格式的文本文件');
            fileInput.value = '';
            return;
        }

        currentFileName = file.name;
        fileInfo.textContent = `已选择: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
        fileViewBtn.disabled = false;

        // 读取文件内容
        const reader = new FileReader();
        reader.onload = function (e) {
            currentFileContent = e.target.result;
            
            // 保存到历史记录
            saveFileToHistory(file.name, currentFileContent);
        };
        reader.onerror = function () {
            showError('读取文件失败，请重试');
            currentFileContent = '';
        };
        reader.readAsText(file, 'UTF-8');
    });

    // 预览文件内容
    fileViewBtn.addEventListener('click', function () {
        if (!currentFileContent) {
            return;
        }

        fileContent.textContent = currentFileContent;
        filePreview.classList.remove('d-none');
    });

    // 关闭预览
    closePreview.addEventListener('click', function () {
        filePreview.classList.add('d-none');
    });
}

/**
 * 获取当前登录用户名
 * @returns {string} 当前用户名或默认值"guest"
 */
function getCurrentUsername() {
    // 首先尝试从页面元素获取用户名
    const navbarUsername = document.getElementById('navbarUsername');
    if (navbarUsername && navbarUsername.textContent.trim()) {
        return navbarUsername.textContent.trim();
    }
    
    // 其次尝试从localStorage获取
    const authUser = localStorage.getItem('authUser');
    try {
        if (authUser) {
            const userData = JSON.parse(authUser);
            if (userData.username) {
                return userData.username;
            }
        }
    } catch (e) {
        console.warn('解析认证用户数据失败', e);
    }
    
    // 最后尝试从window对象获取全局用户信息
    if (window.currentUser && window.currentUser.username) {
        return window.currentUser.username;
    }
    
    // 如果都获取不到，返回默认值
    return "guest";
}

/**
 * 保存文件到历史记录
 * @param {string} fileName - 文件名称
 * @param {string} fileContent - 文件内容
 */
function saveFileToHistory(fileName, fileContent) {
    try {
        // 获取当前用户名，用于构建存储键
        const username = getCurrentUsername();
        const storageKey = `fileHistory_${username}`;
        
        // 获取现有历史记录
        let fileHistory = JSON.parse(localStorage.getItem(storageKey) || '[]');
        
        // 检查是否已存在相同文件名的记录
        const existingIndex = fileHistory.findIndex(item => item.name === fileName);
        if (existingIndex !== -1) {
            // 如果存在，更新内容和时间戳
            fileHistory[existingIndex].content = fileContent;
            fileHistory[existingIndex].timestamp = Date.now();
        } else {
            // 如果不存在，添加新记录
            fileHistory.unshift({
                name: fileName,
                content: fileContent,
                timestamp: Date.now()
            });
            
            // 保持历史记录不超过最大数量
            if (fileHistory.length > MAX_FILE_HISTORY) {
                fileHistory = fileHistory.slice(0, MAX_FILE_HISTORY);
            }
        }
        
        // 保存到本地存储
        localStorage.setItem(storageKey, JSON.stringify(fileHistory));
    } catch (error) {
        console.error('保存文件历史记录失败:', error);
    }
}

/**
 * 显示历史文件记录
 */
function showFileHistory() {
    try {
        // 获取当前用户名，用于构建存储键
        const username = getCurrentUsername();
        const storageKey = `fileHistory_${username}`;
        
        // 获取历史记录
        const fileHistory = JSON.parse(localStorage.getItem(storageKey) || '[]');
        
        // 如果没有历史记录
        if (fileHistory.length === 0) {
            showToast('没有历史文件记录', 'info');
            return;
        }
        
        // 创建模态框内容
        let modalContent = '';
        fileHistory.forEach((file, index) => {
            const date = new Date(file.timestamp);
            const formattedDate = `${date.getFullYear()}-${(date.getMonth()+1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
            
            modalContent += `
                <div class="history-item" data-index="${index}">
                    <div class="history-item-header">
                        <span class="history-item-title">${file.name}</span>
                        <span class="history-item-date">${formattedDate}</span>
                    </div>
                    <div class="history-item-preview">${file.content.substring(0, 100)}${file.content.length > 100 ? '...' : ''}</div>
                    <div class="history-item-actions">
                        <button type="button" class="btn btn-sm btn-primary load-history-file" data-index="${index}">
                            <i class="bi bi-upload"></i> 加载该文件
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-danger delete-history-file" data-index="${index}">
                            <i class="bi bi-trash"></i> 删除
                        </button>
                    </div>
                </div>
            `;
        });
        
        // 创建模态框
        const modal = createModal('历史文件记录', modalContent);
        
        // 显示模态框
        document.body.appendChild(modal);
        const modalInstance = new bootstrap.Modal(modal);
        modalInstance.show();
        
        // 添加事件监听
        modal.querySelectorAll('.load-history-file').forEach(btn => {
            btn.addEventListener('click', function() {
                const index = this.getAttribute('data-index');
                const file = fileHistory[index];
                
                // 加载历史文件
                currentFileContent = file.content;
                currentFileName = file.name;
                
                // 更新界面
                const fileInfo = document.getElementById('fileInfo');
                if (fileInfo) {
                    fileInfo.textContent = `已选择: ${file.name}`;
                }
                
                const fileViewBtn = document.getElementById('fileViewBtn');
                if (fileViewBtn) {
                    fileViewBtn.disabled = false;
                }
                
                // 关闭模态框
                modalInstance.hide();
                
                // 显示成功消息
                showToast('已加载历史文件', 'success');
                
                // 清除文件输入框的值，以便可以再次选择同一个文件
                const fileInput = document.getElementById('fileInput');
                if (fileInput) {
                    fileInput.value = '';
                }
            });
        });
        
        modal.querySelectorAll('.delete-history-file').forEach(btn => {
            btn.addEventListener('click', function() {
                const index = this.getAttribute('data-index');
                
                // 删除历史记录
                fileHistory.splice(index, 1);
                localStorage.setItem(storageKey, JSON.stringify(fileHistory));
                
                // 从界面中移除
                const historyItem = this.closest('.history-item');
                historyItem.classList.add('fade-out');
                setTimeout(() => {
                    historyItem.remove();
                    
                    // 如果没有更多历史记录，关闭模态框
                    if (modal.querySelectorAll('.history-item').length === 0) {
                        modalInstance.hide();
                        showToast('没有更多历史文件记录', 'info');
                    }
                }, 300);
            });
        });
    } catch (error) {
        console.error('显示文件历史记录失败:', error);
        showToast('无法加载历史文件记录', 'error');
    }
}

/**
 * 创建模态框
 * @param {string} title - 模态框标题
 * @param {string} content - 模态框内容
 * @returns {HTMLElement} 模态框元素
 */
function createModal(title, content) {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'historyModal';
    modal.tabIndex = '-1';
    modal.setAttribute('aria-labelledby', 'historyModalLabel');
    modal.setAttribute('aria-hidden', 'true');
    
    modal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="historyModalLabel">${title}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body history-modal-body">
                    ${content}
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">关闭</button>
                </div>
            </div>
        </div>
    `;
    
    return modal;
}

/**
 * 获取当前输入的文本（文本框输入或文件上传）
 * @returns {string} 输入文本
 */
function getInputText() {
    // 检查当前激活的输入标签页
    const textTabActive = document.getElementById('text-tab').classList.contains('active');

    if (textTabActive) {
        // 从文本框获取文本
        return document.getElementById('textInput').value.trim();
    } else {
        // 从上传的文件获取文本
        if (!currentFileContent) {
            showError('请先上传文件');
            return '';
        }
        return currentFileContent.trim();
    }
}

/**
 * 解析文件中的句子（每行作为一个句子）
 * @param {string} text 文件文本内容
 * @returns {string[]} 句子数组
 */
function parseSentencesFromFile(text) {
    if (!text) return [];

    // 按行分割，过滤空行
    return text.split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);
}

// 将文件功能暴露给全局，以便与其他模块集成
window.fileUploadModule = {
    getInputText,
    parseSentencesFromFile,
    getCurrentFileName: () => currentFileName,
    getCurrentFileContent: () => currentFileContent,
    showFileHistory,
    getCurrentUsername   // 导出获取用户名的函数供其他模块使用
};
