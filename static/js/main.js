/**
 * 主模块，包含程序入口和核心功能
 */

/**
 * 获取可用的模型列表
 */
async function fetchModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();

        if (response.ok && data.models) {
            const modelSelect = document.getElementById('modelSelect');
            const fileModelSelect = document.getElementById('fileModelSelect');
            modelSelect.innerHTML = ''; // 清空现有选项
            fileModelSelect.innerHTML = ''; // 清空文件上传区域的模型选项

            // 添加模型选项到文本输入区域
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                option.dataset.description = model.description;
                modelSelect.appendChild(option);

                // 同样添加到文件上传区域
                const fileOption = option.cloneNode(true);
                fileModelSelect.appendChild(fileOption);
            });

            // 设置默认选中的模型
            if (data.default) {
                modelSelect.value = data.default;
                fileModelSelect.value = data.default;
            }

            // 显示模型描述
            updateModelDescription();
        } else {
            console.error('获取模型列表失败:', data.error || '未知错误');
            const modelSelect = document.getElementById('modelSelect');
            const fileModelSelect = document.getElementById('fileModelSelect');
            modelSelect.innerHTML = '<option value="">加载失败</option>';
            fileModelSelect.innerHTML = '<option value="">加载失败</option>';
        }
    } catch (error) {
        console.error('获取模型列表出错:', error);
        const modelSelect = document.getElementById('modelSelect');
        const fileModelSelect = document.getElementById('fileModelSelect');
        modelSelect.innerHTML = '<option value="">加载失败</option>';
        fileModelSelect.innerHTML = '<option value="">加载失败</option>';
    }
}

/**
 * 更新模型描述信息
 */
function updateModelDescription() {
    const modelSelect = document.getElementById('modelSelect');
    const descriptionElement = document.getElementById('modelDescription');

    if (modelSelect.selectedOptions.length > 0) {
        const selectedOption = modelSelect.selectedOptions[0];
        descriptionElement.textContent = selectedOption.dataset.description || '';
    } else {
        descriptionElement.textContent = '';
    }

    // 更新文件上传区域的模型描述
    const fileModelSelect = document.getElementById('fileModelSelect');
    const fileDescriptionElement = document.getElementById('fileModelDescription');

    if (fileModelSelect.selectedOptions.length > 0) {
        const selectedOption = fileModelSelect.selectedOptions[0];
        fileDescriptionElement.textContent = selectedOption.dataset.description || '';
    } else {
        fileDescriptionElement.textContent = '';
    }
}

/**
 * 加载动画控制器
 */
let loadingDotsInterval;

/**
 * 显示加载动画
 */
function showLoadingAnimation() {
    const loadingDots = document.getElementById('loadingDots');
    let dotsCount = 0;

    // 清除之前的间隔器（如果存在）
    if (loadingDotsInterval) clearInterval(loadingDotsInterval);

    // 启动新的动画间隔器
    loadingDotsInterval = setInterval(() => {
        dotsCount = (dotsCount % 3) + 1;
        loadingDots.textContent = '.'.repeat(dotsCount);
    }, 500);

    document.getElementById('loading').style.display = 'flex';
}

/**
 * 停止加载动画
 */
function stopLoadingAnimation() {
    if (loadingDotsInterval) {
        clearInterval(loadingDotsInterval);
        loadingDotsInterval = null;
    }
    document.getElementById('loading').style.display = 'none';
}

/**
 * 分析文本函数
 */
async function analyzeText() {
    // 获取输入文本（从文本框或文件）
    const text = window.fileUploadModule.getInputText();

    if (!text) {
        showError('请输入要分析的文本或上传文件');
        return;
    }

    showLoadingAnimation();
    document.getElementById('errorMessage').style.display = 'none';

    // 移除直接隐藏结果容器的代码，改为在进行分析前先切换到输入区域
    switchSection('input-section');

    // 检查是否为文件上传模式
    const isFileUpload = document.getElementById('file-tab').classList.contains('active');
    let payload = {text};

    // 获取选择的模型类型
    if (isFileUpload) {
        const fileModelSelect = document.getElementById('fileModelSelect');
        if (fileModelSelect.value) {
            payload.model_type = fileModelSelect.value;
        }
    } else {
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect.value) {
            payload.model_type = modelSelect.value;
        }
    }

    // 如果是文件上传，添加文件名和处理模式标记
    if (isFileUpload) {
        payload.fileName = window.fileUploadModule.getCurrentFileName();
        payload.mode = 'file';

        // 预处理: 检查是否需要将文件内容按行分割为句子
        const lines = window.fileUploadModule.parseSentencesFromFile(text);
        if (lines.length > 1) {
            payload.sentences = lines;
        }
    }

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (response.ok) {
            displayResults(data);
        } else {
            showError(data.error || '分析失败');
        }
    } catch (error) {
        showError('请求失败：' + error.message);
    } finally {
        stopLoadingAnimation();
    }
}

/**
 * 显示分析结果
 * @param {Object} data - 分析结果数据
 */
function displayResults(data) {
    if (!data || !data.sentences || !Array.isArray(data.sentences)) {
        showError('返回的数据格式不正确');
        return;
    }

    // 保存分析数据到全局变量，方便视图切换时使用
    window.lastAnalysisData = data;
    window.allSentences = data.sentences;

    console.log('收到分析结果数据:', data);

    // 检查词频数据并进行预处理
    if (!data.wordFreq || !Array.isArray(data.wordFreq) || data.wordFreq.length === 0) {
        console.warn('未检测到词频数据或词频数据为空，创建占位数据');
        // 创建占位词频数据以避免图表错误
        data.wordFreq = [
            {word: "无词频数据", count: 1}
        ];
    } else {
        console.log(`检测到词频数据，包含${data.wordFreq.length}个词条`);
    }

    // 设置句子数据
    allSentences = data.sentences;
    filteredSentences = [...allSentences];
    currentPage = 1;
    sentimentFilter = 'all';
    displayMode = 'normal';

    // 重置过滤器状态
    document.querySelectorAll('.filter-button[data-sentiment]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.sentiment === 'all');
    });

    document.querySelectorAll('.filter-button[data-display]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.display === 'normal');
    });

    // 确保所有结果卡片都显示出来
    const resultCards = ['overallResult', 'sentenceResults', 'wordFreq', 'modelMetrics'];
    resultCards.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.style.display = 'block';
            element.style.visibility = 'visible';
            element.style.opacity = '1';
        }
    });

    // 确保图表容器可见
    const chartContainers = ['chartView', 'wordFreqCharts'];
    chartContainers.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            // 确保DOM结构正确，但实际显示由switchTab控制
            element.style.visibility = 'visible';
            element.style.opacity = '1';
        }
    });

    // 更新分页和句子显示
    updateDisplay();

    // 处理整体分析结果
    if (data.overall && data.overall.probabilities) {
        const overall = data.overall;
        document.getElementById('overallSentiment').textContent = overall.sentiment || 'N/A';
        document.getElementById('overallPositive').style.width = `${overall.probabilities.positive || 0}%`;
        document.getElementById('overallNegative').style.width = `${overall.probabilities.negative || 0}%`;
        document.getElementById('overallPositiveProb').textContent = (overall.probabilities.positive || 0).toFixed(2);
        document.getElementById('overallNegativeProb').textContent = (overall.probabilities.negative || 0).toFixed(2);
    } else {
        showError('整体分析结果数据不完整');
    }

    // 处理模型评估指标
    if (data.modelMetrics) {
        document.getElementById('accuracy').textContent = (data.modelMetrics.accuracy * 100).toFixed(2);
        document.getElementById('f1Score').textContent = (data.modelMetrics.f1_score * 100).toFixed(2);
        document.getElementById('recall').textContent = (data.modelMetrics.recall * 100).toFixed(2);

        // 初始化混淆矩阵
        try {
            initConfusionMatrix(data);
        } catch (e) {
            console.error('初始化混淆矩阵失败:', e);
        }
    }

    try {
        // 初始化所有图表，包括新添加的词频图表
        initCharts(data);

        // 初始化词频标签
        initWordFreqTags(data);

        // 设置标签页事件 - 使用ui.js中的函数
        if (typeof setupTabButtonsEvents === 'function') {
            setupTabButtonsEvents();
        } else {
            // 旧版本兼容：直接绑定事件
            document.querySelectorAll('#sentenceResults .tab-button').forEach(button => {
                button.addEventListener('click', function () {
                    switchTab(this.dataset.view);
                });
            });
        }

        // 初始化词频展示区域，确保词频图表和标签都显示
        if (typeof initWordFreqDisplay === 'function') {
            initWordFreqDisplay();
        }

        // 默认显示列表视图
        switchTab('list');

        // 自动切换到总体分析区块
        switchSection('overall-section');
    } catch (error) {
        console.error('初始化图表失败:', error);
        console.error(error.stack);
        showError('图表初始化失败: ' + error.message);
    }
}

// 删除不再需要的词频图表相关函数
function ensureWordFreqChartsVisible() {
    // 此函数不再需要，因为我们已经删除了词频图表
    console.log('词频图表区域已被移除');
}

// 确保训练脚本正确加载
function checkTrainingScriptLoaded() {
    if (typeof startTraining !== 'function') {
        console.error('训练脚本未正确加载，startTraining函数未定义');
        // 尝试重新加载脚本
        const scriptElement = document.createElement('script');
        scriptElement.src = '/static/js/training.js?_=' + new Date().getTime();
        document.body.appendChild(scriptElement);
        
        scriptElement.onload = function() {
            console.log('训练脚本已重新加载');
        };
        
        scriptElement.onerror = function() {
            console.error('训练脚本加载失败');
        };
    } else {
        console.log('训练脚本已正确加载');
    }
}

// 历史记录相关常量和变量
const MAX_TEXT_HISTORY = 10;
const MAX_RESULTS_HISTORY = 10;
let currentAnalysisResult = null;

/**
 * 获取当前登录用户名
 * @returns {string} 当前用户名或默认值"guest"
 */
function getCurrentUsername() {
    // 首先尝试使用fileUploadModule中的函数（如果已加载）
    if (window.fileUploadModule && typeof window.fileUploadModule.getCurrentUsername === 'function') {
        return window.fileUploadModule.getCurrentUsername();
    }
    
    // 尝试从页面元素获取用户名
    const navbarUsername = document.getElementById('navbarUsername');
    if (navbarUsername && navbarUsername.textContent.trim()) {
        return navbarUsername.textContent.trim();
    }
    
    // 尝试从localStorage获取
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
    
    // 尝试从window对象获取全局用户信息
    if (window.currentUser && window.currentUser.username) {
        return window.currentUser.username;
    }
    
    // 如果都获取不到，返回默认值
    return "guest";
}

/**
 * 页面加载完成时的初始化
 */
document.addEventListener('DOMContentLoaded', function () {
    document.body.classList.add('fade-in');

    // 确保认证信息先加载
    if (typeof fetchUserInfo === 'function') {
        fetchUserInfo().then(() => {
            // 等认证信息加载完成后再设置UI事件
            setupUIEventListeners();
        });
    } else {
        // 如果没有认证模块，也设置UI事件
        setupUIEventListeners();
    }

    initFileUpload(); // 初始化文件上传功能
    fetchModels(); // 获取可用的模型列表

    // 添加模型选择框的事件监听
    document.getElementById('modelSelect').addEventListener('change', updateModelDescription);
    document.getElementById('fileModelSelect').addEventListener('change', updateModelDescription);

    // 初始状态下激活第一个菜单项
    switchSection('input-section');
    
    // 检查训练脚本加载情况
    setTimeout(checkTrainingScriptLoaded, 2000);

    // 菜单项点击事件
    document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', function() {
            // ...existing code...
            
            // 移除这个条件，因为我们不再需要处理词频图表
            // if (this.getAttribute('data-section') === 'word-freq-section') {
            //     ensureWordFreqChartsVisible();
            // }
        });
    });

    // 初始化文本历史功能
    initTextHistory();
    
    // 初始化保存结果功能
    initSaveResultsFeature();
});

/**
 * 初始化文本历史功能
 */
function initTextHistory() {
    // 获取文本输入区域
    const textInput = document.getElementById('textInput');
    if (!textInput) return;
    
    // 添加文本输入指南
    const textGuideDiv = document.createElement('div');
    textGuideDiv.className = 'text-guide mt-3';
    textGuideDiv.innerHTML = `
        <div class="card guide-card">
            <div class="card-header guide-header">
                <i class="bi bi-lightbulb-fill text-warning me-2"></i>
                <span>文本输入指南</span>
            </div>
            <div class="card-body guide-content">
                <ul class="guide-tips">
                    <li>文本长度建议在 50-500 字之间，过短的文本可能导致分析不准确</li>
                    <li>输入完整的句子，避免使用过多缩写或网络用语</li>
                    <li>每段文本应该包含明确的情感表达，以获得更准确的结果</li>
                    <li>可以输入多个段落，系统会对每个句子进行单独分析</li>
                </ul>
                <div class="guide-examples">
                    <p><strong>示例:</strong> "这款产品使用起来非常方便，比我之前用的好多了，但是价格有点贵。"</p>
                </div>
            </div>
        </div>
    `;
    
    // 将指南添加到文本输入区域的后面
    textInput.parentElement.insertBefore(textGuideDiv, textInput.nextSibling);
    
    // 创建历史文本容器
    const textHistoryContainer = document.createElement('div');
    textHistoryContainer.className = 'text-history-container mt-3';
    textHistoryContainer.innerHTML = `
        <div class="history-header">
            <h6 class="mb-0"><i class="bi bi-clock-history me-2"></i>历史文本</h6>
        </div>
        <div class="history-content" id="textHistoryContent">
            <div class="text-center py-3">
                <div class="spinner-border spinner-border-sm text-primary" role="status">
                    <span class="visually-hidden">加载中...</span>
                </div>
                <span class="ms-2">正在加载历史文本...</span>
            </div>
        </div>
    `;
    
    // 找到文本输入标签页内的最后一个元素，将历史文本添加到最后
    const textTab = document.getElementById('textInput-tab');
    if (textTab) {
        // 添加到标签页的最后
        textTab.appendChild(textHistoryContainer);
    } else {
        // 回退方案：添加到文本输入框的父元素末尾
        textInput.parentElement.appendChild(textHistoryContainer);
    }
    
    // 添加指南收起/展开功能
    const guideToggle = textGuideDiv.querySelector('.guide-toggle');
    const guideContent = textGuideDiv.querySelector('.guide-content');
    if (guideToggle && guideContent) {
        guideToggle.addEventListener('click', function() {
            if (guideContent.style.display === 'none') {
                guideContent.style.display = 'block';
                guideToggle.setAttribute('aria-label', '收起');
            } else {
                guideContent.style.display = 'none';
                guideToggle.setAttribute('aria-label', '展开');
            }
        });
    }
    
    // 监听文本输入变化，自动保存到历史记录
    textInput.addEventListener('blur', function() {
        const text = this.value.trim();
        if (text && text.length > 10) { // 只保存有意义的文本
            saveTextToHistory(text);
        }
    });
    
    // 立即加载历史文本列表
    setTimeout(() => {
        loadTextHistoryInline();
    }, 100);
}

/**
 * 加载文本历史记录并内联显示在页面中
 */
function loadTextHistoryInline() {
    try {
        // 获取当前用户名，用于构建存储键
        const username = getCurrentUsername();
        const storageKey = `textHistory_${username}`;
        
        // 获取历史记录
        const textHistory = JSON.parse(localStorage.getItem(storageKey) || '[]');
        
        // 获取历史内容容器
        const historyContent = document.getElementById('textHistoryContent');
        if (!historyContent) return;
        
        // 如果没有历史记录
        if (textHistory.length === 0) {
            historyContent.innerHTML = `
                <div class="empty-history-message">
                    <i class="bi bi-inbox text-muted"></i>
                    <p>暂无历史文本记录</p>
                </div>
            `;
            return;
        }
        
        // 创建历史记录内容
        let historyHtml = '';
        textHistory.forEach((item, index) => {
            const date = new Date(item.timestamp);
            const formattedDate = `${date.getFullYear()}-${(date.getMonth()+1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
            
            historyHtml += `
                <div class="inline-history-item" data-index="${index}">
                    <div class="inline-history-item-header">
                        <div class="inline-history-item-title">${item.title || '未命名文本'}</div>
                        <div class="inline-history-item-date">${formattedDate}</div>
                    </div>
                    <div class="inline-history-item-preview">${item.text.substring(0, 50)}${item.text.length > 50 ? '...' : ''}</div>
                    <div class="inline-history-item-actions">
                        <button type="button" class="btn btn-sm btn-primary load-history-text" data-index="${index}">
                            <i class="bi bi-upload"></i> 加载
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-danger delete-history-text" data-index="${index}">
                            <i class="bi bi-trash"></i> 删除
                        </button>
                    </div>
                </div>
            `;
        });
        
        // 更新历史记录容器
        historyContent.innerHTML = historyHtml;
        
        // 添加事件监听
        historyContent.querySelectorAll('.load-history-text').forEach(btn => {
            btn.addEventListener('click', function() {
                const index = this.getAttribute('data-index');
                const textItem = textHistory[index];
                
                // 加载历史文本
                const textInput = document.getElementById('textInput');
                if (textInput) {
                    textInput.value = textItem.text;
                    
                    // 激活文本输入标签页
                    const textTab = document.getElementById('text-tab');
                    if (textTab && bootstrap.Tab) {
                        const tab = new bootstrap.Tab(textTab);
                        tab.show();
                    }
                }
                
                // 显示成功消息
                showToast('已加载历史文本', 'success');
            });
        });
        
        historyContent.querySelectorAll('.delete-history-text').forEach(btn => {
            btn.addEventListener('click', function() {
                const index = this.getAttribute('data-index');
                
                // 删除历史记录
                textHistory.splice(index, 1);
                localStorage.setItem(storageKey, JSON.stringify(textHistory));
                
                // 从界面中移除
                const historyItem = this.closest('.inline-history-item');
                historyItem.classList.add('fade-out');
                setTimeout(() => {
                    historyItem.remove();
                    
                    // 如果没有更多历史记录，显示空历史消息
                    if (historyContent.querySelectorAll('.inline-history-item').length === 0) {
                        historyContent.innerHTML = `
                            <div class="empty-history-message">
                                <i class="bi bi-inbox text-muted"></i>
                                <p>暂无历史文本记录</p>
                            </div>
                        `;
                    }
                    
                    // 显示消息
                    showToast('已删除历史文本', 'info');
                }, 300);
            });
        });
    } catch (error) {
        console.error('加载文本历史记录失败:', error);
        const historyContent = document.getElementById('textHistoryContent');
        if (historyContent) {
            historyContent.innerHTML = `
                <div class="error-message">
                    <i class="bi bi-exclamation-triangle"></i>
                    <p>加载历史记录失败</p>
                </div>
            `;
        }
    }
}

/**
 * 保存文本到历史记录
 * @param {string} text - 文本内容
 */
function saveTextToHistory(text) {
    try {
        // 获取当前用户名，用于构建存储键
        const username = getCurrentUsername();
        const storageKey = `textHistory_${username}`;
        
        // 获取现有历史记录
        let textHistory = JSON.parse(localStorage.getItem(storageKey) || '[]');
        
        // 检查是否已存在相同内容的记录
        const existingIndex = textHistory.findIndex(item => item.text === text);
        if (existingIndex !== -1) {
            // 如果存在，更新时间戳并将其移动到顶部
            textHistory.splice(existingIndex, 1);
        }
        
        // 添加新记录
        textHistory.unshift({
            text: text,
            timestamp: Date.now(),
            title: generateTextTitle(text)
        });
        
        // 保持历史记录不超过最大数量
        if (textHistory.length > MAX_TEXT_HISTORY) {
            textHistory = textHistory.slice(0, MAX_TEXT_HISTORY);
        }
        
        // 保存到本地存储
        localStorage.setItem(storageKey, JSON.stringify(textHistory));
        
        // 重新加载内联显示的历史记录
        loadTextHistoryInline();
    } catch (error) {
        console.error('保存文本历史记录失败:', error);
    }
}

/**
 * 生成文本标题
 * @param {string} text - 文本内容
 * @returns {string} - 生成的标题
 */
function generateTextTitle(text) {
    // 简单地截取前面一部分文本作为标题
    return text.substring(0, 30) + (text.length > 30 ? '...' : '');
}

/**
 * 显示文本历史记录
 */
function showTextHistory() {
    try {
        // 获取当前用户名，用于构建存储键
        const username = getCurrentUsername();
        const storageKey = `textHistory_${username}`;
        
        // 获取历史记录
        const textHistory = JSON.parse(localStorage.getItem(storageKey) || '[]');
        
        // 如果没有历史记录
        if (textHistory.length === 0) {
            showToast('没有历史文本记录', 'info');
            return;
        }
        
        // 创建模态框内容
        let modalContent = '';
        textHistory.forEach((item, index) => {
            const date = new Date(item.timestamp);
            const formattedDate = `${date.getFullYear()}-${(date.getMonth()+1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
            
            modalContent += `
                <div class="history-item" data-index="${index}">
                    <div class="history-item-header">
                        <span class="history-item-title">${item.title}</span>
                        <span class="history-item-date">${formattedDate}</span>
                    </div>
                    <div class="history-item-preview">${item.text.substring(0, 100)}${item.text.length > 100 ? '...' : ''}</div>
                    <div class="history-item-actions">
                        <button type="button" class="btn btn-sm btn-primary load-history-text" data-index="${index}">
                            <i class="bi bi-upload"></i> 加载该文本
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-danger delete-history-text" data-index="${index}">
                            <i class="bi bi-trash"></i> 删除
                        </button>
                    </div>
                </div>
            `;
        });
        
        // 创建模态框
        const modal = createHistoryModal('历史文本记录', modalContent);
        
        // 显示模态框
        document.body.appendChild(modal);
        const modalInstance = new bootstrap.Modal(modal);
        modalInstance.show();
        
        // 添加事件监听
        modal.querySelectorAll('.load-history-text').forEach(btn => {
            btn.addEventListener('click', function() {
                const index = this.getAttribute('data-index');
                const textItem = textHistory[index];
                
                // 加载历史文本
                const textInput = document.getElementById('textInput');
                if (textInput) {
                    textInput.value = textItem.text;
                    
                    // 激活文本输入标签页
                    const textTab = document.getElementById('text-tab');
                    if (textTab && bootstrap.Tab) {
                        const tab = new bootstrap.Tab(textTab);
                        tab.show();
                    }
                }
                
                // 关闭模态框
                modalInstance.hide();
                
                // 显示成功消息
                showToast('已加载历史文本', 'success');
            });
        });
        
        modal.querySelectorAll('.delete-history-text').forEach(btn => {
            btn.addEventListener('click', function() {
                const index = this.getAttribute('data-index');
                
                // 删除历史记录
                textHistory.splice(index, 1);
                localStorage.setItem(storageKey, JSON.stringify(textHistory));
                
                // 从界面中移除
                const historyItem = this.closest('.history-item');
                historyItem.classList.add('fade-out');
                setTimeout(() => {
                    historyItem.remove();
                    
                    // 如果没有更多历史记录，关闭模态框
                    if (modal.querySelectorAll('.history-item').length === 0) {
                        modalInstance.hide();
                        showToast('没有更多历史文本记录', 'info');
                    }
                }, 300);
            });
        });
    } catch (error) {
        console.error('显示文本历史记录失败:', error);
        showToast('无法加载历史文本记录', 'error');
    }
}

/**
 * 创建历史记录模态框
 */
function createHistoryModal(title, content) {
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
 * 初始化保存结果功能
 */
function initSaveResultsFeature() {
    // 检查分析结果区域
    const resultsArea = document.getElementById('results-area');
    if (!resultsArea) return;
    
    // 创建保存结果和历史结果按钮容器
    const resultBtnsContainer = document.createElement('div');
    resultBtnsContainer.className = 'result-actions d-flex justify-content-end mt-3 mb-3';
    
    // 创建保存结果按钮
    const saveResultBtn = document.createElement('button');
    saveResultBtn.type = 'button';
    saveResultBtn.className = 'btn btn-success me-2';
    saveResultBtn.id = 'saveResultBtn';
    saveResultBtn.innerHTML = '<i class="bi bi-save me-1"></i> 保存结果';
    saveResultBtn.addEventListener('click', saveAnalysisResult);
    saveResultBtn.disabled = true; // 初始状态禁用
    
    // 创建查看历史结果按钮
    const historyResultBtn = document.createElement('button');
    historyResultBtn.type = 'button';
    historyResultBtn.className = 'btn btn-outline-primary';
    historyResultBtn.id = 'historyResultBtn';
    historyResultBtn.innerHTML = '<i class="bi bi-clock-history me-1"></i> 历史结果';
    historyResultBtn.addEventListener('click', showResultsHistory);
    
    // 将按钮添加到容器
    resultBtnsContainer.appendChild(saveResultBtn);
    resultBtnsContainer.appendChild(historyResultBtn);
    
    // 将容器插入到结果区域
    const firstChild = resultsArea.firstChild;
    if (firstChild) {
        resultsArea.insertBefore(resultBtnsContainer, firstChild);
    } else {
        resultsArea.appendChild(resultBtnsContainer);
    }
}

/**
 * 保存当前分析结果
 */
function saveAnalysisResult() {
    try {
        // 确保有分析结果
        if (!currentAnalysisResult) {
            showToast('没有可保存的分析结果', 'warning');
            return;
        }
        
        // 显示保存对话框
        const modal = createSaveResultModal();
        document.body.appendChild(modal);
        const modalInstance = new bootstrap.Modal(modal);
        modalInstance.show();
        
        // 处理保存按钮点击事件
        const saveBtn = modal.querySelector('#confirmSaveResult');
        saveBtn.addEventListener('click', function() {
            // 获取用户输入的标题
            const titleInput = modal.querySelector('#resultTitleInput');
            let title = titleInput.value.trim();
            
            // 如果没有输入标题，使用默认标题
            if (!title) {
                // 根据来源类型生成默认标题
                const isFile = window.fileUploadModule && window.fileUploadModule.getCurrentFileName();
                if (isFile) {
                    title = `文件分析: ${window.fileUploadModule.getCurrentFileName()}`;
                } else {
                    const textInput = document.getElementById('textInput');
                    const text = textInput ? textInput.value.trim() : '';
                    title = `文本分析: ${text.substring(0, 20)}${text.length > 20 ? '...' : ''}`;
                }
            }
            
            // 准备要保存的结果数据
            const resultData = {
                title: title,
                timestamp: Date.now(),
                result: currentAnalysisResult,
                source: window.fileUploadModule && window.fileUploadModule.getCurrentFileName() ? 'file' : 'text',
                sourceContent: window.fileUploadModule && window.fileUploadModule.getCurrentFileName() ? 
                    window.fileUploadModule.getCurrentFileContent() : 
                    document.getElementById('textInput')?.value || ''
            };
            
            // 保存到本地存储
            saveResultToHistory(resultData);
            
            // 关闭模态框
            modalInstance.hide();
            
            // 显示成功消息
            showToast('分析结果已成功保存', 'success');
        });
    } catch (error) {
        console.error('保存分析结果失败:', error);
        showToast('保存分析结果时出错', 'danger');
    }
}

/**
 * 创建保存结果模态框
 */
function createSaveResultModal() {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'saveResultModal';
    modal.tabIndex = '-1';
    modal.setAttribute('aria-labelledby', 'saveResultModalLabel');
    modal.setAttribute('aria-hidden', 'true');
    
    // 确定默认标题
    let defaultTitle = '';
    if (window.fileUploadModule && window.fileUploadModule.getCurrentFileName()) {
        defaultTitle = `文件分析: ${window.fileUploadModule.getCurrentFileName()}`;
    } else {
        const textInput = document.getElementById('textInput');
        const text = textInput ? textInput.value.trim() : '';
        if (text) {
            defaultTitle = `文本分析: ${text.substring(0, 20)}${text.length > 20 ? '...' : ''}`;
        } else {
            defaultTitle = `分析结果 ${new Date().toLocaleString()}`;
        }
    }
    
    modal.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="saveResultModalLabel">保存分析结果</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label for="resultTitleInput" class="form-label">结果标题</label>
                        <input type="text" class="form-control" id="resultTitleInput" placeholder="为分析结果起个标题" value="${defaultTitle}">
                    </div>
                    <p class="text-muted small">保存后可在"历史结果"中查看</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                    <button type="button" class="btn btn-success" id="confirmSaveResult">
                        <i class="bi bi-save me-1"></i> 保存
                    </button>
                </div>
            </div>
        </div>
    `;
    
    return modal;
}

/**
 * 保存分析结果到历史记录
 */
function saveResultToHistory(resultData) {
    try {
        // 获取当前用户名，用于构建存储键
        const username = getCurrentUsername();
        const storageKey = `resultsHistory_${username}`;
        
        // 获取现有历史记录
        let resultsHistory = JSON.parse(localStorage.getItem(storageKey) || '[]');
        
        // 添加新记录
        resultsHistory.unshift(resultData);
        
        // 保持历史记录不超过最大数量
        if (resultsHistory.length > MAX_RESULTS_HISTORY) {
            resultsHistory = resultsHistory.slice(0, MAX_RESULTS_HISTORY);
        }
        
        // 保存到本地存储
        localStorage.setItem(storageKey, JSON.stringify(resultsHistory));
    } catch (error) {
        console.error('保存结果历史记录失败:', error);
        showToast('无法保存结果到历史记录', 'danger');
    }
}

/**
 * 显示历史分析结果
 */
function showResultsHistory() {
    try {
        // 获取当前用户名，用于构建存储键
        const username = getCurrentUsername();
        const storageKey = `resultsHistory_${username}`;
        
        // 获取历史记录
        const resultsHistory = JSON.parse(localStorage.getItem(storageKey) || '[]');
        
        // 如果没有历史记录
        if (resultsHistory.length === 0) {
            showToast('没有历史分析结果记录', 'info');
            return;
        }
        
        // 创建模态框内容
        let modalContent = '';
        resultsHistory.forEach((item, index) => {
            const date = new Date(item.timestamp);
            const formattedDate = `${date.getFullYear()}-${(date.getMonth()+1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
            
            const sourceType = item.source === 'file' ? 
                `<span class="badge bg-info"><i class="bi bi-file-text"></i> 文件</span>` : 
                `<span class="badge bg-primary"><i class="bi bi-keyboard"></i> 文本</span>`;
            
            // 从结果中获取情感分析结果
            let sentimentResult = '';
            if (item.result && item.result.overall_sentiment) {
                const sentiment = item.result.overall_sentiment;
                const sentimentText = sentiment > 0 ? '积极' : (sentiment < 0 ? '消极' : '中性');
                const sentimentClass = sentiment > 0 ? 'bg-success' : (sentiment < 0 ? 'bg-danger' : 'bg-secondary');
                
                sentimentResult = `<span class="badge ${sentimentClass}">整体情感: ${sentimentText}</span>`;
            }
            
            modalContent += `
                <div class="history-item result-history-item" data-index="${index}">
                    <div class="history-item-header">
                        <div class="history-item-title-wrapper">
                            <span class="history-item-title">${item.title}</span>
                            ${sourceType} ${sentimentResult}
                        </div>
                        <span class="history-item-date">${formattedDate}</span>
                    </div>
                    <div class="history-item-preview">
                        ${item.sourceContent.substring(0, 100)}${item.sourceContent.length > 100 ? '...' : ''}
                    </div>
                    <div class="history-item-actions">
                        <button type="button" class="btn btn-sm btn-primary view-history-result" data-index="${index}">
                            <i class="bi bi-search"></i> 查看结果
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-secondary load-history-source" data-index="${index}">
                            <i class="bi bi-upload"></i> 加载源内容
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-danger delete-history-result" data-index="${index}">
                            <i class="bi bi-trash"></i> 删除
                        </button>
                    </div>
                </div>
            `;
        });
        
        // 创建模态框
        const modal = createHistoryModal('历史分析结果', modalContent);
        
        // 显示模态框
        document.body.appendChild(modal);
        const modalInstance = new bootstrap.Modal(modal);
        modalInstance.show();
        
        // 添加事件监听
        modal.querySelectorAll('.view-history-result').forEach(btn => {
            btn.addEventListener('click', function() {
                const index = this.getAttribute('data-index');
                const resultItem = resultsHistory[index];
                
                // 显示分析结果
                displayAnalysisResult(resultItem.result);
                
                // 将当前分析结果设为这个历史结果
                currentAnalysisResult = resultItem.result;
                
                // 启用保存按钮
                const saveResultBtn = document.getElementById('saveResultBtn');
                if (saveResultBtn) {
                    saveResultBtn.disabled = false;
                }
                
                // 关闭模态框
                modalInstance.hide();
            });
        });
        
        modal.querySelectorAll('.load-history-source').forEach(btn => {
            btn.addEventListener('click', function() {
                const index = this.getAttribute('data-index');
                const resultItem = resultsHistory[index];
                
                if (resultItem.source === 'file') {
                    // 如果来源是文件，设置文件内容
                    if (window.fileUploadModule) {
                        // 模拟文件上传
                        const fileInfo = document.getElementById('fileInfo');
                        const fileViewBtn = document.getElementById('fileViewBtn');
                        
                        if (fileInfo && fileViewBtn) {
                            fileInfo.textContent = `已选择: ${resultItem.title.replace('文件分析: ', '')}`;
                            fileViewBtn.disabled = false;
                            
                            // 设置当前文件内容
                            window.fileUploadModule.currentFileContent = resultItem.sourceContent;
                            window.fileUploadModule.currentFileName = resultItem.title.replace('文件分析: ', '');
                            
                            // 激活文件标签页
                            const fileTab = document.getElementById('file-tab');
                            if (fileTab && bootstrap.Tab) {
                                const tab = new bootstrap.Tab(fileTab);
                                tab.show();
                            }
                        }
                    }
                } else {
                    // 如果来源是文本，设置文本内容
                    const textInput = document.getElementById('textInput');
                    if (textInput) {
                        textInput.value = resultItem.sourceContent;
                        
                        // 激活文本标签页
                        const textTab = document.getElementById('text-tab');
                        if (textTab && bootstrap.Tab) {
                            const tab = new bootstrap.Tab(textTab);
                            tab.show();
                        }
                    }
                }
                
                // 关闭模态框
                modalInstance.hide();
                
                // 显示成功消息
                showToast('已加载源内容', 'success');
            });
        });
        
        modal.querySelectorAll('.delete-history-result').forEach(btn => {
            btn.addEventListener('click', function() {
                const index = this.getAttribute('data-index');
                
                // 删除历史记录
                resultsHistory.splice(index, 1);
                localStorage.setItem(storageKey, JSON.stringify(resultsHistory));
                
                // 从界面中移除
                const historyItem = this.closest('.history-item');
                historyItem.classList.add('fade-out');
                setTimeout(() => {
                    historyItem.remove();
                    
                    // 如果没有更多历史记录，关闭模态框
                    if (modal.querySelectorAll('.history-item').length === 0) {
                        modalInstance.hide();
                        showToast('没有更多历史分析结果', 'info');
                    }
                }, 300);
            });
        });
    } catch (error) {
        console.error('显示结果历史记录失败:', error);
        showToast('无法加载历史分析结果', 'error');
    }
}

/**
 * 显示Toast通知
 * @param {string} message - 通知消息
 * @param {string} type - 通知类型 (success, info, warning, danger)
 * @param {number} duration - 显示时长 (ms)
 */
function showToast(message, type = 'info', duration = 3000) {
    // 检查是否存在toast容器
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // 创建toast元素
    const toastId = `toast-${Date.now()}`;
    const toast = document.createElement('div');
    toast.className = `toast align-items-center border-0 text-white bg-${type}`;
    toast.id = toastId;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    // 使用Bootstrap的Toast API显示通知
    const bsToast = new bootstrap.Toast(toast, {
        delay: duration
    });
    bsToast.show();
    
    // 当toast隐藏后移除元素
    toast.addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}

/**
 * 显示分析结果
 * @param {Object} result - 分析结果对象
 */
function displayAnalysisResult(result) {
    // 保存当前分析结果
    currentAnalysisResult = result;
    
    // 启用保存按钮
    const saveResultBtn = document.getElementById('saveResultBtn');
    if (saveResultBtn) {
        saveResultBtn.disabled = false;
    }
    
    // 这里应该是原有的分析结果展示逻辑
    // 如果没有这部分代码，可以根据实际情况调用相应的函数
    
    // 显示结果区域
    const resultsArea = document.getElementById('results-area');
    if (resultsArea) {
        resultsArea.classList.remove('d-none');
    }
}

/**
 * 处理分析结果
 * @param {Object} data - API返回的分析结果数据
 */
function processAnalysisResult(data) {
    // 保存当前分析结果
    currentAnalysisResult = data;
    
    // 启用保存按钮
    const saveResultBtn = document.getElementById('saveResultBtn');
    if (saveResultBtn) {
        saveResultBtn.disabled = false;
    }
    
    // 这里应该保留原有的结果处理逻辑
    // ...
}

/**
 * 导出分析结果为CSV或TXT文件
 * @param {string} type - 导出类型：'overall', 'sentences', 'wordfreq'
 * @param {string} format - 导出格式：'csv', 'txt'
 */
function exportResults(type, format) {
    // 检查是否有分析结果
    if (!window.lastAnalysisData) {
        showToast('没有可导出的分析结果', 'warning');
        return;
    }

    try {
        let content = '';
        let filename = '';
        const timestamp = new Date().toISOString().replace(/[-:]/g, '').substring(0, 15);
        
        switch (type) {
            case 'overall':
                content = formatOverallResults(window.lastAnalysisData, format);
                filename = `整体分析结果_${timestamp}.${format}`;
                break;
            case 'sentences':
                content = formatSentencesResults(window.lastAnalysisData.sentences, format);
                filename = `句子分析结果_${timestamp}.${format}`;
                break;
            case 'wordfreq':
                content = formatWordFreqResults(window.lastAnalysisData.wordFreq, format);
                filename = `词频统计结果_${timestamp}.${format}`;
                break;
            default:
                showToast('未知的导出类型', 'danger');
                return;
        }

        // 创建文件下载
        downloadFile(content, filename, format);
        
        // 显示成功消息
        showToast(`已成功导出为${format.toUpperCase()}文件`, 'success');
    } catch (error) {
        console.error('导出结果失败:', error);
        showToast('导出结果失败', 'danger');
    }
}

/**
 * 格式化整体分析结果
 * @param {Object} data - 分析结果数据
 * @param {string} format - 导出格式：'csv', 'txt'
 * @returns {string} - 格式化后的内容
 */
function formatOverallResults(data, format) {
    if (format === 'csv') {
        let csv = '分析项目,结果\n';
        csv += `整体情感,${data.overall.sentiment || 'N/A'}\n`;
        csv += `积极概率,${(data.overall.probabilities?.positive || 0).toFixed(2)}%\n`;
        csv += `消极概率,${(data.overall.probabilities?.negative || 0).toFixed(2)}%\n`;
        
        // 添加模型评估指标
        if (data.modelMetrics) {
            csv += `准确率,${(data.modelMetrics.accuracy * 100).toFixed(2)}%\n`;
            csv += `F1分数,${(data.modelMetrics.f1_score * 100).toFixed(2)}%\n`;
            csv += `召回率,${(data.modelMetrics.recall * 100).toFixed(2)}%\n`;
        }
        
        // 添加混淆矩阵数据（如果有）
        if (data.modelMetrics?.confusion_matrix) {
            const cm = data.modelMetrics.confusion_matrix;
            csv += '\n混淆矩阵:\n';
            csv += ',预测积极,预测消极\n';
            csv += `实际积极,${cm[0][0]},${cm[0][1]}\n`;
            csv += `实际消极,${cm[1][0]},${cm[1][1]}\n`;
        }
        
        return csv;
    } else {
        let txt = '============== 情感分析整体结果 ==============\n\n';
        txt += `整体情感倾向: ${data.overall.sentiment || 'N/A'}\n`;
        txt += `积极概率: ${(data.overall.probabilities?.positive || 0).toFixed(2)}%\n`;
        txt += `消极概率: ${(data.overall.probabilities?.negative || 0).toFixed(2)}%\n\n`;
        
        // 添加模型评估指标
        if (data.modelMetrics) {
            txt += '-------------- 模型评估指标 --------------\n\n';
            txt += `准确率: ${(data.modelMetrics.accuracy * 100).toFixed(2)}%\n`;
            txt += `F1分数: ${(data.modelMetrics.f1_score * 100).toFixed(2)}%\n`;
            txt += `召回率: ${(data.modelMetrics.recall * 100).toFixed(2)}%\n\n`;
        }
        
        // 添加混淆矩阵数据（如果有）
        if (data.modelMetrics?.confusion_matrix) {
            const cm = data.modelMetrics.confusion_matrix;
            txt += '-------------- 混淆矩阵 --------------\n\n';
            txt += '             | 预测积极 | 预测消极 \n';
            txt += '-------------------------------\n';
            txt += `实际积极 |    ${cm[0][0]}    |    ${cm[0][1]}    \n`;
            txt += `实际消极 |    ${cm[1][0]}    |    ${cm[1][1]}    \n\n`;
        }
        
        txt += `分析时间: ${new Date().toLocaleString()}\n`;
        txt += '==========================================\n';
        return txt;
    }
}

/**
 * 格式化句子分析结果
 * @param {Array} sentences - 句子分析结果数组
 * @param {string} format - 导出格式：'csv', 'txt'
 * @returns {string} - 格式化后的内容
 */
function formatSentencesResults(sentences, format) {
    if (!sentences || sentences.length === 0) {
        return format === 'csv' ? '没有句子分析结果\n' : '没有句子分析结果';
    }
    
    if (format === 'csv') {
        let csv = '序号,句子内容,情感倾向,积极概率,消极概率\n';
        sentences.forEach((sentence, index) => {
            // 处理CSV中的引号和逗号
            const content = `"${sentence.text.replace(/"/g, '""')}"`;
            const sentiment = sentence.sentiment || 'N/A';
            const positive = (sentence.probabilities?.positive || 0).toFixed(2);
            const negative = (sentence.probabilities?.negative || 0).toFixed(2);
            csv += `${index + 1},${content},${sentiment},${positive},${negative}\n`;
        });
        return csv;
    } else {
        let txt = '============== 句子分析结果 ==============\n\n';
        txt += `共分析 ${sentences.length} 个句子\n\n`;
        
        sentences.forEach((sentence, index) => {
            txt += `[${index + 1}] 句子: ${sentence.text}\n`;
            txt += `    情感倾向: ${sentence.sentiment || 'N/A'}\n`;
            txt += `    积极概率: ${(sentence.probabilities?.positive || 0).toFixed(2)}%\n`;
            txt += `    消极概率: ${(sentence.probabilities?.negative || 0).toFixed(2)}%\n\n`;
        });
        
        txt += `分析时间: ${new Date().toLocaleString()}\n`;
        txt += '==========================================\n';
        return txt;
    }
}

/**
 * 格式化词频统计结果
 * @param {Array} wordFreq - 词频统计数组
 * @param {string} format - 导出格式：'csv', 'txt'
 * @returns {string} - 格式化后的内容
 */
function formatWordFreqResults(wordFreq, format) {
    if (!wordFreq || wordFreq.length === 0) {
        return format === 'csv' ? '没有词频统计结果\n' : '没有词频统计结果';
    }
    
    // 对词频排序（从高到低）
    const sortedWordFreq = [...wordFreq].sort((a, b) => b.count - a.count);
    
    if (format === 'csv') {
        let csv = '排名,词语,出现次数\n';
        sortedWordFreq.forEach((item, index) => {
            const word = `"${item.word.replace(/"/g, '""')}"`;
            csv += `${index + 1},${word},${item.count}\n`;
        });
        return csv;
    } else {
        let txt = '============== 词频统计结果 ==============\n\n';
        txt += `共统计 ${sortedWordFreq.length} 个词语\n\n`;
        txt += '排名\t词语\t出现次数\n';
        txt += '---------------------------------------\n';
        
        sortedWordFreq.forEach((item, index) => {
            txt += `${(index + 1).toString().padEnd(4)}\t${item.word.padEnd(12, ' ')}\t${item.count}\n`;
        });
        
        txt += '\n';
        txt += `分析时间: ${new Date().toLocaleString()}\n`;
        txt += '==========================================\n';
        return txt;
    }
}

/**
 * 下载文件的通用函数
 * @param {string} content - 文件内容
 * @param {string} filename - 文件名
 * @param {string} format - 文件格式：'csv', 'txt'
 */
function downloadFile(content, filename, format) {
    // 创建Blob对象
    let blob;
    if (format === 'csv') {
        // 为CSV添加BOM，解决Excel中文乱码问题
        blob = new Blob([new Uint8Array([0xEF, 0xBB, 0xBF]), content], { type: 'text/csv;charset=utf-8' });
    } else {
        blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    }
    
    // 创建下载链接
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    
    // 添加到页面并触发点击
    document.body.appendChild(link);
    link.click();
    
    // 清理
    setTimeout(() => {
        URL.revokeObjectURL(url);
        document.body.removeChild(link);
    }, 100);
}
