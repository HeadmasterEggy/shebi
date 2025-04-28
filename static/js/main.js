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
        // 初始化所有图表，包括词频图表
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

        if (typeof setupWordFreqTabButtonsEvents === 'function') {
            setupWordFreqTabButtonsEvents();
        } else {
            // 旧版本兼容：直接绑定事件
            document.querySelectorAll('#wordFreq .tab-button').forEach(button => {
                button.addEventListener('click', function () {
                    switchWordFreqTab(this.dataset.view);
                });
            });
        }

        // 默认显示列表视图
        switchTab('list');
        switchWordFreqTab('tags');

        // 自动切换到总体分析区块
        switchSection('overall-section');
    } catch (error) {
        console.error('初始化图表失败:', error);
        console.error(error.stack);
        showError('图表初始化失败: ' + error.message);
    }
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
});
