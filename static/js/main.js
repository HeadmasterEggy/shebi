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
    document.getElementById('overallResult').style.display = 'none';
    document.getElementById('sentenceResults').style.display = 'none';
    document.getElementById('wordFreq').style.display = 'none';
    document.getElementById('modelMetrics').style.display = 'none';

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

    allSentences = data.sentences;
    filteredSentences = [...allSentences];
    currentPage = 1;
    sentimentFilter = 'all';
    displayMode = 'normal';

    document.querySelectorAll('.filter-button[data-sentiment]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.sentiment === 'all');
    });

    document.querySelectorAll('.filter-button[data-display]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.display === 'normal');
    });

    const resultCards = document.querySelectorAll('.result-card');
    resultCards.forEach((card, index) => {
        setTimeout(() => {
            card.style.display = 'block';
            card.classList.add('show');
        }, index * 100);
    });

    updateDisplay();

    if (data.overall && data.overall.probabilities) {
        const overall = data.overall;
        document.getElementById('overallSentiment').textContent = overall.sentiment || 'N/A';
        document.getElementById('overallPositive').style.width = `${overall.probabilities.positive || 0}%`;
        document.getElementById('overallNegative').style.width = `${overall.probabilities.negative || 0}%`;
        document.getElementById('overallPositiveProb').textContent = (overall.probabilities.positive || 0).toFixed(2);
        document.getElementById('overallNegativeProb').textContent = (overall.probabilities.negative || 0).toFixed(2);
        document.getElementById('overallResult').style.display = 'block';
    } else {
        showError('整体分析结果数据不完整');
    }

    if (data.modelMetrics) {
        document.getElementById('accuracy').textContent = (data.modelMetrics.accuracy * 100).toFixed(2);
        document.getElementById('f1Score').textContent = (data.modelMetrics.f1_score * 100).toFixed(2);
        document.getElementById('recall').textContent = (data.modelMetrics.recall * 100).toFixed(2);
        document.getElementById('modelMetrics').style.display = 'block';

        // 初始化混淆矩阵
        initConfusionMatrix(data);
    }

    if (data.wordFreq && Array.isArray(data.wordFreq)) {
        document.getElementById('wordFreq').style.display = 'block';
    }

    try {
        initCharts(data);
    } catch (error) {
        console.error('初始化图表失败:', error);
        showError('图表初始化失败: ' + error.message);
    }
}

/**
 * 页面加载完成时的初始化
 */
document.addEventListener('DOMContentLoaded', function () {
    document.body.classList.add('fade-in');
    setupUIEventListeners();
    initFileUpload(); // 初始化文件上传功能
    fetchModels(); // 获取可用的模型列表

    // 添加模型选择框的事件监听
    document.getElementById('modelSelect').addEventListener('change', updateModelDescription);
    document.getElementById('fileModelSelect').addEventListener('change', updateModelDescription);
});
