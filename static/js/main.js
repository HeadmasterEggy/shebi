/**
 * 主模块，包含程序入口和核心功能
 */

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

    document.getElementById('loading').style.display = 'flex';
    document.getElementById('errorMessage').style.display = 'none';
    document.getElementById('overallResult').style.display = 'none';
    document.getElementById('sentenceResults').style.display = 'none';
    document.getElementById('wordFreq').style.display = 'none';
    document.getElementById('modelMetrics').style.display = 'none';

    // 检查是否为文件上传模式
    const isFileUpload = document.getElementById('file-tab').classList.contains('active');
    let payload = { text };
    
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
        document.getElementById('loading').style.display = 'none';
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
document.addEventListener('DOMContentLoaded', function() {
    document.body.classList.add('fade-in');
    setupUIEventListeners();
    initFileUpload(); // 初始化文件上传功能
});
