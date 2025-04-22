/**
 * 模型训练功能模块
 */

// 全局变量
let trainingInProgress = false;
let trainingSocket = null;
let trainingInterval = null;
let trainingStartTime = null;
let currentEpoch = 0;
let totalEpochs = 0;
let lossChart = null;
let accuracyChart = null;
let confusionMatrixChart = null;
let trainingData = {
    epochs: [],
    trainLoss: [],
    valLoss: [],
    trainAcc: [],
    valAcc: []
};

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化图表
    initCharts();
    
    // 显示/隐藏模型特定选项
    const modelTypeSelect = document.getElementById('modelTypeSelect');
    if (modelTypeSelect) {
        modelTypeSelect.addEventListener('change', function() {
            toggleModelSpecificOptions(this.value);
        });
        // 初始化显示/隐藏
        toggleModelSpecificOptions(modelTypeSelect.value);
    }
    
    // Dropout范围滑块值显示
    const dropoutRange = document.getElementById('dropoutRange');
    const dropoutValue = document.getElementById('dropoutValue');
    if (dropoutRange && dropoutValue) {
        dropoutRange.addEventListener('input', function() {
            dropoutValue.textContent = this.value;
        });
    }
    
    // 早停选项切换
    const earlyStopping = document.getElementById('earlyStopping');
    const earlyStoppingOptions = document.getElementById('earlyStoppingOptions');
    if (earlyStopping && earlyStoppingOptions) {
        earlyStopping.addEventListener('change', function() {
            earlyStoppingOptions.style.display = this.checked ? 'block' : 'none';
        });
    }
    
    // 训练控制按钮事件绑定
    const pauseTrainingBtn = document.getElementById('pauseTrainingBtn');
    const stopTrainingBtn = document.getElementById('stopTrainingBtn');
    const saveModelBtn = document.getElementById('saveModelBtn');
    
    if (pauseTrainingBtn) {
        pauseTrainingBtn.addEventListener('click', function() {
            if (trainingInProgress) {
                pauseTraining();
            }
        });
    }
    
    if (stopTrainingBtn) {
        stopTrainingBtn.addEventListener('click', function() {
            if (trainingInProgress) {
                stopTraining();
            }
        });
    }
    
    if (saveModelBtn) {
        saveModelBtn.addEventListener('click', saveModel);
    }
});

/**
 * 初始化图表
 */
function initCharts() {
    // 初始化损失图表
    if (document.getElementById('lossChart')) {
        lossChart = echarts.init(document.getElementById('lossChart'));
        const lossOption = {
            title: {
                text: '训练/验证损失',
                left: 'center'
            },
            tooltip: {
                trigger: 'axis'
            },
            legend: {
                data: ['训练损失', '验证损失'],
                bottom: 10
            },
            xAxis: {
                type: 'category',
                name: '轮次',
                nameLocation: 'middle',
                nameGap: 30,
                data: []
            },
            yAxis: {
                type: 'value',
                name: '损失',
                nameLocation: 'middle',
                nameGap: 40
            },
            series: [
                {
                    name: '训练损失',
                    type: 'line',
                    data: [],
                    smooth: true,
                    lineStyle: {
                        width: 2
                    }
                },
                {
                    name: '验证损失',
                    type: 'line',
                    data: [],
                    smooth: true,
                    lineStyle: {
                        width: 2
                    }
                }
            ],
            grid: {
                left: '10%',
                right: '10%',
                bottom: '15%',
                top: '15%'
            }
        };
        lossChart.setOption(lossOption);
    }
    
    // 初始化准确率图表
    if (document.getElementById('accuracyChart')) {
        accuracyChart = echarts.init(document.getElementById('accuracyChart'));
        const accOption = {
            title: {
                text: '训练/验证准确率',
                left: 'center'
            },
            tooltip: {
                trigger: 'axis'
            },
            legend: {
                data: ['训练准确率', '验证准确率'],
                bottom: 10
            },
            xAxis: {
                type: 'category',
                name: '轮次',
                nameLocation: 'middle',
                nameGap: 30,
                data: []
            },
            yAxis: {
                type: 'value',
                name: '准确率 (%)',
                nameLocation: 'middle',
                nameGap: 40,
                min: 0,
                max: 100
            },
            series: [
                {
                    name: '训练准确率',
                    type: 'line',
                    data: [],
                    smooth: true,
                    lineStyle: {
                        width: 2
                    }
                },
                {
                    name: '验证准确率',
                    type: 'line',
                    data: [],
                    smooth: true,
                    lineStyle: {
                        width: 2
                    }
                }
            ],
            grid: {
                left: '10%',
                right: '10%',
                bottom: '15%',
                top: '15%'
            }
        };
        accuracyChart.setOption(accOption);
    }
    
    // 初始化混淆矩阵图表
    if (document.getElementById('confusionMatrix')) {
        confusionMatrixChart = echarts.init(document.getElementById('confusionMatrix'));
        const matrixOption = {
            title: {
                text: '混淆矩阵',
                left: 'center'
            },
            tooltip: {
                position: 'top'
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: ['预测: 消极', '预测: 积极'],
                position: 'top'
            },
            yAxis: {
                type: 'category',
                data: ['实际: 积极', '实际: 消极'],
            },
            visualMap: {
                min: 0,
                max: 100,
                calculable: true,
                orient: 'horizontal',
                left: 'center',
                bottom: '0%',
                text: ['高', '低'],
                inRange: {
                    color: ['#f5f5f5', '#4575b4']
                }
            },
            series: [
                {
                    name: '混淆矩阵',
                    type: 'heatmap',
                    data: [
                        [0, 0, 0],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 1, 0]
                    ],
                    label: {
                        show: true
                    },
                    emphasis: {
                        itemStyle: {
                            shadowBlur: 10,
                            shadowColor: 'rgba(0, 0, 0, 0.5)'
                        }
                    }
                }
            ]
        };
        confusionMatrixChart.setOption(matrixOption);
    }
    
    // 窗口大小变化时调整图表大小
    window.addEventListener('resize', function() {
        if (lossChart) lossChart.resize();
        if (accuracyChart) accuracyChart.resize();
        if (confusionMatrixChart) confusionMatrixChart.resize();
    });
}

/**
 * 切换模型特定选项
 */
function toggleModelSpecificOptions(modelType) {
    const lstmOptions = document.querySelectorAll('.lstm-only');
    const cnnOptions = document.querySelectorAll('.cnn-only');
    
    if (modelType.includes('lstm') || modelType.includes('bilstm')) {
        lstmOptions.forEach(el => el.style.display = 'flex');
        cnnOptions.forEach(el => el.style.display = 'none');
    } else if (modelType === 'cnn') {
        lstmOptions.forEach(el => el.style.display = 'none');
        cnnOptions.forEach(el => el.style.display = 'flex');
    } else {
        lstmOptions.forEach(el => el.style.display = 'none');
        cnnOptions.forEach(el => el.style.display = 'none');
    }
}

/**
 * 启动训练过程
 */
async function startTraining() {
    // 重置训练数据
    trainingData = {
        epochs: [],
        trainLoss: [],
        valLoss: [],
        trainAcc: [],
        valAcc: []
    };
    
    // 收集表单参数
    const params = {
        model_type: document.getElementById('modelTypeSelect').value,
        batch_size: parseInt(document.getElementById('batchSizeSelect').value),
        epochs: parseInt(document.getElementById('epochsInput').value),
        learning_rate: parseFloat(document.getElementById('learningRateInput').value),
        dropout: parseFloat(document.getElementById('dropoutRange').value),
        optimizer: document.getElementById('optimizerSelect').value,
        weight_decay: parseFloat(document.getElementById('weightDecayInput').value),
    };
    
    // 根据模型类型添加特定参数
    if (params.model_type.includes('lstm') || params.model_type.includes('bilstm')) {
        params.hidden_dim = parseInt(document.getElementById('hiddenDimSelect').value);
        params.num_layers = parseInt(document.getElementById('numLayersSelect').value);
    } else if (params.model_type === 'cnn') {
        params.num_filters = parseInt(document.getElementById('numFiltersSelect').value);
    }
    
    // 如果启用了早停
    const earlyStopping = document.getElementById('earlyStopping');
    if (earlyStopping && earlyStopping.checked) {
        params.early_stopping = true;
        params.patience = parseInt(document.getElementById('patienceInput').value);
    }
    
    try {
        // 隐藏占位符和错误信息
        document.getElementById('trainingPlaceholder').classList.add('d-none');
        document.getElementById('errorMessage').style.display = 'none';
        
        // 显示训练进度区域
        document.getElementById('trainingProgress').classList.remove('d-none');
        document.getElementById('trainingControls').classList.remove('d-none');
        document.getElementById('modelEvaluation').classList.add('d-none');
        
        // 更新UI
        document.getElementById('trainingStatus').textContent = '初始化中';
        document.getElementById('trainingStatus').className = 'badge bg-secondary';
        document.getElementById('currentEpoch').textContent = `0/${params.epochs}`;
        document.getElementById('currentLoss').textContent = '-';
        document.getElementById('currentAccuracy').textContent = '-';
        document.getElementById('remainingTime').textContent = '计算中...';
        document.getElementById('totalProgressBar').style.width = '0%';
        document.getElementById('totalProgressBar').textContent = '0%';
        
        // 清空训练日志
        const logContent = document.querySelector('.log-content');
        if (logContent) logContent.innerHTML = '';
        
        // 清空图表
        if (lossChart) {
            lossChart.setOption({
                xAxis: { data: [] },
                series: [{ data: [] }, { data: [] }]
            });
        }
        if (accuracyChart) {
            accuracyChart.setOption({
                xAxis: { data: [] },
                series: [{ data: [] }, { data: [] }]
            });
        }
        
        // 禁用表单
        toggleFormElements(false);
        
        // 设置训练状态
        trainingInProgress = true;
        trainingStartTime = new Date();
        totalEpochs = params.epochs;
        currentEpoch = 0;
        
        // 添加训练日志
        addTrainingLog('info', '准备启动训练任务...');
        
        // 发送训练请求
        const response = await fetch('/api/train/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || '启动训练失败');
        }
        
        // 成功启动训练
        const trainingId = data.training_id || "default";
        
        document.getElementById('trainingStatus').textContent = '训练中';
        document.getElementById('trainingStatus').className = 'badge bg-primary';
        addTrainingLog('success', `训练任务已启动 (PID: ${data.pid || 'N/A'})`);
        
        // 设置轮询获取训练进度
        trainingInterval = setInterval(() => {
            // 使用AJAX检查训练状态和进度
            checkTrainingProgress();
        }, 3000);
        
    } catch (error) {
        console.error('启动训练出错:', error);
        document.getElementById('errorMessage').textContent = '启动训练失败: ' + error.message;
        document.getElementById('errorMessage').style.display = 'block';
        
        addTrainingLog('error', '启动训练失败: ' + error.message);
        
        // 恢复UI
        toggleFormElements(true);
        trainingInProgress = false;
    }
}

/**
 * 检查训练进度
 */
async function checkTrainingProgress() {
    try {
        const response = await fetch('/api/train/progress');
        if (response.ok) {
            const data = await response.json();
            console.log("进度数据:", data); // 添加日志调试
            
            // 更新进度
            updateTrainingProgress(data);
            
            // 如果训练已完成或失败，停止轮询
            if (data.status === 'completed' || data.status === 'failed' || data.status === 'stopped') {
                clearInterval(trainingInterval);
                finishTraining(data);
                
                // 如果训练失败，显示错误消息
                if (data.status === 'failed' && data.error) {
                    showError(data.error);
                    addTrainingLog('error', `训练失败: ${data.error}`);
                } else if (data.status === 'stopped') {
                    addTrainingLog('warning', '训练已停止');
                }
            }
            
            // 显示最近的日志
            if (data.recent_logs && Array.isArray(data.recent_logs)) {
                data.recent_logs.forEach(log => {
                    // 避免重复显示同样的日志
                    if (!recentLogMessages.includes(log)) {
                        addTrainingLog('info', log);
                        recentLogMessages.push(log);
                        // 保持日志列表在合理大小
                        if (recentLogMessages.length > 50) {
                            recentLogMessages.shift();
                        }
                    }
                });
            }
        } else {
            console.error("进度API响应错误:", response.status);
            try {
                const errorData = await response.json();
                console.error("错误详情:", errorData);
            } catch (e) {
                console.error("无法解析错误响应");
            }
        }
    } catch (error) {
        console.error('检查训练进度出错:', error);
        // 添加错误到训练日志，但不停止轮询，以便重试
        addTrainingLog('error', `检查训练进度出错: ${error.message}`);
    }
}

// 用于跟踪已显示的日志消息
const recentLogMessages = [];

/**
 * 更新训练进度
 */
function updateTrainingProgress(data) {
    if (!data) return;
    
    // 更新当前轮次
    if (data.current_epoch !== undefined) {
        currentEpoch = data.current_epoch;
        totalEpochs = data.total_epochs || totalEpochs;
        document.getElementById('currentEpoch').textContent = `${currentEpoch}/${totalEpochs}`;
        
        // 更新进度条
        const progress = Math.round((currentEpoch / totalEpochs) * 100);
        const progressBar = document.getElementById('totalProgressBar');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.textContent = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
        }
    }
    
    // 更新损失和准确率
    if (data.train_loss !== undefined && data.train_loss !== null) {
        document.getElementById('currentLoss').textContent = 
            typeof data.train_loss === 'number' ? data.train_loss.toFixed(4) : data.train_loss;
    }
    
    if (data.val_acc !== undefined && data.val_acc !== null) {
        document.getElementById('currentAccuracy').textContent = 
            typeof data.val_acc === 'number' ? (data.val_acc * 100).toFixed(2) + '%' : data.val_acc;
    }
    
    // 如果有消息，添加到训练日志
    if (data.message && data.message !== lastMessage) {
        addTrainingLog('info', data.message);
        lastMessage = data.message; // 避免重复显示相同的消息
    }
    
    // 如果有错误信息，添加到训练日志
    if (data.error && data.error !== lastError) {
        addTrainingLog('error', data.error);
        lastError = data.error;
    }
    
    // 更新图表数据
    if (data.history && data.history.train_loss && data.history.train_loss.length > 0) {
        const epochs = Array.from({length: data.history.train_loss.length}, (_, i) => i + 1);
        
        if (lossChart) {
            lossChart.setOption({
                xAxis: {
                    data: epochs
                },
                series: [
                    {
                        name: '训练损失',
                        data: data.history.train_loss
                    },
                    {
                        name: '验证损失',
                        data: data.history.val_loss || []
                    }
                ]
            });
        }
        
        if (accuracyChart && data.history.val_acc) {
            accuracyChart.setOption({
                xAxis: {
                    data: epochs
                },
                series: [
                    {
                        name: '训练准确率',
                        data: (data.history.train_acc || []).map(v => typeof v === 'number' ? v * 100 : 0)
                    },
                    {
                        name: '验证准确率',
                        data: (data.history.val_acc || []).map(v => typeof v === 'number' ? v * 100 : 0)
                    }
                ]
            });
        }
    }
    
    // 更新状态
    if (data.status) {
        const statusEl = document.getElementById('trainingStatus');
        if (statusEl) {
            statusEl.textContent = getStatusText(data.status);
            statusEl.className = getStatusClass(data.status);
        }
    }
    
    // 更新估计剩余时间
    if (trainingStartTime && currentEpoch > 0 && totalEpochs > 0 && currentEpoch < totalEpochs) {
        const elapsedTime = (new Date() - trainingStartTime) / 1000; // 秒
        const timePerEpoch = elapsedTime / currentEpoch;
        const remainingEpochs = totalEpochs - currentEpoch;
        const remainingSeconds = Math.round(timePerEpoch * remainingEpochs);
        
        document.getElementById('remainingTime').textContent = formatTime(remainingSeconds);
    } else if (currentEpoch >= totalEpochs) {
        document.getElementById('remainingTime').textContent = '00:00';
    }
}

// 存储最后显示的消息和错误
let lastMessage = '';
let lastError = '';

/**
 * 获取状态文本
 */
function getStatusText(status) {
    const statusMap = {
        'initializing': '初始化中',
        'preparing': '准备数据',
        'running': '训练中',
        'paused': '已暂停',
        'stopping': '正在停止',
        'stopped': '已停止',
        'completed': '已完成',
        'failed': '训练失败'
    };
    return statusMap[status] || status;
}

/**
 * 获取状态CSS类
 */
function getStatusClass(status) {
    const classMap = {
        'initializing': 'badge bg-secondary',
        'preparing': 'badge bg-info',
        'running': 'badge bg-primary',
        'paused': 'badge bg-warning',
        'stopping': 'badge bg-warning',
        'stopped': 'badge bg-secondary',
        'completed': 'badge bg-success',
        'failed': 'badge bg-danger'
    };
    return classMap[status] || 'badge bg-secondary';
}

/**
 * 显示错误消息
 * @param {string} message - 错误消息
 */
function showError(message) {
    const errorElement = document.getElementById('errorMessage');
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }
}

/**
 * 暂停训练
 */
async function pauseTraining() {
    try {
        const response = await fetch('/api/train/pause', {
            method: 'POST'
        });
        
        if (response.ok) {
            const data = await response.json();
            const pauseBtn = document.getElementById('pauseTrainingBtn');
            
            if (data.status === 'paused') {
                pauseBtn.innerHTML = '<i class="bi bi-play-fill"></i> 继续';
                document.getElementById('trainingStatus').textContent = '已暂停';
                document.getElementById('trainingStatus').className = 'badge bg-warning';
                addTrainingLog('warning', '训练已暂停');
            } else if (data.status === 'resumed') {
                pauseBtn.innerHTML = '<i class="bi bi-pause-fill"></i> 暂停';
                document.getElementById('trainingStatus').textContent = '训练中';
                document.getElementById('trainingStatus').className = 'badge bg-primary';
                addTrainingLog('info', '训练已继续');
            }
        } else {
            const data = await response.json();
            addTrainingLog('error', data.error || '暂停训练失败');
        }
    } catch (error) {
        console.error('暂停/继续训练出错:', error);
        addTrainingLog('error', '暂停/继续训练失败: ' + error.message);
    }
}

/**
 * 停止训练
 */
async function stopTraining() {
    if (!trainingInProgress) return;
    
    try {
        const response = await fetch('/api/train/stop', {
            method: 'POST'
        });
        
        if (response.ok) {
            document.getElementById('trainingStatus').textContent = '正在停止';
            document.getElementById('trainingStatus').className = 'badge bg-warning';
            addTrainingLog('warning', '正在停止训练...');
        } else {
            const data = await response.json();
            addTrainingLog('error', data.error || '停止训练失败');
        }
    } catch (error) {
        console.error('停止训练出错:', error);
        addTrainingLog('error', '停止训练失败: ' + error.message);
    }
}

/**
 * 保存模型
 */
async function saveModel() {
    try {
        const modelName = prompt('请输入模型名称:', `model_${new Date().toISOString().slice(0,10)}`);
        
        if (!modelName) return;
        
        const response = await fetch('/api/train/save_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name: modelName })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            alert(`模型已成功保存为: ${data.filename || modelName}`);
            addTrainingLog('success', `模型已保存为: ${data.filename || modelName}`);
        } else {
            alert('保存模型失败: ' + (data.error || '未知错误'));
            addTrainingLog('error', '保存模型失败: ' + (data.error || '未知错误'));
        }
    } catch (error) {
        console.error('保存模型出错:', error);
        addTrainingLog('error', '保存模型失败: ' + error.message);
    }
}

/**
 * 完成训练
 */
function finishTraining(evalData) {
    // 清除轮询
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
    
    // 重置训练状态
    trainingInProgress = false;
    
    // 恢复表单
    toggleFormElements(true);
    
    // 更新UI
    const status = evalData && evalData.status ? evalData.status : 'completed';
    document.getElementById('trainingStatus').textContent = getStatusText(status);
    document.getElementById('trainingStatus').className = getStatusClass(status);
    
    if (status === 'completed') {
        addTrainingLog('success', '训练完成！');
        
        // 显示评估结果
        if (evalData && evalData.results) {
            displayEvaluationResults(evalData.results);
        }
    } else if (status === 'failed') {
        addTrainingLog('error', '训练失败: ' + (evalData.error || '未知错误'));
    } else {
        addTrainingLog('warning', '训练已停止');
    }
}

/**
 * 显示评估结果
 */
function displayEvaluationResults(results) {
    const evalSection = document.getElementById('modelEvaluation');
    if (!evalSection) return;
    
    evalSection.classList.remove('d-none');
    
    // 更新指标
    document.getElementById('testAccuracy').textContent = (results.accuracy * 100).toFixed(2) + '%';
    document.getElementById('f1Score').textContent = (results.f1_score * 100).toFixed(2) + '%';
    document.getElementById('recall').textContent = (results.recall * 100).toFixed(2) + '%';
    document.getElementById('precision').textContent = (results.precision * 100).toFixed(2) + '%';
    
    // 更新混淆矩阵
    if (results.confusion_matrix && confusionMatrixChart) {
        const cm = results.confusion_matrix;
        const matrixData = [
            [0, 0, cm[0][0]],
            [0, 1, cm[0][1]],
            [1, 0, cm[1][0]],
            [1, 1, cm[1][1]]
        ];
        
        const maxValue = Math.max(
            cm[0][0], cm[0][1], 
            cm[1][0], cm[1][1]
        );
        
        confusionMatrixChart.setOption({
            visualMap: {
                min: 0,
                max: maxValue
            },
            series: [{
                data: matrixData
            }]
        });
    }
}

/**
 * 添加训练日志条目
 */
function addTrainingLog(type, message) {
    const logContent = document.querySelector('.log-content');
    if (logContent) {
        const timestamp = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.innerHTML = `[${timestamp}] ${message}`;
        logContent.appendChild(entry);
        
        // 滚动到底部
        const trainingLog = document.getElementById('trainingLog');
        if (trainingLog) {
            trainingLog.scrollTop = trainingLog.scrollHeight;
        }
    }
}

/**
 * 启用/禁用表单元素
 */
function toggleFormElements(enabled) {
    const form = document.getElementById('trainingForm');
    if (form) {
        const elements = form.elements;
        for (let i = 0; i < elements.length; i++) {
            elements[i].disabled = !enabled;
        }
    }
    
    // 更新开始训练按钮状态
    const startBtn = document.getElementById('startTrainingBtn');
    if (startBtn) {
        if (enabled) {
            startBtn.classList.remove('disabled');
            startBtn.innerHTML = '<i class="bi bi-play-fill"></i> 开始训练';
        } else {
            startBtn.classList.add('disabled');
            startBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> 训练中...';
        }
    }
    
    // 训练控制按钮
    const controlBtns = document.querySelectorAll('#trainingControls button');
    controlBtns.forEach(btn => {
        if (btn.id === 'saveModelBtn') {
            // 只有在训练完成后才能保存模型
            btn.disabled = !enabled;
        } else {
            // 暂停和停止按钮在训练中才能使用
            btn.disabled = enabled;
        }
    });
}

/**
 * 格式化时间（秒 -> mm:ss）
 */
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}
