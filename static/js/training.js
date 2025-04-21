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
    
    // 表单提交处理
    const trainingForm = document.getElementById('trainingForm');
    if (trainingForm) {
        trainingForm.addEventListener('submit', function(e) {
            e.preventDefault();
            if (!trainingInProgress) {
                startTraining();
            }
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
        lstmOptions.forEach(el => el.style.display = 'block');
        cnnOptions.forEach(el => el.style.display = 'none');
    } else if (modelType === 'cnn') {
        lstmOptions.forEach(el => el.style.display = 'none');
        cnnOptions.forEach(el => el.style.display = 'block');
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
        learning_rate: parseFloat(document.getElementById('learningRateInput').value),
        dropout: parseFloat(document.getElementById('dropoutRange').value),
        epochs: parseInt(document.getElementById('epochsInput').value),
        hidden_dim: parseInt(document.getElementById('hiddenDimSelect').value),
        optimizer: document.getElementById('optimizerSelect').value,
        weight_decay: parseFloat(document.getElementById('weightDecayInput').value),
        early_stopping: document.getElementById('earlyStopping').checked,
        patience: parseInt(document.getElementById('patienceInput').value)
    };
    
    // 根据模型类型添加特定参数
    if (params.model_type.includes('lstm') || params.model_type.includes('bilstm')) {
        params.num_layers = parseInt(document.getElementById('numLayersSelect').value);
    } else if (params.model_type === 'cnn') {
        params.num_filters = parseInt(document.getElementById('numFiltersSelect').value);
    }
    
    try {
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
        const trainingId = data.training_id;
        totalEpochs = params.epochs;
        currentEpoch = 0;
        
        // 更新UI显示
        document.getElementById('trainingPlaceholder').classList.add('d-none');
        document.getElementById('trainingProgress').classList.remove('d-none');
        document.getElementById('trainingControls').classList.remove('d-none');
        document.getElementById('modelEvaluation').classList.add('d-none');
        document.getElementById('trainingStatus').textContent = '训练中';
        document.getElementById('trainingStatus').className = 'badge bg-primary';
        document.getElementById('currentEpoch').textContent = `0/${totalEpochs}`;
        
        // 禁用表单
        toggleFormElements(false);
        
        // 记录训练开始时间
        trainingStartTime = new Date();
        
        // 设置训练状态
        trainingInProgress = true;
        
        // 连接WebSocket接收实时训练更新
        setupTrainingSocket(trainingId);
        
        // 添加训练日志
        addTrainingLog('info', '训练任务已启动...');
        
        // 开始定时查询训练状态
        trainingInterval = setInterval(() => {
            checkTrainingStatus(trainingId);
        }, 5000);
        
    } catch (error) {
        console.error('启动训练出错:', error);
        showError('启动训练失败: ' + error.message);
    }
}

/**
 * 设置WebSocket连接接收训练更新
 */
function setupTrainingSocket(trainingId) {
    // 关闭之前的连接
    if (trainingSocket) {
        trainingSocket.close();
    }
    
    // 创建WebSocket连接
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/training/${trainingId}`;
    
    trainingSocket = new WebSocket(wsUrl);
    
    trainingSocket.onopen = function() {
        addTrainingLog('success', 'WebSocket连接已建立');
    };
    
    trainingSocket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        processTrainingUpdate(data);
    };
    
    trainingSocket.onerror = function(error) {
        console.error('WebSocket错误:', error);
        addTrainingLog('error', 'WebSocket连接错误');
    };
    
    trainingSocket.onclose = function() {
        addTrainingLog('info', 'WebSocket连接已关闭');
    };
}

/**
 * 处理训练更新数据
 */
function processTrainingUpdate(data) {
    switch(data.type) {
        case 'epoch_start':
            currentEpoch = data.epoch;
            addTrainingLog('info', `开始第 ${currentEpoch}/${totalEpochs} 轮训练...`);
            document.getElementById('currentEpoch').textContent = `${currentEpoch}/${totalEpochs}`;
            updateProgressBars();
            break;
            
        case 'epoch_end':
            // 更新损失和准确率
            document.getElementById('currentLoss').textContent = data.train_loss.toFixed(4);
            document.getElementById('currentAccuracy').textContent = (data.val_acc * 100).toFixed(2) + '%';
            
            // 更新图表数据
            trainingData.epochs.push(data.epoch);
            trainingData.trainLoss.push(data.train_loss);
            trainingData.valLoss.push(data.val_loss);
            trainingData.trainAcc.push(data.train_acc * 100);
            trainingData.valAcc.push(data.val_acc * 100);
            
            updateCharts();
            
            addTrainingLog('success', `第 ${data.epoch}/${totalEpochs} 轮训练完成 - 训练损失: ${data.train_loss.toFixed(4)}, 验证准确率: ${(data.val_acc * 100).toFixed(2)}%`);
            break;
            
        case 'early_stopping':
            addTrainingLog('warning', `触发早停机制，在第 ${data.epoch} 轮后停止训练`);
            break;
            
        case 'completed':
            finishTraining(data);
            break;
            
        case 'error':
            showError(data.message);
            addTrainingLog('error', data.message);
            stopTraining();
            break;
            
        default:
            console.log('未知更新类型:', data);
    }
    
    // 更新剩余时间估计
    if (trainingStartTime && currentEpoch > 0) {
        const elapsedTime = (new Date() - trainingStartTime) / 1000; // 秒
        const timePerEpoch = elapsedTime / currentEpoch;
        const remainingEpochs = totalEpochs - currentEpoch;
        const remainingSeconds = Math.round(timePerEpoch * remainingEpochs);
        
        document.getElementById('remainingTime').textContent = formatTime(remainingSeconds);
    }
}

/**
 * 更新进度条
 */
function updateProgressBars() {
    const totalProgress = Math.round((currentEpoch / totalEpochs) * 100);
    document.getElementById('totalProgressBar').style.width = `${totalProgress}%`;
    document.getElementById('totalProgressBar').textContent = `${totalProgress}%`;
    document.getElementById('totalProgressBar').setAttribute('aria-valuenow', totalProgress);
}

/**
 * 更新图表
 */
function updateCharts() {
    if (lossChart) {
        lossChart.setOption({
            xAxis: {
                data: trainingData.epochs
            },
            series: [
                {
                    name: '训练损失',
                    data: trainingData.trainLoss
                },
                {
                    name: '验证损失',
                    data: trainingData.valLoss
                }
            ]
        });
    }
    
    if (accuracyChart) {
        accuracyChart.setOption({
            xAxis: {
                data: trainingData.epochs
            },
            series: [
                {
                    name: '训练准确率',
                    data: trainingData.trainAcc
                },
                {
                    name: '验证准确率',
                    data: trainingData.valAcc
                }
            ]
        });
    }
}

/**
 * 检查训练状态
 */
async function checkTrainingStatus(trainingId) {
    try {
        const response = await fetch(`/api/train/status/${trainingId}`);
        const data = await response.json();
        
        if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(trainingInterval);
        }
        
        if (data.status === 'failed' && !data.error_reported) {
            showError(`训练失败: ${data.error}`);
            addTrainingLog('error', `训练失败: ${data.error}`);
            stopTraining();
        }
    } catch (error) {
        console.error('检查训练状态出错:', error);
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
            const pauseBtn = document.getElementById('pauseTrainingBtn');
            
            if (pauseBtn.innerHTML.includes('暂停')) {
                pauseBtn.innerHTML = '<i class="bi bi-play-fill"></i> 继续';
                document.getElementById('trainingStatus').textContent = '已暂停';
                document.getElementById('trainingStatus').className = 'badge bg-warning';
                addTrainingLog('warning', '训练已暂停');
            } else {
                pauseBtn.innerHTML = '<i class="bi bi-pause-fill"></i> 暂停';
                document.getElementById('trainingStatus').textContent = '训练中';
                document.getElementById('trainingStatus').className = 'badge bg-primary';
                addTrainingLog('info', '训练已继续');
            }
        } else {
            const data = await response.json();
            showError(data.error || '暂停训练失败');
        }
    } catch (error) {
        console.error('暂停/继续训练出错:', error);
        showError('暂停/继续训练失败: ' + error.message);
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
            finishTraining();
        } else {
            const data = await response.json();
            showError(data.error || '停止训练失败');
        }
    } catch (error) {
        console.error('停止训练出错:', error);
        showError('停止训练失败: ' + error.message);
    }
}

/**
 * 完成训练
 */
function finishTraining(evalData) {
    // 清除定时检查
    clearInterval(trainingInterval);
    
    // 关闭WebSocket连接
    if (trainingSocket) {
        trainingSocket.close();
        trainingSocket = null;
    }
    
    // 重置训练状态
    trainingInProgress = false;
    
    // 更新UI
    document.getElementById('trainingStatus').textContent = '已完成';
    document.getElementById('trainingStatus').className = 'badge bg-success';
    document.getElementById('pauseTrainingBtn').innerHTML = '<i class="bi bi-pause-fill"></i> 暂停';
    toggleFormElements(true);
    
    // 添加训练日志
    addTrainingLog('success', '训练完成！');
    
    // 如果有评估数据，显示模型评估
    if (evalData) {
        displayModelEvaluation(evalData);
    }
}

/**
 * 显示模型评估结果
 */
function displayModelEvaluation(data) {
    // 显示评估面板
    document.getElementById('modelEvaluation').classList.remove('d-none');
    
    // 更新指标
    document.getElementById('testAccuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
    document.getElementById('f1Score').textContent = (data.f1_score * 100).toFixed(2) + '%';
    document.getElementById('recall').textContent = (data.recall * 100).toFixed(2) + '%';
    document.getElementById('precision').textContent = (data.precision * 100).toFixed(2) + '%';
    
    // 更新混淆矩阵
    if (confusionMatrixChart && data.confusion_matrix) {
        // 假设二分类问题，混淆矩阵为2x2
        const cm = data.confusion_matrix;
        
        // 数据格式：[行, 列, 值]
        const matrixData = [
            [0, 0, cm[0][0]],
            [0, 1, cm[0][1]],
            [1, 0, cm[1][0]],
            [1, 1, cm[1][1]]
        ];
        
        // 计算最大值以设置visualMap范围
        const maxValue = Math.max(...cm[0], ...cm[1]);
        
        confusionMatrixChart.setOption({
            visualMap: {
                min: 0,
                max: maxValue
            },
            series: [
                {
                    name: '混淆矩阵',
                    data: matrixData
                }
            ]
        });
    }
}

/**
 * 保存训练后的模型
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
            alert(`模型已成功保存为: ${data.filename}`);
            addTrainingLog('success', `模型已保存为: ${data.filename}`);
        } else {
            showError(data.error || '保存模型失败');
        }
    } catch (error) {
        console.error('保存模型出错:', error);
        showError('保存模型失败: ' + error.message);
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
}

/**
 * 格式化时间（秒 -> mm:ss）
 */
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * 显示错误消息
 */
function showError(message) {
    alert('错误: ' + message);
}
