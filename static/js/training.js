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
document.addEventListener('DOMContentLoaded', function () {
    console.log("页面加载完成，准备初始化...");

    // 延迟初始化图表，确保DOM完全可见
    setTimeout(function () {
        const trainingProgress = document.getElementById('trainingProgress');
        if (trainingProgress) {
            console.log("训练进度区域是否可见:",
                trainingProgress.style.display !== 'none',
                trainingProgress.offsetHeight > 0);

            // 如果需要，暂时强制显示以便初始化图表
            const wasHidden = trainingProgress.classList.contains('d-none');
            if (wasHidden) {
                trainingProgress.classList.remove('d-none');
                console.log("暂时显示训练进度区域以初始化图表");
            }

            // 初始化图表
            initCharts();

            // 如果之前是隐藏的，恢复隐藏状态
            if (wasHidden) {
                setTimeout(() => {
                    trainingProgress.classList.add('d-none');
                    console.log("恢复训练进度区域的隐藏状态");
                }, 1000);
            }
        }
    }, 300);

    // 添加Dropout滑块变更事件
    initDropoutSlider();

    // 确保开始训练按钮绑定了事件
    const startTrainingBtn = document.getElementById('startTrainingBtn');
    if (startTrainingBtn) {
        console.log('找到开始训练按钮，添加点击事件');
        // 移除可能存在的内联onclick属性，避免冲突
        startTrainingBtn.removeAttribute('onclick');
        startTrainingBtn.addEventListener('click', function() {
            console.log('开始训练按钮点击');
            startTraining();
        });
    } else {
        console.error('找不到开始训练按钮');
    }

    // 显示/隐藏模型特定选项
    const modelTypeSelect = document.getElementById('modelTypeSelect');
    if (modelTypeSelect) {
        modelTypeSelect.addEventListener('change', function () {
            toggleModelSpecificOptions(this.value);
        });
        // 初始化显示/隐藏
        toggleModelSpecificOptions(modelTypeSelect.value);
    }

    // 早停选项切换
    const earlyStopping = document.getElementById('earlyStopping');
    const earlyStoppingOptions = document.getElementById('earlyStoppingOptions');
    if (earlyStopping && earlyStoppingOptions) {
        earlyStopping.addEventListener('change', function () {
            earlyStoppingOptions.style.display = this.checked ? 'block' : 'none';
        });
    }

    // 训练控制按钮事件绑定
    const pauseTrainingBtn = document.getElementById('pauseTrainingBtn');
    const stopTrainingBtn = document.getElementById('stopTrainingBtn');
    const saveModelBtn = document.getElementById('saveModelBtn');

    if (pauseTrainingBtn) {
        pauseTrainingBtn.addEventListener('click', function () {
            if (trainingInProgress) {
                pauseTraining();
            }
        });
    }

    if (stopTrainingBtn) {
        stopTrainingBtn.addEventListener('click', function () {
            if (trainingInProgress) {
                stopTraining();
            }
        });
    }

    if (saveModelBtn) {
        saveModelBtn.addEventListener('click', saveModel);
    }

    // 初始化模型选择卡片和同步选择框
    initModelTypeSelection();
    
    // 初始化权重衰减范围滑块
    initWeightDecayRange();
    
    // 初始化Bootstrap工具提示
    initTooltips();
    
    // 初始化模型类型卡片选择器
    initModelTypeCards();
});

/**
 * 初始化模型类型卡片选择器
 */
function initModelTypeCards() {
    const modelCards = document.querySelectorAll('.model-type-card');
    const modelTypeSelect = document.getElementById('modelTypeSelect');
    
    if (modelCards.length && modelTypeSelect) {
        // 设置初始选中状态
        const initialValue = modelTypeSelect.value;
        modelCards.forEach(card => {
            const radioInput = card.querySelector('input[type="radio"]');
            if (radioInput && radioInput.value === initialValue) {
                card.classList.add('selected');
                radioInput.checked = true;
            }
            
            // 添加点击事件
            card.addEventListener('click', function() {
                // 移除所有卡片的选中状态
                modelCards.forEach(c => c.classList.remove('selected'));
                
                // 设置当前卡片为选中状态
                this.classList.add('selected');
                
                // 选中单选按钮
                const radio = this.querySelector('input[type="radio"]');
                if (radio) {
                    radio.checked = true;
                    
                    // 更新隐藏的select值并触发change事件
                    modelTypeSelect.value = radio.value;
                    modelTypeSelect.dispatchEvent(new Event('change'));
                }
            });
        });
    }
}

/**
 * 初始化图表
 */
function initCharts() {
    console.log("正在初始化图表...");
    
    try {
        // 完全延迟初始化，确保DOM已完全加载和渲染
        setTimeout(() => {
            console.log("开始延迟初始化图表...");
            initLossChart();
            initAccuracyChart();

            // 再次触发窗口resize事件，确保图表正确渲染
            if (typeof window !== 'undefined') {
                window.dispatchEvent(new Event('resize'));
            }
        }, 500);
    } catch (error) {
        console.error("初始化图表出错:", error);
    }
}

/**
 * 初始化损失图表
 */
function initLossChart() {
    const lossChartContainer = document.getElementById('lossChart');
    if (!lossChartContainer) {
        console.error("找不到lossChart容器");
        return;
    }

    console.log("损失图表容器尺寸:", lossChartContainer.offsetWidth, lossChartContainer.offsetHeight);

    // 如果容器尺寸为0，添加内联样式确保可见
    if (lossChartContainer.offsetWidth < 10 || lossChartContainer.offsetHeight < 10) {
        console.warn("损失图表容器尺寸过小，添加内联样式");
        lossChartContainer.style.width = '100%';
        lossChartContainer.style.height = '300px';
        lossChartContainer.style.minHeight = '300px';
    }

    try {
        // 销毁旧实例（如果存在）
        if (lossChart) {
            lossChart.dispose();
        }

        // 最简单的配置，确保能正确初始化
        lossChart = echarts.init(lossChartContainer);
        const baseOption = {
            grid: {
                top: 60,
                right: 40,
                bottom: 60,
                left: 60,
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: [1, 2, 3]  // 放一些示例数据先初始化
            },
            yAxis: {
                type: 'value'
            },
            series: [
                {
                    name: '训练损失',
                    type: 'line',
                    data: [0.5, 0.4, 0.3],
                    lineStyle: {
                        width: 3,
                        color: '#4e73df'
                    },
                    itemStyle: {
                        color: '#4e73df'
                    },
                    symbolSize: 6
                },
                {
                    name: '验证损失',
                    type: 'line',
                    data: [0.6, 0.5, 0.4],
                    lineStyle: {
                        width: 3,
                        color: '#e74a3b'
                    },
                    itemStyle: {
                        color: '#e74a3b'
                    },
                    symbolSize: 6
                }
            ]
        };

        // 先用简单配置初始化
        lossChart.setOption(baseOption);
        console.log("损失图表基础初始化成功");

        // 再设置完整配置
        const fullOption = {
            title: {
                text: '训练/验证损失',
                left: 'center',
                textStyle: {
                    fontSize: 14,
                    fontWeight: 'normal',
                    color: '#495057'
                }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                borderColor: '#e0e0e0',
                borderWidth: 1,
                textStyle: {
                    color: '#333'
                },
                axisPointer: {
                    type: 'cross',
                    lineStyle: {
                        color: '#aaa'
                    }
                }
            },
            legend: {
                data: ['训练损失', '验证损失'],
                bottom: 10,
                icon: 'circle',
                itemWidth: 10,
                itemHeight: 10,
                textStyle: {
                    color: '#666'
                }
            }
        };

        // 增量更新配置
        lossChart.setOption(fullOption);
        console.log("损失图表完整配置已应用");

        // 重新调整大小
        lossChart.resize();
    } catch (e) {
        console.error("初始化损失图表出错:", e);
    }
}

/**
 * 初始化准确率图表
 */
function initAccuracyChart() {
    const accuracyChartContainer = document.getElementById('accuracyChart');
    if (!accuracyChartContainer) {
        console.error("找不到accuracyChart容器");
        return;
    }

    console.log("准确率图表容器尺寸:", accuracyChartContainer.offsetWidth, accuracyChartContainer.offsetHeight);

    // 如果容器尺寸为0，添加内联样式确保可见
    if (accuracyChartContainer.offsetWidth < 10 || accuracyChartContainer.offsetHeight < 10) {
        console.warn("准确率图表容器尺寸过小，添加内联样式");
        accuracyChartContainer.style.width = '100%';
        accuracyChartContainer.style.height = '300px';
        accuracyChartContainer.style.minHeight = '300px';
    }

    try {
        // 销毁旧实例（如果存在）
        if (accuracyChart) {
            accuracyChart.dispose();
        }

        // 最简单的配置，确保能正确初始化
        accuracyChart = echarts.init(accuracyChartContainer);
        const baseOption = {
            grid: {
                top: 60,
                right: 40,
                bottom: 60,
                left: 60,
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: [1, 2, 3]  // 放一些示例数据先初始化
            },
            yAxis: {
                type: 'value'
            },
            series: [
                {
                    name: '训练准确率',
                    type: 'line',
                    data: [80, 85, 90],
                    lineStyle: {
                        width: 3,
                        color: '#1cc88a'
                    },
                    itemStyle: {
                        color: '#1cc88a'
                    },
                    symbolSize: 6
                },
                {
                    name: '验证准确率',
                    type: 'line',
                    data: [75, 80, 85],
                    lineStyle: {
                        width: 3,
                        color: '#36b9cc'
                    },
                    itemStyle: {
                        color: '#36b9cc'
                    },
                    symbolSize: 6
                }
            ]
        };

        // 先用简单配置初始化
        accuracyChart.setOption(baseOption);
        console.log("准确率图表基础初始化成功");

        // 再设置完整配置
        const fullOption = {
            title: {
                text: '训练/验证准确率',
                left: 'center',
                textStyle: {
                    fontSize: 14,
                    fontWeight: 'normal',
                    color: '#495057'
                }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                borderColor: '#e0e0e0',
                borderWidth: 1,
                textStyle: {
                    color: '#333'
                },
                axisPointer: {
                    type: 'cross',
                    lineStyle: {
                        color: '#aaa'
                    }
                }
            },
            legend: {
                data: ['训练准确率', '验证准确率'],
                bottom: 10,
                icon: 'circle',
                itemWidth: 10,
                itemHeight: 10,
                textStyle: {
                    color: '#666'
                }
            },
            yAxis: {
                min: 0,
                max: 100,
                name: '准确率 (%)',
                axisLabel: {
                    formatter: '{value}%'
                }
            }
        };

        // 增量更新配置
        accuracyChart.setOption(fullOption);
        console.log("准确率图表完整配置已应用");

        // 重新调整大小
        accuracyChart.resize();
    } catch (e) {
        console.error("初始化准确率图表出错:", e);
    }
}

/**
 * 触发图表重新调整大小
 */
function triggerResize() {
    // 执行多次resize确保图表正确渲染
    setTimeout(() => {
        if (lossChart) {
            try {
                lossChart.resize();
            } catch (e) {
                console.warn("重绘损失图表出错", e);
            }
        }
        if (accuracyChart) {
            try {
                accuracyChart.resize();
            } catch (e) {
                console.warn("重绘准确率图表出错", e);
            }
        }
        if (confusionMatrixChart) {
            try {
                confusionMatrixChart.resize();
            } catch (e) {
                console.warn("重绘混淆矩阵图表出错", e);
            }
        }
    }, 300);

    // 额外的resize以防止渲染问题
    setTimeout(() => {
        if (lossChart) {
            try {
                lossChart.resize();
            } catch (e) {
                console.warn("重绘损失图表出错", e);
            }
        }
        if (accuracyChart) {
            try {
                accuracyChart.resize();
            } catch (e) {
                console.warn("重绘准确率图表出错", e);
            }
        }
    }, 600);
}

/**
 * 切换模型特定选项
 */
function toggleModelSpecificOptions(modelType) {
    const lstmOptions = document.querySelectorAll('.lstm-only');
    const cnnOptions = document.querySelectorAll('.cnn-only');

    if (modelType.includes('lstm') || modelType.includes('bilstm')) {
        // 显示LSTM选项并添加动画
        lstmOptions.forEach(el => {
            el.style.display = 'block';
            // 使用setTimeout确保CSS过渡生效
            setTimeout(() => {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }, 50);
        });
        
        // 隐藏CNN选项
        cnnOptions.forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(10px)';
            // 使用setTimeout确保CSS过渡完成后再隐藏
            setTimeout(() => {
                el.style.display = 'none';
            }, 300);
        });
        
        // 更新隐藏层维度的badge显示
        const hiddenDimSelect = document.getElementById('hiddenDimSelect');
        if (hiddenDimSelect) {
            const badges = document.querySelectorAll('.dimension-indicator .badge');
            badges.forEach(badge => {
                if (badge.closest('.lstm-param-grid')) {
                    badge.textContent = hiddenDimSelect.value + '维向量';
                }
            });
            
            // 添加变更事件
            hiddenDimSelect.addEventListener('change', function() {
                const badges = document.querySelectorAll('.dimension-indicator .badge');
                badges.forEach(badge => {
                    if (badge.closest('.lstm-param-grid')) {
                        badge.textContent = this.value + '维向量';
                    }
                });
            });
        }
        
        // 更新LSTM层数的badge显示
        const numLayersSelect = document.getElementById('numLayersSelect');
        if (numLayersSelect) {
            const badges = document.querySelectorAll('.dimension-indicator .badge');
            badges.forEach(badge => {
                if (badge.closest('.lstm-param-grid') && badge.previousElementSibling.querySelector('small').textContent.includes('网络深度')) {
                    badge.textContent = numLayersSelect.value + '层LSTM';
                }
            });
            
            // 添加变更事件
            numLayersSelect.addEventListener('change', function() {
                const badges = document.querySelectorAll('.dimension-indicator .badge');
                badges.forEach(badge => {
                    if (badge.closest('.lstm-param-grid') && badge.previousElementSibling.querySelector('small').textContent.includes('网络深度')) {
                        badge.textContent = this.value + '层LSTM';
                    }
                });
            });
        }
    } else if (modelType === 'cnn') {
        // 显示CNN选项并添加动画
        cnnOptions.forEach(el => {
            el.style.display = 'block';
            // 使用setTimeout确保CSS过渡生效
            setTimeout(() => {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }, 50);
        });
        
        // 隐藏LSTM选项
        lstmOptions.forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(10px)';
            // 使用setTimeout确保CSS过渡完成后再隐藏
            setTimeout(() => {
                el.style.display = 'none';
            }, 300);
        });
        
        // 更新卷积核数量的badge显示
        const numFiltersSelect = document.getElementById('numFiltersSelect');
        if (numFiltersSelect) {
            const badges = document.querySelectorAll('.dimension-indicator .badge');
            badges.forEach(badge => {
                if (badge.closest('.cnn-only')) {
                    badge.textContent = numFiltersSelect.value + '个卷积核';
                }
            });
            
            // 添加变更事件
            numFiltersSelect.addEventListener('change', function() {
                const badges = document.querySelectorAll('.dimension-indicator .badge');
                badges.forEach(badge => {
                    if (badge.closest('.cnn-only')) {
                        badge.textContent = this.value + '个卷积核';
                    }
                });
            });
        }
        
        // 初始化卷积核尺寸选择
        initKernelSizeSelectors();
    } else {
        lstmOptions.forEach(el => el.style.display = 'none');
        cnnOptions.forEach(el => el.style.display = 'none');
    }
}

/**
 * 收集训练参数
 */
function collectTrainingParams() {
    const params = {
        model_type: document.getElementById('modelTypeSelect').value,
        batch_size: parseInt(document.getElementById('batchSizeSelect').value),
        epochs: parseInt(document.getElementById('epochsInput').value),
        dropout: parseFloat(document.getElementById('dropoutInput').value),
        optimizer: document.getElementById('optimizerSelect').value,
        weight_decay: document.getElementById('weightDecaySelect') ? 
                      parseFloat(document.getElementById('weightDecaySelect').value) :
                      parseFloat(document.getElementById('weightDecayInput').value)
    };
    
    // 根据模型类型添加特定参数
    if (params.model_type.includes('lstm') || params.model_type.includes('bilstm')) {
        params.hidden_dim = parseInt(document.getElementById('hiddenDimSelect').value);
        params.num_layers = parseInt(document.getElementById('numLayersSelect').value);
    } else if (params.model_type === 'cnn') {
        params.num_filters = parseInt(document.getElementById('numFiltersSelect').value);
        
        // 添加CNN模型中的卷积核大小
        params.kernel_sizes = [];
        for (let i = 2; i <= 5; i++) {
            const kernelCheckbox = document.getElementById(`kernelSize${i}`);
            if (kernelCheckbox && kernelCheckbox.checked) {
                params.kernel_sizes.push(i);
            }
        }
        
        // 如果没有选择任何卷积核大小，默认使用全部
        if (params.kernel_sizes.length === 0) {
            params.kernel_sizes = [2, 3, 4, 5];
        }
    }

    // 如果启用了早停
    const earlyStopping = document.getElementById('earlyStopping');
    if (earlyStopping && earlyStopping.checked) {
        params.early_stopping = true;
        params.patience = parseInt(document.getElementById('patienceInput').value);
    }
    
    return params;
}

/**
 * 启动训练过程
 */
async function startTraining() {
    console.log("startTraining函数被调用");
    
    // 重置训练数据
    trainingData = {
        epochs: [],
        trainLoss: [],
        valLoss: [],
        trainAcc: [],
        valAcc: []
    };

<<<<<<< HEAD
    // 使用 collectTrainingParams 函数收集参数，而不是直接访问 DOM
    const params = collectTrainingParams();
=======
    // 收集表单参数
    const params = collectTrainingParams();
    console.log("训练参数:", params);
>>>>>>> front-1

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
        document.getElementById('trainingStatus').className = 'training-status-badge bg-secondary';
        document.getElementById('currentEpoch').textContent = `0/${params.epochs}`;
        document.getElementById('remainingTime').textContent = '计算中...';
        document.getElementById('progressBar').style.width = '0%';
        document.getElementById('progressLabel').textContent = '0%';
        document.getElementById('progressPercentage').textContent = '0%';

        // 清空训练日志
        const logContent = document.querySelector('.log-content');
        if (logContent) logContent.innerHTML = '';

        // 延迟200毫秒，确保训练进度区域已经显示出来
        setTimeout(() => {
            // 检查图表是否已初始化，如果没有则初始化
            if (!lossChart || !accuracyChart) {
                console.log("图表未初始化，重新初始化");
                initCharts();
            } else {
                // 清空图表数据
                try {
                    lossChart.setOption({
                        xAxis: {data: []},
                        series: [
                            {name: '训练损失', data: []},
                            {name: '验证损失', data: []}
                        ]
                    });
                    accuracyChart.setOption({
                        xAxis: {data: []},
                        series: [
                            {name: '训练准确率', data: []},
                            {name: '验证准确率', data: []}
                        ]
                    });
                    console.log("图表数据已清空");
                } catch (e) {
                    console.error("清空图表数据出错，重新初始化图表:", e);
                    initCharts();
                }
            }
        }, 200);

        // 禁用表单
        toggleFormElements(false);

        // 设置训练状态
        trainingInProgress = true;
        trainingStartTime = new Date();
        totalEpochs = params.epochs;
        currentEpoch = 0;

        // 添加训练日志
        addTrainingLog('info', '准备启动训练任务...');
        
        try {
            // 发送训练请求 - 增强错误处理
            const response = await fetch('/api/train/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'  // 确保服务器返回JSON而不是HTML
                },
                body: JSON.stringify(params)
            });

            // 检查响应类型，如果不是JSON，则进行特殊处理
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || '启动训练失败');
                }

                // 成功启动训练
                const trainingId = data.training_id || "default";

                document.getElementById('trainingStatus').textContent = '训练中';
                document.getElementById('trainingStatus').className = 'training-status-badge bg-primary';
                addTrainingLog('success', `训练任务已启动 (PID: ${data.pid || 'N/A'})`);

                // 设置轮询获取训练进度
                trainingInterval = setInterval(() => {
                    // 使用AJAX检查训练状态和进度
                    checkTrainingProgress();
                }, 3000);
            } else {
                // 如果响应不是JSON，先尝试获取文本
                const textResponse = await response.text();
                console.error('服务器响应非JSON数据:', textResponse);

                // 检查是否是HTML错误页面
                if (textResponse.includes('<!DOCTYPE html>') || textResponse.includes('<html>')) {
                    throw new Error('服务器返回了HTML错误页面，可能是服务器内部错误或API端点不存在');
                } else {
                    throw new Error(`服务器响应无效: ${textResponse.substring(0, 100)}...`);
                }
            }
        } catch (fetchError) {
            console.error('API请求失败:', fetchError);
            throw new Error('API请求失败: ' + fetchError.message);
        }

    } catch (error) {
        console.error('启动训练出错:', error);
        document.getElementById('errorMessage').textContent = '启动训练失败: ' + error.message;
        document.getElementById('errorMessage').style.display = 'block';

        // 恢复训练占位符
        document.getElementById('trainingPlaceholder').classList.remove('d-none');
        document.getElementById('trainingProgress').classList.add('d-none');
        document.getElementById('trainingControls').classList.remove('d-none');

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
        // 添加请求时间戳防止缓存
        const timestamp = new Date().getTime();
        const response = await fetch(`/api/train/progress?t=${timestamp}`, {
            headers: {
                'Accept': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            cache: 'no-store'  // 强制不使用缓存
        });

        // 检查响应类型
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            console.error('进度API返回非JSON数据，Content-Type:', contentType);
            const textResponse = await response.text();
            console.error('响应文本:', textResponse);

            // 尝试手动将响应解析为JSON
            try {
                // 检查是否是JSON字符串但Content-Type设置错误
                const manualParsed = JSON.parse(textResponse);
                console.log('成功手动解析为JSON:', manualParsed);
                updateTrainingProgress(manualParsed);
                return;
            } catch (parseError) {
                console.error('手动解析JSON失败:', parseError);
                // 显示错误但继续轮询
                addTrainingLog('error', '获取进度时返回非JSON数据，将继续尝试');

                // 降级处理：手动构建进度数据以保持界面更新
                const fallbackData = {
                    status: 'running',
                    message: '正在等待有效的进度数据...'
                };
                updateTrainingProgress(fallbackData);
                return;
            }
        }

        let data;
        try {
            data = await response.json();
        } catch (jsonError) {
            console.error('解析JSON响应失败:', jsonError);
            addTrainingLog('error', '解析进度数据失败');
            return;
        }

        if (response.ok && data) {
            console.log("进度数据:", data);
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
                console.error("错误详情:", data);
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
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            
            // 更新进度百分比显示
            const progressLabel = document.getElementById('progressLabel');
            if (progressLabel) {
                progressLabel.textContent = `${progress}%`;
            }
        }
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
            try {
                console.log("更新损失图表，数据长度:", data.history.train_loss.length);

                // 使用简单明确的方式更新
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

                // 确保图表可见并调整大小
                setTimeout(() => {
                    try {
                        lossChart.resize();
                    } catch (e) {
                    }
                }, 100);
            } catch (e) {
                console.error("更新损失图表出错:", e);
                // 尝试重新初始化图表
                setTimeout(initLossChart, 200);
            }
        }

        if (accuracyChart && data.history.val_acc) {
            try {
                console.log("更新准确率图表，数据长度:", data.history.val_acc.length);

                // 使用简单明确的方式更新
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
                            data: data.history.val_acc.map(v => typeof v === 'number' ? v * 100 : 0)
                        }
                    ]
                });

                // 确保图表可见并调整大小
                setTimeout(() => {
                    try {
                        accuracyChart.resize();
                    } catch (e) {
                    }
                }, 100);
            } catch (e) {
                console.error("更新准确率图表出错:", e);
                // 尝试重新初始化图表
                setTimeout(initAccuracyChart, 200);
            }
        }
    }

    // 更新状态
    if (data.status) {
        const statusEl = document.getElementById('trainingStatus');
        if (statusEl) {
            statusEl.textContent = getStatusText(data.status);
            statusEl.className = `training-status-badge ${getStatusClass(data.status)}`;
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
        'initializing': 'bg-secondary',
        'preparing': 'bg-info',
        'running': 'bg-primary',
        'paused': 'bg-warning',
        'stopping': 'bg-warning',
        'stopped': 'bg-secondary',
        'completed': 'bg-success',
        'failed': 'bg-danger'
    };
    return classMap[status] || 'bg-secondary';
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
        const modelName = prompt('请输入模型名称:', `model_${new Date().toISOString().slice(0, 10)}`);

        if (!modelName) return;

        const response = await fetch('/api/train/save_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({name: modelName})
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

/**
 * 初始化模型类型选择
 */
function initModelTypeSelection() {
    // 获取所有模型选择单选按钮和实际的select元素
    const modelRadios = document.querySelectorAll('input[name="modelType"]');
    const modelSelect = document.getElementById('modelTypeSelect');
    
    if (!modelRadios.length || !modelSelect) return;
    
    // 初始化单选按钮状态
    const currentValue = modelSelect.value;
    modelRadios.forEach(radio => {
        if (radio.value === currentValue) {
            radio.checked = true;
        }
        
        // 添加变更事件
        radio.addEventListener('change', function() {
            if (this.checked) {
                console.log(`模型类型更改为: ${this.value}`);
                // 更新隐藏的select值
                modelSelect.value = this.value;
                // 触发select的change事件，以触发现有的toggleModelSpecificOptions逻辑
                modelSelect.dispatchEvent(new Event('change'));
            }
        });
    });
}

/**
 * 初始化权重衰减范围滑块
 */
function initWeightDecayRange() {
    const rangeInput = document.getElementById('weightDecayRange');
    const selectInput = document.getElementById('weightDecaySelect');
    
    if (!rangeInput || !selectInput) return;
    
    // 设置初始值
    const initialValue = Math.log10(parseFloat(selectInput.value));
    rangeInput.value = initialValue;
    
    // 范围滑块变更事件
    rangeInput.addEventListener('input', function() {
        const value = Math.pow(10, parseInt(this.value));
        // 找到最接近的选项
        let closestIndex = 0;
        let minDiff = Number.MAX_VALUE;
        
        for (let i = 0; i < selectInput.options.length; i++) {
            const diff = Math.abs(parseFloat(selectInput.options[i].value) - value);
            if (diff < minDiff) {
                minDiff = diff;
                closestIndex = i;
            }
        }
        
        selectInput.selectedIndex = closestIndex;
    });
    
    // 选择框变更事件
    selectInput.addEventListener('change', function() {
        const value = Math.log10(parseFloat(this.value));
        rangeInput.value = value;
    });
}

/**
 * 初始化Bootstrap工具提示
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl, {
            trigger: 'hover focus'
        });
    });
}

/**
 * 初始化Dropout滑块
 */
function initDropoutSlider() {
    const dropoutRange = document.getElementById('dropoutRange');
    const dropoutDisplay = document.getElementById('dropoutDisplay');
    const dropoutInput = document.getElementById('dropoutInput');
    
    if (dropoutRange && dropoutDisplay && dropoutInput) {
        // 更新显示值
        function updateDropoutValue() {
            const value = (parseInt(dropoutRange.value) + 1) / 10;
            dropoutDisplay.textContent = value.toFixed(1);
            dropoutInput.value = value.toFixed(1);
        }
        
        // 初始设置
        updateDropoutValue();
        
        // 监听滑块变化
        dropoutRange.addEventListener('input', updateDropoutValue);
    }
}

/**
 * 初始化卷积核尺寸选择器
 */
function initKernelSizeSelectors() {
    // 获取所有卷积核选项
    const kernelCheckboxes = document.querySelectorAll('.kernel-checkbox');
    
    // 确保至少有一个被选中
    kernelCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            // 检查是否至少有一个被选中
            const anyChecked = Array.from(kernelCheckboxes).some(cb => cb.checked);
            
            // 如果没有任何一个被选中，则强制选中当前这个
            if (!anyChecked) {
                this.checked = true;
                
                // 显示提示
                const tipElement = document.createElement('div');
                tipElement.className = 'alert alert-warning alert-dismissible fade show mt-2';
                tipElement.innerHTML = `
                    <i class="bi bi-exclamation-triangle me-2"></i>
                    至少需要选择一种卷积核尺寸
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                `;
                
                // 将提示添加到卷积核选择器容器中
                const containerEl = document.querySelector('.kernel-size-container');
                if (containerEl) {
                    containerEl.appendChild(tipElement);
                    
                    // 3秒后自动隐藏
                    setTimeout(() => {
                        tipElement.classList.remove('show');
                        setTimeout(() => {
                            containerEl.removeChild(tipElement);
                        }, 150);
                    }, 3000);
                }
            }
            
            // 设置选中的样式
            updateKernelVisual();
        });
    });
    
    // 初始更新可视化
    updateKernelVisual();
}

/**
 * 更新卷积核可视化
 */
function updateKernelVisual() {
    const kernels = document.querySelectorAll('.cnn-kernel');
    const checkboxes = document.querySelectorAll('.kernel-checkbox');
    
    kernels.forEach((kernel, index) => {
        // 获取对应的checkbox
        const checkbox = checkboxes[index];
        if (checkbox) {
            // 根据checkbox的选中状态设置卷积核的样式
            if (checkbox.checked) {
                kernel.style.opacity = '1';
                kernel.style.transform = 'scale(1)';
            } else {
                kernel.style.opacity = '0.4';
                kernel.style.transform = 'scale(0.9)';
            }
        }
    });
}
