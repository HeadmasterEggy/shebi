/**
 * 初始化所有图表
 * @param {Object} data - 分析结果数据
 */
function initCharts(data) {
    console.log('开始初始化图表...');
    window.allSentences = data.sentences; // 确保全局保存句子数据

    // 调试日志：检查图表容器
    checkChartContainers();

    // 确保图表容器在初始化前有正确的尺寸
    prepareChartContainers();

    // 延迟执行图表初始化，确保DOM已完全加载和渲染
    setTimeout(() => {
        try {
            // 句子分析图表
            const pieChart = initSentimentPieChart(data);
            const barChart = initSentimentBarChart(data);
            const scatterChart = initSentimentScatterChart(data);
            
            // 添加词频图表初始化 - 确保按顺序进行初始化
            const wordFreqBarChart = initWordFreqBarChart(data);
            // 词云图延迟初始化，确保容器准备好
            let wordCloudChart = null;
            setTimeout(() => {
                wordCloudChart = initWordCloudChart(data);
                // 确保图表实例添加到全局对象
                if (wordCloudChart && window.chartInstances) {
                    window.chartInstances.wordCloudChart = wordCloudChart;
                }
            }, 500);

            // 保存图表实例到window对象，以便全局访问
            window.chartInstances = {
                pieChart,
                barChart,
                scatterChart,
                wordFreqBarChart,
                wordCloudChart: null // 先设置为null，后面更新
            };

            console.log('图表初始化队列已启动');

            // 强制重绘所有图表
            setTimeout(resizeAllCharts, 1000);
        } catch (error) {
            console.error('图表初始化过程中出错:', error);
        }
    }, 300);
}

/**
 * 检查图表容器是否存在
 */
function checkChartContainers() {
    const containers = [
        'sentimentPieChart',
        'sentimentBarChart',
        'sentimentScatterChart',
        'wordFreqBarChart',  // 添加词频图表容器
        'wordCloudChart'     // 添加词云图容器
    ];

    containers.forEach(id => {
        const container = document.getElementById(id);
        console.log(`图表容器 ${id}: ${container ? '存在' : '不存在'}`);
        if (container) {
            console.log(`容器尺寸: ${container.offsetWidth}x${container.offsetHeight}`);
            console.log(`容器样式: display=${getComputedStyle(container).display}, visibility=${getComputedStyle(container).visibility}`);
        }
    });

    // 检查父容器
    const chartView = document.getElementById('chartView');
    console.log(`图表视图容器: ${chartView ? '存在' : '不存在'}`);
    if (chartView) {
        console.log(`图表视图样式: display=${getComputedStyle(chartView).display}, visibility=${getComputedStyle(chartView).visibility}`);
    }


}

/**
 * 准备图表容器，确保它们有正确的尺寸和可见性
 */
function prepareChartContainers() {
    const containers = [
        'sentimentPieChart',
        'sentimentBarChart',
        'sentimentScatterChart'
    ];

    // 临时显示图表视图容器以便初始化
    const chartView = document.getElementById('chartView');
    const chartViewOrigDisplay = chartView ? chartView.style.display : 'none';
    
    // 临时设置为block以便初始化
    if (chartView) chartView.style.display = 'block';
    
    // 确保图表容器可见
    if (wordFreqCharts) {
        wordFreqCharts.style.display = 'grid';
        wordFreqCharts.style.gridTemplateColumns = 'repeat(auto-fit, minmax(300px, 1fr))';
        wordFreqCharts.style.gap = '20px';
    }

    // 确保每个容器有明确的尺寸
    containers.forEach(id => {
        const container = document.getElementById(id);
        if (container) {
            container.style.width = '100%';
            container.style.height = '400px';
            container.style.minHeight = '300px';
            container.style.visibility = 'visible';
            container.style.opacity = '1';
        }
    });

    // 打印日志确认容器已准备好
    console.log('图表容器已准备好，将在初始化后恢复原始显示状态');

    // 设置延时恢复原始显示状态 - 只针对句子分析图表
    setTimeout(() => {
        if (chartView) chartView.style.display = chartViewOrigDisplay;
        // 词频图表保持显示状态
        console.log('图表容器已恢复原始显示状态');
    }, 1000);
}

/**
 * 强制重绘所有图表
 */
function resizeAllCharts() {
    if (!window.chartInstances) return;

    Object.values(window.chartInstances).forEach(chart => {
        if (chart && typeof chart.resize === 'function') {
            try {
                chart.resize();
                console.log('图表已重绘:', chart);
            } catch (e) {
                console.warn('重绘图表时出错:', e);
            }
        }
    });
}

/**
 * 初始化词频标签
 */
function initWordFreqTags(data) {
    const wordFreqTags = document.getElementById('wordFreqTagsContent');
    if (!wordFreqTags) return;

    // 简化检查
    if (!data || !data.wordFreq || !data.wordFreq.length) {
        wordFreqTags.innerHTML = '<p class="text-muted text-center py-4">没有词频数据</p>';
        return;
    }

    try {
        const maxCount = Math.max(...data.wordFreq.map(item => item.count));
        const minCount = Math.min(...data.wordFreq.map(item => item.count));
        let tagsHtml = '';

        data.wordFreq.forEach((item, index) => {
            // 简化颜色计算
            const intensity = (item.count - minCount) / (maxCount - minCount || 1);
            const color = `hsl(210, ${80 - intensity * 40}%, ${85 - intensity * 30}%)`;
            const textColor = intensity > 0.6 ? '#fff' : '#333';

            tagsHtml += `<span class="word-freq-item" style="--delay: ${index}; background: ${color}; color: ${textColor}">${item.word} (${item.count})</span>`;
        });

        wordFreqTags.innerHTML = tagsHtml;
    } catch (error) {
        wordFreqTags.innerHTML = '<p class="text-danger text-center py-4">词频标签生成出错</p>';
    }
}

/**
 * 初始化整体饼图
 */
function initOverallPieChart(data) {
    const overallPieChart = echarts.init(document.getElementById('overallPieChart'));
    const overallPieOption = {
        title: {
            text: '整体情感分布',
            left: 'center',
            top: 20
        },
        tooltip: {
            trigger: 'item',
            formatter: '{b}: {c}% ({d}%)'
        },
        legend: {
            orient: 'vertical',
            left: '5%',
            top: 'middle'
        },
        series: [{
            type: 'pie',
            radius: ['40%', '70%'],
            center: ['60%', '50%'],
            avoidLabelOverlap: true,
            itemStyle: {
                borderRadius: 10,
                borderColor: '#fff',
                borderWidth: 2
            },
            label: {
                show: true,
                formatter: '{b}\n{c}%'
            },
            data: [
                {
                    value: Number(data.overall.probabilities.positive.toFixed(2)),
                    name: '积极',
                    itemStyle: {color: '#28a745'}
                },
                {
                    value: Number(data.overall.probabilities.negative.toFixed(2)),
                    name: '消极',
                    itemStyle: {color: '#dc3545'}
                }
            ]
        }]
    };
    overallPieChart.setOption(overallPieOption);
}

/**
 * 初始化情感饼图
 */
function initSentimentPieChart(data) {
    const pieChart = echarts.init(document.getElementById('sentimentPieChart'));
    const pieOption = {
        title: {
            text: '整体情感分布',
            left: 'center',
            top: 20
        },
        tooltip: {
            trigger: 'item',
            formatter: '{b}: {c}% ({d}%)'
        },
        legend: {
            orient: 'vertical',
            left: '5%',
            top: 'middle'
        },
        grid: {
            containLabel: true
        },
        series: [{
            type: 'pie',
            radius: ['40%', '70%'],
            center: ['60%', '50%'],
            avoidLabelOverlap: true,
            itemStyle: {
                borderRadius: 10,
                borderColor: '#fff',
                borderWidth: 2
            },
            label: {
                show: true,
                formatter: '{b}\n{c}%'
            },
            data: [
                {
                    value: Number(data.overall.probabilities.positive.toFixed(2)),
                    name: '积极',
                    itemStyle: {color: '#28a745'}
                },
                {
                    value: Number(data.overall.probabilities.negative.toFixed(2)),
                    name: '消极',
                    itemStyle: {color: '#dc3545'}
                }
            ]
        }]
    };
    pieChart.setOption(pieOption);
}

/**
 * 初始化情感柱状图
 */
function initSentimentBarChart(data) {
    const barChart = echarts.init(document.getElementById('sentimentBarChart'));
    const sentenceData = data.sentences.map((s, index) => ({
        positive: Number(s.probabilities.positive.toFixed(2)),
        negative: Number(s.probabilities.negative.toFixed(2)),
        index: `句子${index + 1}`
    }));

    const barOption = {
        title: {
            text: '句子情感分布',
            left: 'center',
            top: 20
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            },
            formatter: function (params) {
                return params[0].name + '<br/>' +
                    params[0].seriesName + ': ' + params[0].value.toFixed(2) + '%<br/>' +
                    params[1].seriesName + ': ' + params[1].value.toFixed(2) + '%';
            }
        },
        legend: {
            data: ['积极', '消极'],
            top: 50
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true,
            top: 100
        },
        xAxis: {
            type: 'value',
            boundaryGap: [0, 0.01],
            axisLabel: {
                formatter: '{value}%'
            }
        },
        yAxis: {
            type: 'category',
            data: sentenceData.map(d => d.index)
        },
        series: [
            {
                name: '积极',
                type: 'bar',
                data: sentenceData.map(d => d.positive),
                itemStyle: {color: '#28a745'}
            },
            {
                name: '消极',
                type: 'bar',
                data: sentenceData.map(d => d.negative),
                itemStyle: {color: '#dc3545'}
            }
        ]
    };
    barChart.setOption(barOption);
}

/**
 * 初始化情感散点图
 */
function initSentimentScatterChart(data) {
    const scatterChart = echarts.init(document.getElementById('sentimentScatterChart'));
    const scatterData = data.sentences.map((s, index) => ([
        Number(s.probabilities.positive.toFixed(2)),
        Number(s.confidence.toFixed(2)),
        index + 1,
        s.sentiment === '积极' ? 0 : 1
    ]));

    const scatterOption = {
        title: {
            text: '情感分析散点图',
            left: 'center',
            top: 20
        },
        tooltip: {
            trigger: 'item',
            formatter: function (params) {
                const sentimentType = params.data[3] === 0 ? '积极' : '消极';
                return `句子${params.data[2]}<br/>` +
                    `积极概率: ${params.data[0]}%<br/>` +
                    `置信度: ${params.data[1]}%<br/>` +
                    `情感类型: ${sentimentType}`;
            }
        },
        legend: {
            data: ['积极', '消极'],
            top: 50
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true,
            top: 100
        },
        xAxis: {
            type: 'value',
            name: '积极情绪概率',
            nameLocation: 'middle',
            nameGap: 30,
            min: 0,
            max: 100,
            axisLabel: {
                formatter: '{value}%'
            }
        },
        yAxis: {
            type: 'value',
            name: '置信度',
            nameLocation: 'middle',
            nameGap: 30,
            min: 0,
            max: 100,
            axisLabel: {
                formatter: '{value}%'
            }
        },
        series: [
            {
                name: '积极',
                type: 'scatter',
                symbolSize: 12,
                data: scatterData.filter(item => item[3] === 0),
                itemStyle: {
                    color: '#28a745'
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            },
            {
                name: '消极',
                type: 'scatter',
                symbolSize: 12,
                data: scatterData.filter(item => item[3] === 1),
                itemStyle: {
                    color: '#dc3545'
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
    scatterChart.setOption(scatterOption);
}

/**
 * 初始化混淆矩阵
 */
function initConfusionMatrix(data) {
    const confusionMatrixChart = echarts.init(document.getElementById('confusionMatrixChart'));
    const confusionData = [
        [Math.round(data.sentences.filter(s => s.sentiment === '消极').length * data.modelMetrics.accuracy),
            Math.round(data.sentences.filter(s => s.sentiment === '消极').length * (1 - data.modelMetrics.accuracy))],
        [Math.round(data.sentences.filter(s => s.sentiment === '积极').length * (1 - data.modelMetrics.accuracy)),
            Math.round(data.sentences.filter(s => s.sentiment === '积极').length * data.modelMetrics.accuracy)]
    ];

    const confusionOption = {
        tooltip: {
            position: 'top',
            formatter: function (params) {
                const labels = ['真实消极', '真实积极'];
                const predictions = ['预测消极', '预测积极'];
                return `${labels[params.data[0]]}<br>${predictions[params.data[1]]}<br>数量: ${params.data[2]}`;
            }
        },
        grid: {
            left: '3%',
            right: '7%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: ['预测消极', '预测积极'],
            splitArea: {
                show: true
            }
        },
        yAxis: {
            type: 'category',
            data: ['真实消极', '真实积极'],
            splitArea: {
                show: true
            }
        },
        visualMap: {
            min: 0,
            max: Math.max(...confusionData.flat()) || 10,
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: '0%',
            inRange: {
                color: ['#f2f2f2', '#5470c6', '#91cc75']
            }
        },
        series: [{
            name: '混淆矩阵',
            type: 'heatmap',
            data: [
                [0, 0, confusionData[0][0]],
                [0, 1, confusionData[0][1]],
                [1, 0, confusionData[1][0]],
                [1, 1, confusionData[1][1]]
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
        }]
    };
    confusionMatrixChart.setOption(confusionOption);

    return confusionMatrixChart;
}

/**
 * 初始化词频柱状图
 * @param {Object} data - 分析结果数据
 */
function initWordFreqBarChart(data) {
    if (!data || !data.wordFreq) return null;
    
    const container = document.getElementById('wordFreqBarChart');
    if (!container) return null;
    
    try {
        // 清理现有实例
        const existingChart = echarts.getInstanceByDom(container);
        if (existingChart) {
            existingChart.dispose();
        }
        
        // 准备数据
        const wordFreq = data.wordFreq.slice(0, 30); // 取前15个高频词
        const words = wordFreq.map(item => item.word);
        const counts = wordFreq.map(item => item.count);
        
        const chart = echarts.init(container);
        const option = {
            title: {
                text: '词频统计',
                left: 'center'
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '10%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: words,
                axisLabel: {
                    interval: 0,
                    rotate: 30,
                    textStyle: {
                        fontSize: 12
                    }
                }
            },
            yAxis: {
                type: 'value',
                name: '出现次数'
            },
            series: [{
                name: '词频',
                type: 'bar',
                data: counts,
                itemStyle: {
                    color: function(params) {
                        // 使用渐变色
                        const colorList = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272'];
                        return colorList[params.dataIndex % colorList.length];
                    }
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                },
                label: {
                    show: true,
                    position: 'top'
                }
            }]
        };
        
        chart.setOption(option);
        return chart;
    } catch (e) {
        console.error('初始化词频柱状图出错:', e);
        return null;
    }
}

/**
 * 完全重写的词云图初始化函数
 * @param {Object} data - 分析结果数据
 * @param {number} attempt - 尝试次数，默认1
 */
function initWordCloudChart(data, attempt = 1) {
    console.log(`尝试初始化词云图 (第${attempt}次)`);
    
    if (!data || !data.wordFreq) {
        console.error('初始化词云图失败: 缺少词频数据');
        return null;
    }
    
    const container = document.getElementById('wordCloudChart');
    if (!container) {
        console.error('初始化词云图失败: 找不到容器 #wordCloudChart');
        return null;
    }
    
    // 输出容器尺寸，帮助调试
    console.log(`词云图容器尺寸: ${container.offsetWidth}x${container.offsetHeight}`);
    console.log(`容器可见性: display=${getComputedStyle(container).display}, visibility=${getComputedStyle(container).visibility}`);
    
    // 确保容器有适当的尺寸
    if (container.offsetWidth < 100 || container.offsetHeight < 100) {
        console.warn('词云图容器尺寸不足，设置明确尺寸');
        container.style.width = '100%';
        container.style.height = '400px';
        container.style.minHeight = '350px';
        container.style.visibility = 'visible';
        container.style.opacity = '1';
        
        // 立刻刷新容器尺寸
        const parentNode = container.parentNode;
        if (parentNode) {
            parentNode.style.display = 'block';
            parentNode.style.width = '100%';
            parentNode.style.height = '400px';
        }
    }
    
    try {
        // 清理现有实例避免冲突
        if (window.chartInstances && window.chartInstances.wordCloudChart) {
            try {
                window.chartInstances.wordCloudChart.dispose();
                console.log('已清理旧的词云图实例');
            } catch (e) {}
        }
        
        // 检查echarts是否可用
        if (typeof echarts === 'undefined') {
            console.error('echarts 库未加载，无法创建词云图');
            if (attempt < 3) {
                console.log(`将在1秒后重试 (第${attempt+1}次)`);
                setTimeout(() => initWordCloudChart(data, attempt + 1), 1000);
            }
            return null;
        }
        
        // 准备词云数据 - 最简化处理
        let wordCloudData = [];
        try {
            if (data.wordFreq && data.wordFreq.length > 0) {
                wordCloudData = data.wordFreq.slice(0, 50).map(item => ({
                    name: item.word,
                    value: item.count,
                    // 添加自定义属性以增强交互体验
                    itemStyle: {
                        // 默认正常状态下的样式
                        normal: {
                            fontWeight: 'bold',
                            shadowBlur: 3,
                            shadowColor: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                }));
            } else {
                wordCloudData = [{ name: '无词频数据', value: 1 }];
            }
            console.log(`已准备${wordCloudData.length}个词条数据`);
        } catch (e) {
            console.error('准备词云数据出错:', e);
            wordCloudData = [{ name: '数据错误', value: 1 }];
        }

        // 创建词云图表实例
        const chart = echarts.init(container);
        
        // 增强的配置，优化交互功能
        const option = {
            tooltip: {
                show: true,
                trigger: 'item',
                formatter: function(params) {
                    return `<div style="font-weight:bold; font-size:14px;">${params.name}</div>
                            <div>出现次数：<span style="color:#ff9500; font-weight:bold">${params.value}</span></div>`;
                },
                backgroundColor: 'rgba(50,50,50,0.9)',
                borderColor: '#eee',
                borderWidth: 1,
                padding: [8, 12],
                textStyle: {
                    color: '#fff',
                    fontSize: 13
                },
                extraCssText: 'box-shadow: 0 3px 14px rgba(0,0,0,0.3); border-radius: 6px;',
                axisPointer: {
                    type: 'none'  // 禁用轴指示器
                },
                position: function(point, params, dom, rect, size) {
                    // 动态计算tooltip位置，避免被遮挡
                    return [point[0] + 15, point[1] - 20];
                }
            },
            series: [{
                type: 'wordCloud',
                shape: 'circle',
                left: 'center',
                top: 'center',
                width: '90%',
                height: '90%',
                right: null,
                bottom: null,
                sizeRange: [12, 60],  // 字体大小范围
                rotationRange: [-45, 45], // 旋转角度范围
                rotationStep: 5,  // 旋转角度步长
                gridSize: 8,  // 网格大小，值越大，词间距越大
                drawOutOfBound: false, // 防止词超出画布边界
                layoutAnimation: true, // 开启布局动画
                textStyle: {
                    fontFamily: 'sans-serif',
                    fontWeight: 'bold',
                    color: function () {
                        return 'rgb(' + [
                            Math.round(Math.random() * 160) + 60,
                            Math.round(Math.random() * 160) + 60,
                            Math.round(Math.random() * 160) + 60
                        ].join(',') + ')';
                    }
                },
                emphasis: {
                    focus: 'self',  // 聚焦当前项
                    textStyle: {
                        fontSize: 'bolder', // 更明显的字体加粗
                        shadowBlur: 15,    // 增加阴影模糊半径
                        shadowColor: 'rgba(0, 0, 0, 0.5)', // 更明显的阴影
                        color: function(params) {
                            // 悬停时使用更亮的颜色版本
                            const originalColor = params.color || '#5470c6';
                            // 将颜色转成更亮的版本
                            // 如果是rgb格式
                            if (originalColor.startsWith('rgb')) {
                                const rgb = originalColor.match(/\d+/g);
                                if (rgb && rgb.length >= 3) {
                                    // 使颜色更亮但保持在有效范围内
                                    const r = Math.min(255, parseInt(rgb[0]) + 40);
                                    const g = Math.min(255, parseInt(rgb[1]) + 40);
                                    const b = Math.min(255, parseInt(rgb[2]) + 40);
                                    return `rgb(${r},${g},${b})`;
                                }
                            }
                            return originalColor; // 如果无法解析，返回原始颜色
                        }
                    }
                },
                data: wordCloudData
            }]
        };
        
        // 应用配置
        console.log('开始渲染词云图...');
        chart.setOption(option);
        
        // 添加调试信息
        console.log('词云图渲染完成，当前容器尺寸:', 
                   `${container.offsetWidth}x${container.offsetHeight}`);
        
        // 优化鼠标事件监听器
        // 使用全局变量来跟踪鼠标状态，解决鼠标快速移动时的问题
        let isMouseOverCloud = false;
        
        chart.on('mouseover', function(params) {
            isMouseOverCloud = true;
            container.style.cursor = 'pointer';
            console.log('词云图鼠标悬停:', params.name);
            
            // 显示更大的强调效果
            try {
                chart.dispatchAction({
                    type: 'highlight',
                    seriesIndex: 0,
                    dataIndex: params.dataIndex
                });
            } catch(e) {
                console.warn('无法高亮显示词云项:', e);
            }
        });
        
        chart.on('mouseout', function(params) {
            isMouseOverCloud = false;
            container.style.cursor = 'default';
            
            // 移除高亮效果
            try {
                chart.dispatchAction({
                    type: 'downplay',
                    seriesIndex: 0,
                    dataIndex: params.dataIndex
                });
            } catch(e) {
                console.warn('无法移除词云项高亮:', e);
            }
        });
        
        // 添加全局鼠标移动监听器，确保鼠标指针正确变化
        container.addEventListener('mousemove', function(e) {
            if (!isMouseOverCloud) {
                // 检查当前点在图表中的位置，判断是否在词云元素上
                const point = [e.offsetX, e.offsetY];
                const isOverItem = chart.containPixel({seriesIndex: 0}, point);
                container.style.cursor = isOverItem ? 'pointer' : 'default';
            }
        });
        
        // 存储实例供后续使用
        if (window.chartInstances) {
            window.chartInstances.wordCloudChart = chart;
        } else {
            window.chartInstances = {
                ...window.chartInstances,
                wordCloudChart: chart
            };
        }
        
        // 添加专门的强制渲染函数
        window.renderWordCloud = function() {
            try {
                if (chart) {
                    chart.resize();
                    console.log('已重新调整词云图大小');
                }
            } catch (e) {
                console.error('强制渲染词云图出错:', e);
            }
        };
        
        // 延时强制重渲染，确保尺寸正确
        setTimeout(() => {
            if (chart) {
                chart.resize();
                console.log('已延时重新渲染词云图');
            }
        }, 1000);
        
        return chart;
        
    } catch (e) {
        console.error('初始化词云图出错:', e);
        
        // 如果是首次尝试，延迟后重试
        if (attempt < 3) {
            console.log(`将在${attempt * 1000}毫秒后重试 (第${attempt+1}次)`);
            setTimeout(() => initWordCloudChart(data, attempt + 1), attempt * 1000);
        } else {
            console.error('词云图创建失败，已达到最大重试次数');
            
            // 失败后尝试创建备用图表
            try {
                console.log('尝试创建备用饼图显示词频数据');
                createFallbackPieChart(container, data.wordFreq);
            } catch (fallbackError) {
                console.error('创建备用图表也失败了:', fallbackError);
            }
        }
        return null;
    }
}

/**
 * 创建备用饼图以显示词频数据
 */
function createFallbackPieChart(container, wordFreq) {
    if (!container || !wordFreq) return;
    
    try {
        // 取前10个词频数据
        const data = wordFreq.slice(0, 10);
        const chart = echarts.init(container);
        
        const option = {
            title: {
                text: '词频统计(备用图)',
                left: 'center'
            },
            tooltip: {
                trigger: 'item',
                formatter: '{a} <br/>{b} : {c} ({d}%)'
            },
            legend: {
                orient: 'vertical',
                left: 'left',
                data: data.map(item => item.word)
            },
            series: [
                {
                    name: '词频',
                    type: 'pie',
                    radius: '55%',
                    center: ['50%', '60%'],
                    data: data.map(item => ({
                        name: item.word,
                        value: item.count
                    })),
                    emphasis: {
                        itemStyle: {
                            shadowBlur: 10,
                            shadowOffsetX: 0,
                            shadowColor: 'rgba(0, 0, 0, 0.5)'
                        }
                    }
                }
            ]
        };
        
        chart.setOption(option);
        console.log('已创建备用饼图显示词频数据');
        
        // 保存实例
        if (window.chartInstances) {
            window.chartInstances.wordCloudChart = chart;
        }
        
        return chart;
    } catch (e) {
        console.error('创建备用饼图失败:', e);
        return null;
    }
}

/**
 * 应用图表动画
 */
function applyChartAnimations() {
    const charts = [
        echarts.getInstanceByDom(document.getElementById('overallPieChart')),
        echarts.getInstanceByDom(document.getElementById('sentimentPieChart')),
        echarts.getInstanceByDom(document.getElementById('sentimentBarChart')),
        echarts.getInstanceByDom(document.getElementById('sentimentScatterChart')),
        // 移除词频图表
        echarts.getInstanceByDom(document.getElementById('confusionMatrixChart'))
    ].filter(chart => chart); // 过滤掉null或undefined

    const chartOptions = {
        animation: {
            duration: 1000,
            easing: 'cubicInOut'
        }
    };

    charts.forEach(chart => {
        const option = chart.getOption();
        chart.setOption({
            ...option,
            ...chartOptions
        });
    });
}

/**
 * 设置图表大小调整
 */
function setupChartResizing() {
    const charts = [
        echarts.getInstanceByDom(document.getElementById('overallPieChart')),
        echarts.getInstanceByDom(document.getElementById('sentimentPieChart')),
        echarts.getInstanceByDom(document.getElementById('sentimentBarChart')),
        echarts.getInstanceByDom(document.getElementById('sentimentScatterChart')),
        // 移除词频图表
        echarts.getInstanceByDom(document.getElementById('confusionMatrixChart'))
    ].filter(chart => chart); // 过滤掉null或undefined

    window.addEventListener('resize', function () {
        charts.forEach(chart => chart.resize());
    });

    charts.forEach(chart => chart.resize());
}
