/**
 * 图表相关功能
 */

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

            // 词频图表
            const wordFreqBarChart = initWordFreqBarChart(data);
            const wordCloudChart = initWordCloudChart(data);

            // 保存图表实例到window对象，以便全局访问
            window.chartInstances = {
                pieChart,
                barChart,
                scatterChart,
                wordFreqBarChart,
                wordCloudChart
            };

            console.log('所有图表初始化完成，图表实例:', window.chartInstances);

            // 强制重绘所有图表
            setTimeout(resizeAllCharts, 500);
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
        'wordFreqBarChart',
        'wordCloudChart'
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

    const wordFreqCharts = document.getElementById('wordFreqCharts');
    console.log(`词频图表容器: ${wordFreqCharts ? '存在' : '不存在'}`);
    if (wordFreqCharts) {
        console.log(`词频图表样式: display=${getComputedStyle(wordFreqCharts).display}, visibility=${getComputedStyle(wordFreqCharts).visibility}`);
    }
}

/**
 * 准备图表容器，确保它们有正确的尺寸和可见性
 */
function prepareChartContainers() {
    const containers = [
        'sentimentPieChart',
        'sentimentBarChart',
        'sentimentScatterChart',
        'wordFreqBarChart',
        'wordCloudChart'
    ];

    // 临时显示图表视图容器以便初始化
    const chartView = document.getElementById('chartView');
    const wordFreqCharts = document.getElementById('wordFreqCharts');
    const chartViewOrigDisplay = chartView ? chartView.style.display : 'none';
    const wordFreqChartsOrigDisplay = wordFreqCharts ? wordFreqCharts.style.display : 'none';

    // 临时设置为block以便初始化
    if (chartView) chartView.style.display = 'block';
    if (wordFreqCharts) wordFreqCharts.style.display = 'block';

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

    // 设置延时恢复原始显示状态
    setTimeout(() => {
        if (chartView) chartView.style.display = chartViewOrigDisplay;
        if (wordFreqCharts) wordFreqCharts.style.display = wordFreqChartsOrigDisplay;
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
    const wordFreqTags = document.getElementById('wordFreqTags');
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
 * 初始化词频柱状图
 */
function initWordFreqBarChart(data) {
    console.log('初始化词频柱状图...');

    const container = document.getElementById('wordFreqBarChart');
    if (!container) {
        console.error('找不到词频柱状图容器');
        return null;
    }

    // 先确保容器有明确的尺寸
    container.style.width = '100%';
    container.style.height = '350px';
    container.style.minHeight = '300px';
    container.style.visibility = 'visible';

    // 确保父容器也是可见的
    const parent = container.closest('.chart-wrapper');
    if (parent) {
        parent.style.display = 'block';
        parent.style.visibility = 'visible';
    }

    // 销毁可能存在的旧实例
    try {
        const existingChart = echarts.getInstanceByDom(container);
        if (existingChart) {
            existingChart.dispose();
            console.log('已销毁旧的词频柱状图实例');
        }
    } catch (e) {
        console.warn('尝试销毁旧图表实例时出错:', e);
    }

    // 创建图表前再次检查容器
    console.log(`词频柱状图容器尺寸: ${container.offsetWidth}x${container.offsetHeight}`);

    // 确保数据有效
    if (!data.wordFreq || !Array.isArray(data.wordFreq) || data.wordFreq.length === 0) {
        console.warn('词频数据为空');
        container.innerHTML = '<div class="text-center text-muted p-5">无词频数据可显示</div>';
        return null;
    }

    const wordFreqData = data.wordFreq.slice(0, 20); // 只显示前20个词

    try {
        // 使用echarts.init强制指定大小
        const wordFreqBar = echarts.init(container, null, {
            width: container.offsetWidth || 500,
            height: container.offsetHeight || 350
        });

        // 设置选项
        const wordFreqBarOption = {
            title: {
                text: '高频词汇统计',
                left: 'center',
                top: 20
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
                bottom: '15%',
                top: 60,
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: wordFreqData.map(item => item.word),
                axisLabel: {
                    interval: 0,
                    rotate: 45
                }
            },
            yAxis: {
                type: 'value'
            },
            series: [{
                data: wordFreqData.map(item => ({
                    value: item.count,
                    itemStyle: {
                        color: `rgb(${Math.random() * 150 + 50}, ${Math.random() * 150 + 50}, ${Math.random() * 150 + 50})`
                    }
                })),
                type: 'bar',
                barWidth: '40%'
            }]
        };

        // 应用选项
        wordFreqBar.setOption(wordFreqBarOption);
        console.log('词频柱状图初始化成功');

        // 立即调整大小
        wordFreqBar.resize();

        return wordFreqBar;
    } catch (e) {
        console.error('初始化词频柱状图出错:', e);
        return null;
    }
}

/**
 * 初始化词云图
 */
function initWordCloudChart(data) {
    // 类似上面的实现，增强错误处理和日志
    // ...existing code with similar improvements...

    console.log('初始化词云图...');

    const container = document.getElementById('wordCloudChart');
    if (!container) {
        console.error('找不到词云图容器');
        return null;
    }

    // 先确保容器有明确的尺寸
    container.style.width = '100%';
    container.style.height = '350px';
    container.style.minHeight = '300px';
    container.style.visibility = 'visible';

    // 确保父容器也是可见的
    const parent = container.closest('.chart-wrapper');
    if (parent) {
        parent.style.display = 'block';
        parent.style.visibility = 'visible';
    }

    // 销毁可能存在的旧实例
    try {
        const existingChart = echarts.getInstanceByDom(container);
        if (existingChart) {
            existingChart.dispose();
            console.log('已销毁旧的词云图实例');
        }
    } catch (e) {
        console.warn('尝试销毁旧词云图实例时出错:', e);
    }

    // 检查wordCloud插件是否可用
    if (!echarts.getMap) {
        console.error('echarts-wordcloud插件未找到，词云图无法初始化');
        container.innerHTML = '<div class="text-center text-danger p-3">无法加载词云图组件</div>';
        return null;
    }

    // 创建图表前再次检查容器
    console.log(`词云图容器尺寸: ${container.offsetWidth}x${container.offsetHeight}`);

    if (!data.wordFreq || data.wordFreq.length === 0) {
        console.warn('词频数据为空');
        container.innerHTML = '<div class="text-center text-muted p-5">无词频数据可显示</div>';
        return null;
    }

    try {
        // 使用echarts.init强制指定大小
        const wordCloud = echarts.init(container, null, {
            width: container.offsetWidth || 500,
            height: container.offsetHeight || 350
        });

        const wordCloudOption = {
            title: {
                text: '词云展示',
                left: 'center',
                top: 20
            },
            tooltip: {
                show: true
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
                sizeRange: [15, 80],
                rotationRange: [-45, 45],
                rotationStep: 30,
                gridSize: 8,
                drawOutOfBound: false,
                textStyle: {
                    fontFamily: 'sans-serif',
                    fontWeight: 'bold',
                    color: function () {
                        return 'rgb(' +
                            Math.round(Math.random() * 150 + 50) + ',' +
                            Math.round(Math.random() * 150 + 50) + ',' +
                            Math.round(Math.random() * 150 + 50) + ')'
                    }
                },
                emphasis: {
                    focus: 'self',
                    textStyle: {
                        shadowBlur: 10,
                        shadowColor: '#333'
                    }
                },
                data: data.wordFreq.map(item => ({
                    name: item.word,
                    value: item.count * 100
                }))
            }]
        };

        // 应用选项
        wordCloud.setOption(wordCloudOption);
        console.log('词云图初始化成功');

        // 立即调整大小
        wordCloud.resize();

        return wordCloud;
    } catch (e) {
        console.error('初始化词云图出错:', e);
        console.error(e.stack);
        return null;
    }
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
 * 应用图表动画
 */
function applyChartAnimations() {
    const charts = [
        echarts.getInstanceByDom(document.getElementById('overallPieChart')),
        echarts.getInstanceByDom(document.getElementById('sentimentPieChart')),
        echarts.getInstanceByDom(document.getElementById('sentimentBarChart')),
        echarts.getInstanceByDom(document.getElementById('sentimentScatterChart')),
        echarts.getInstanceByDom(document.getElementById('wordFreqBarChart')),
        echarts.getInstanceByDom(document.getElementById('wordCloudChart')),
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
        echarts.getInstanceByDom(document.getElementById('wordFreqBarChart')),
        echarts.getInstanceByDom(document.getElementById('wordCloudChart')),
        echarts.getInstanceByDom(document.getElementById('confusionMatrixChart'))
    ].filter(chart => chart); // 过滤掉null或undefined

    window.addEventListener('resize', function () {
        charts.forEach(chart => chart.resize());
    });

    charts.forEach(chart => chart.resize());
}
