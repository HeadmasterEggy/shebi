/**
 * 图表相关功能
 */

/**
 * 初始化所有图表
 * @param {Object} data - 分析结果数据
 */
function initCharts(data) {
    initWordFreqTags(data);

    setTimeout(() => {
        initOverallPieChart(data);
        initSentimentPieChart(data);
        initSentimentBarChart(data);
        initSentimentScatterChart(data);
        initWordFreqBarChart(data);
        initWordCloudChart(data);

        // 应用共同的图表动画选项
        applyChartAnimations();

        // 添加窗口大小改变事件监听器
        setupChartResizing();
    }, 0);
}

/**
 * 初始化词频标签
 */
function initWordFreqTags(data) {
    const wordFreqTags = document.getElementById('wordFreqTags');
    let tagsHtml = '';

    if (data.wordFreq && data.wordFreq.length > 0) {
        const maxCount = Math.max(...data.wordFreq.map(item => item.count));
        const minCount = Math.min(...data.wordFreq.map(item => item.count));

        data.wordFreq.forEach((item, index) => {
            const colorIntensity = Math.max(0, Math.min(1, (item.count - minCount) / (maxCount - minCount || 1)));
            const r = Math.round(220 - colorIntensity * 150);
            const g = Math.round(240 - colorIntensity * 150);
            const b = Math.round(255 - colorIntensity * 150);
            const textColor = colorIntensity > 0.6 ? '#ffffff' : '#333333';
            const borderColor = `rgba(${r - 30}, ${g - 30}, ${b - 30}, 0.3)`;

            tagsHtml += `<span class="word-freq-item" style="--delay: ${index}; background: linear-gradient(135deg, rgb(${r}, ${g}, ${b}), rgb(${r + 10}, ${g + 10}, ${b + 10})); color: ${textColor}; border-color: ${borderColor};">${item.word} (${item.count})</span>`;
        });
        wordFreqTags.innerHTML = tagsHtml;
    } else {
        wordFreqTags.innerHTML = '<p class="text-muted">没有词频数据</p>';
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
    const wordFreqBar = echarts.init(document.getElementById('wordFreqBarChart'));
    const wordFreqData = data.wordFreq.slice(0, 20);
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
            },
            formatter: function (params) {
                return params[0].name + ': ' + params[0].value;
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
    wordFreqBar.setOption(wordFreqBarOption);
}

/**
 * 初始化词云图
 */
function initWordCloudChart(data) {
    const wordCloud = echarts.init(document.getElementById('wordCloudChart'));
    const wordCloudOption = {
        title: {
            text: '词云展示',
            left: 'center',
            top: 20
        },
        tooltip: {
            show: true,
            formatter: function (params) {
                return params.name + ': ' + (params.value / 100).toFixed(0);
            }
        },
        series: [{
            type: 'wordCloud',
            shape: 'circle',
            size: ['100%', '100%'],
            left: 'center',
            top: 'center',
            width: '100%',
            height: '100%',
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
    wordCloud.setOption(wordCloudOption);
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
