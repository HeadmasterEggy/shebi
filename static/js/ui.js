/**
 * 用户界面相关功能
 */

/**
 * 切换标签页
 * @param {string} view - 视图类型 ('list' 或 'chart')
 */
function switchTab(view) {
    console.log(`切换句子分析视图: ${view}`);
    const listView = document.getElementById('listView');
    const chartView = document.getElementById('chartView');
    const buttons = document.querySelectorAll('#sentenceResults .tab-button');

    if (!listView || !chartView) {
        console.error('找不到视图容器元素:', {listView, chartView});
        return;
    }

    // 防止事件连续触发，添加一个小延时
    if (window.switchTabInProgress) {
        console.log('视图切换正在进行中，请稍候...');
        return;
    }
    
    window.switchTabInProgress = true;
    setTimeout(() => {
        window.switchTabInProgress = false;
    }, 300);

    if (view === 'list') {
        listView.style.display = 'block';
        chartView.style.display = 'none';
        console.log('已切换到列表视图');
    } else {
        // 先显示图表容器
        listView.style.display = 'none';
        chartView.style.display = 'block';
        chartView.style.visibility = 'visible';
        
        console.log('图表视图容器已显示:', chartView);
        console.log(`图表容器尺寸: ${chartView.offsetWidth}x${chartView.offsetHeight}`);
        
        // 给DOM一点时间完成布局
        setTimeout(() => {
            try {
                // 获取图表实例方法1：从全局对象获取
                let charts = [];
                if (window.chartInstances) {
                    charts = [
                        window.chartInstances.pieChart,
                        window.chartInstances.barChart,
                        window.chartInstances.scatterChart
                    ].filter(chart => chart);
                }
                
                // 获取图表实例方法2：从DOM元素获取
                if (charts.length === 0) {
                    charts = [
                        echarts.getInstanceByDom(document.getElementById('sentimentPieChart')),
                        echarts.getInstanceByDom(document.getElementById('sentimentBarChart')),
                        echarts.getInstanceByDom(document.getElementById('sentimentScatterChart'))
                    ].filter(chart => chart);
                }
                
                console.log(`找到 ${charts.length} 个图表实例`);
                
                // 如果找不到图表实例，尝试重新初始化
                if (charts.length === 0) {
                    console.log('找不到图表实例，尝试重新初始化');
                    if (window.lastAnalysisData) {
                        console.log('使用存储的分析数据重新初始化图表');
                        initSentimentPieChart(window.lastAnalysisData);
                        initSentimentBarChart(window.lastAnalysisData);
                        initSentimentScatterChart(window.lastAnalysisData);
                    } else if (window.allSentences && window.allSentences.length > 0) {
                        console.log('使用存储的句子数据重新初始化图表');
                        // 构造数据对象
                        const data = {
                            sentences: window.allSentences,
                            overall: { 
                                probabilities: {
                                    positive: window.allSentences.filter(s => s.sentiment === '积极').length / window.allSentences.length * 100,
                                    negative: window.allSentences.filter(s => s.sentiment === '消极').length / window.allSentences.length * 100
                                }
                            }
                        };
                        
                        initSentimentPieChart(data);
                        initSentimentBarChart(data);
                        initSentimentScatterChart(data);
                    } else {
                        console.error('没有分析数据可用于初始化图表');
                    }
                    
                    // 重新获取图表实例
                    charts = [
                        echarts.getInstanceByDom(document.getElementById('sentimentPieChart')),
                        echarts.getInstanceByDom(document.getElementById('sentimentBarChart')),
                        echarts.getInstanceByDom(document.getElementById('sentimentScatterChart'))
                    ].filter(chart => chart);
                }
                
                // 调整所有图表大小
                charts.forEach(chart => {
                    if (chart) {
                        try {
                            chart.resize();
                            console.log('图表已调整大小');
                        } catch (e) {
                            console.error('调整图表大小失败:', e);
                        }
                    }
                });
            } catch (error) {
                console.error('处理图表视图切换时出错:', error);
            }
        }, 300);
    }

    // 更新标签按钮状态
    buttons.forEach(button => {
        if ((view === 'list' && button.textContent.includes('列表')) ||
            (view === 'chart' && button.textContent.includes('图表'))) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
    
    // 重新绑定标签按钮事件，确保切换后仍能点击
    setupTabButtonsEvents();
}

/**
 * 设置标签页按钮事件
 */
function setupTabButtonsEvents() {
    console.log('设置标签页按钮事件');
    
    // 移除所有现有事件，防止重复绑定
    document.querySelectorAll('#sentenceResults .tab-button').forEach(button => {
        const newButton = button.cloneNode(true);
        button.parentNode.replaceChild(newButton, button);
    });
    
    // 重新绑定事件
    document.querySelectorAll('#sentenceResults .tab-button').forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation(); // 防止事件冒泡
            const view = this.dataset.view;
            console.log(`标签按钮点击: ${view}`);
            switchTab(view);
        });
    });
}

/**
 * 切换词频视图
 * @param {string} view - 视图类型 ('tags' 或 'chart')
 */
function switchWordFreqTab(view) {
    console.log(`切换词频视图: ${view}`);
    const wordFreqTags = document.getElementById('wordFreqTags');
    const wordFreqCharts = document.getElementById('wordFreqCharts');
    const buttons = document.querySelectorAll('#wordFreq .tab-button');

    if (!wordFreqTags || !wordFreqCharts) {
        console.error('找不到词频视图容器元素:', {wordFreqTags, wordFreqCharts});
        return;
    }
    
    // 防止连续触发
    if (window.switchWordFreqTabInProgress) {
        console.log('词频视图切换正在进行中，请稍候...');
        return;
    }
    
    window.switchWordFreqTabInProgress = true;
    setTimeout(() => {
        window.switchWordFreqTabInProgress = false;
    }, 300);

    if (view === 'tags') {
        wordFreqTags.style.display = 'block';
        wordFreqCharts.style.display = 'none';
        console.log('已切换到词频标签视图');
    } else {
        // 先确保词频图表容器可见
        wordFreqTags.style.display = 'none';
        wordFreqCharts.style.display = 'flex';
        wordFreqCharts.style.visibility = 'visible';
        
        console.log('词频图表容器已显示');
        console.log(`词频图表容器尺寸: ${wordFreqCharts.offsetWidth}x${wordFreqCharts.offsetHeight}`);
        
        // 找到所有图表容器并检查其可见性
        const chartContainers = wordFreqCharts.querySelectorAll('.chart');
        chartContainers.forEach(container => {
            console.log(`图表容器: ${container.id}, 尺寸: ${container.offsetWidth}x${container.offsetHeight}`);
            // 确保每个容器都是可见的
            container.style.width = '100%';
            container.style.height = '350px';
            container.style.visibility = 'visible';
        });
        
        // 给DOM一点时间完成布局
        setTimeout(() => {
            try {
                // 获取图表实例方法1：从全局对象获取
                let charts = [];
                if (window.chartInstances) {
                    charts = [
                        window.chartInstances.wordFreqBarChart,
                        window.chartInstances.wordCloudChart
                    ].filter(chart => chart);
                }
                
                // 获取图表实例方法2：从DOM元素获取
                if (charts.length === 0) {
                    charts = [
                        echarts.getInstanceByDom(document.getElementById('wordFreqBarChart')),
                        echarts.getInstanceByDom(document.getElementById('wordCloudChart'))
                    ].filter(chart => chart);
                }
                
                console.log(`找到 ${charts.length} 个词频图表实例`);
                
                // 如果找不到图表实例，尝试重新初始化
                if (charts.length === 0 && window.lastAnalysisData) {
                    console.log('找不到词频图表实例，尝试重新初始化');
                    const wordFreqBarChart = initWordFreqBarChart(window.lastAnalysisData);
                    const wordCloudChart = initWordCloudChart(window.lastAnalysisData);
                    
                    if (wordFreqBarChart || wordCloudChart) {
                        charts = [wordFreqBarChart, wordCloudChart].filter(chart => chart);
                    }
                }
                
                // 调整所有图表大小
                charts.forEach(chart => {
                    if (chart) {
                        try {
                            chart.resize();
                            console.log('词频图表已调整大小');
                        } catch (e) {
                            console.error('调整词频图表大小失败:', e);
                        }
                    }
                });
            } catch (error) {
                console.error('处理词频图表视图切换时出错:', error);
            }
        }, 300);
    }

    // 更新标签按钮状态
    buttons.forEach(button => {
        if ((view === 'tags' && button.textContent.includes('标签')) ||
            (view === 'chart' && button.textContent.includes('图表'))) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
    
    // 重新绑定标签按钮事件
    setupWordFreqTabButtonsEvents();
}

/**
 * 设置词频标签页按钮事件
 */
function setupWordFreqTabButtonsEvents() {
    console.log('设置词频标签页按钮事件');
    
    // 移除所有现有事件，防止重复绑定
    document.querySelectorAll('#wordFreq .tab-button').forEach(button => {
        const newButton = button.cloneNode(true);
        button.parentNode.replaceChild(newButton, button);
    });
    
    // 重新绑定事件
    document.querySelectorAll('#wordFreq .tab-button').forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation(); // 防止事件冒泡
            const view = this.dataset.view;
            console.log(`词频标签按钮点击: ${view}`);
            switchWordFreqTab(view);
        });
    });
}

/**
 * 切换卡片展开/折叠状态
 * @param {number} index - 句子索引
 */
function toggleCard(index) {
    const card = document.querySelector(`.sentiment-card[data-index="${index}"]`);
    if (card) {
        card.classList.toggle('collapsed');
    }
}

/**
 * 设置显示模式
 * @param {string} mode - 显示模式 ('normal', 'compact', 或 'expanded')
 */
function setDisplayMode(mode) {
    displayMode = mode;
    updateDisplay();
}

/**
 * 设置情感筛选
 * @param {string} filter - 筛选模式 ('all', 'positive', 或 'negative')
 */
function setSentimentFilter(filter) {
    sentimentFilter = filter;
    currentPage = 1;
    updateDisplay();
}

/**
 * 切换功能区块
 * @param {string} sectionId - 区块ID
 */
function switchSection(sectionId) {
    console.log(`切换到区块: ${sectionId}, 是否管理员: ${window.isAdmin}`);
    
    // 首先记录滚动位置，以便切换回来时恢复
    const currentSection = document.querySelector('.content-section.active');
    if (currentSection) {
        currentSection.dataset.scrollPosition = currentSection.scrollTop;
    }

    // 隐藏所有内容区块
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.remove('active');
    });
    
    // 特殊处理管理员区域
    if (sectionId === 'admin-section') {
        // 检查用户是否为管理员
        if (window.isAdmin === true) {
            const adminSection = document.getElementById('admin-section');
            if (adminSection) {
                adminSection.classList.add('active');
                console.log('管理员区域已激活');
                
                // 如果已定义fetchUsers函数，调用它获取用户列表
                if (typeof fetchUsers === 'function') {
                    fetchUsers();
                } else {
                    console.warn('fetchUsers 函数未定义');
                }
            } else {
                console.error('找不到管理员区域元素');
            }
        } else {
            // 非管理员用户，显示输入区域
            console.log('非管理员用户尝试访问管理员区域，已重定向到输入区域');
            sectionId = 'input-section';
            document.getElementById('input-section').classList.add('active');
        }
    } else {
        // 普通区域，直接显示
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
        } else {
            console.error(`找不到目标区块: ${sectionId}`);
        }
    }
    
    // 恢复滚动位置
    const targetSection = document.getElementById(sectionId);
    if (targetSection && targetSection.dataset.scrollPosition) {
        targetSection.scrollTop = parseInt(targetSection.dataset.scrollPosition);
    }
    
    // 更新菜单项状态
    document.querySelectorAll('.menu-item').forEach(item => {
        item.classList.toggle('active', item.dataset.section === sectionId);
    });
    
    // 如果切换到词频统计区域，重新调整图表大小
    if (sectionId === 'word-freq-section' || sectionId === 'overall-section') {
        // 给DOM一点时间完成布局
        setTimeout(resizeChartsInSection, 100, sectionId);
    }

    // 如果切换到总体分析区域，重新调整图表大小
    if (sectionId === 'overall-section') {
        setTimeout(() => {
            console.log('调整总体分析区域中的所有图表大小');
            resizeChartsInSection(sectionId);
            
            // 特别处理词频图表
            const wordFreqCharts = document.getElementById('wordFreqCharts');
            if (wordFreqCharts && getComputedStyle(wordFreqCharts).display !== 'none') {
                const charts = [
                    echarts.getInstanceByDom(document.getElementById('wordFreqBarChart')),
                    echarts.getInstanceByDom(document.getElementById('wordCloudChart'))
                ].filter(chart => chart);
                
                charts.forEach(chart => {
                    if (chart) {
                        chart.resize();
                    }
                });
            }
        }, 300);
    }
}

/**
 * 根据区块ID调整该区块内的所有图表大小
 * @param {string} sectionId - 区块ID
 */
function resizeChartsInSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (!section) return;
    
    // 获取区块内的所有图表容器
    const chartContainers = section.querySelectorAll('.chart');
    chartContainers.forEach(container => {
        const chart = echarts.getInstanceByDom(container);
        if (chart) {
            chart.resize();
            
            // 双重保险：使用requestAnimationFrame确保在下一帧渲染时图表尺寸正确
            requestAnimationFrame(() => {
                chart.resize();
            });
        }
    });
}

/**
 * 设置界面事件监听器
 */
function setupUIEventListeners() {
    document.getElementById('perPageSelect').addEventListener('change', function () {
        sentencesPerPage = parseInt(this.value);
        currentPage = 1;
        updateDisplay();
    });

    document.querySelectorAll('.filter-button[data-sentiment]').forEach(button => {
        button.addEventListener('click', function () {
            document.querySelectorAll('.filter-button[data-sentiment]').forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');
            setSentimentFilter(this.dataset.sentiment);
        });
    });

    document.querySelectorAll('.filter-button[data-display]').forEach(button => {
        button.addEventListener('click', function () {
            document.querySelectorAll('.filter-button[data-display]').forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');
            setDisplayMode(this.dataset.display);
        });
    });

    // 添加模型选择下拉框的事件监听器
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', updateModelDescription);
    }
    
    // 添加侧边栏菜单点击事件 - 优化处理
    document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', function() {
            const targetSection = item.dataset.section;
            
            // 管理员区域特殊处理
            if (targetSection === 'admin-section') {
                if (window.isAdmin !== true) {
                    console.log('非管理员用户，无法访问系统管理功能');
                    alert('您没有管理员权限，无法访问系统管理功能');
                    return; // 非管理员用户点击无效
                }
            }
            
            switchSection(targetSection);
        });
    });
    
    // 监听窗口大小变化，重新调整当前活动区块的图表
    window.addEventListener('resize', function() {
        const activeSection = document.querySelector('.content-section.active');
        if (activeSection) {
            resizeChartsInSection(activeSection.id);
        }
    });

    // 设置标签页事件
    setupTabButtonsEvents();
    setupWordFreqTabButtonsEvents();
}
