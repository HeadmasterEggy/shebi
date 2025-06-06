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
                // 获取所有图表实例并调整大小
                const chartIds = [
                    'sentimentPieChart', 
                    'sentimentBarChart', 
                    'sentimentScatterChart',
                    'sentimentRadarChart'  // 添加雷达图
                ];
                
                chartIds.forEach(id => {
                    const chart = echarts.getInstanceByDom(document.getElementById(id));
                    if (chart) {
                        try {
                            chart.resize();
                            console.log(`已调整 ${id} 图表大小`);
                        } catch (e) {
                            console.error(`调整 ${id} 图表大小失败:`, e);
                        }
                    } else {
                        console.log(`找不到图表实例: ${id}`);
                    }
                });
                
                // 如果找不到图表实例，尝试重新初始化
                if (window.lastAnalysisData) {
                    console.log('使用存储的分析数据重新初始化图表');
                    initSentimentPieChart(window.lastAnalysisData);
                    initSentimentBarChart(window.lastAnalysisData);
                    initSentimentScatterChart(window.lastAnalysisData);
                    
                    // 检查并初始化雷达图
                    if (!echarts.getInstanceByDom(document.getElementById('sentimentRadarChart'))) {
                        console.log('重新初始化情感雷达图');
                        initSentimentRadarChart(window.lastAnalysisData);
                    }
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
                    
                    // 检查并初始化雷达图
                    if (!echarts.getInstanceByDom(document.getElementById('sentimentRadarChart'))) {
                        console.log('重新初始化情感雷达图');
                        initSentimentRadarChart(data);
                    }
                } else {
                    console.error('没有分析数据可用于初始化图表');
                }
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
        button.addEventListener('click', function (e) {
            e.stopPropagation(); // 防止事件冒泡
            const view = this.dataset.view;
            console.log(`标签按钮点击: ${view}`);
            switchTab(view);
        });
    });
}

/**
 * 初始化词频展示区域
 * 将原来的切换函数改为初始化函数，确保两个区域都正确显示
 */
function initWordFreqDisplay() {
    const wordFreqTags = document.getElementById('wordFreqTagsContent');
    const wordFreqCharts = document.getElementById('wordFreqCharts');
    
    if (!wordFreqTags || !wordFreqCharts) {
        console.error('找不到词频视图容器');
        return;
    }

    console.log('初始化词频展示区域');
    
    // 确保词频标签区域正确显示
    wordFreqTags.style.display = 'block';
    
    // 确保图表区域样式正确
    wordFreqCharts.style.display = 'grid';
    wordFreqCharts.style.gridTemplateColumns = 'repeat(auto-fit, minmax(300px, 1fr))';
    wordFreqCharts.style.gap = '20px';
    wordFreqCharts.style.width = '100%';
    wordFreqCharts.style.visibility = 'visible';
    
    // 确保图表容器显示正确
    const chartWrappers = wordFreqCharts.querySelectorAll('.chart-wrapper');
    chartWrappers.forEach(wrapper => {
        wrapper.style.width = '100%';
        wrapper.style.height = '300px';
    });
    
    // 确保图表容器有明确的尺寸
    const chartContainers = wordFreqCharts.querySelectorAll('.chart');
    chartContainers.forEach(container => {
        container.style.width = '100%';
        container.style.height = '300px';
        container.style.minHeight = '250px';
        container.style.visibility = 'visible';
        container.style.opacity = '1';
        console.log(`设置图表容器 ${container.id} 尺寸: ${container.style.width}x${container.style.height}`);
    });

    // 延迟处理图表渲染，确保DOM完全更新
    setTimeout(() => {
        console.log('重新渲染词频图表...');
        
        // 特殊处理词云图
        if (window.chartInstances && window.chartInstances.wordCloudChart) {
            try {
                window.chartInstances.wordCloudChart.resize();
                console.log('已调整词云图大小');
            } catch (e) {
                console.error('调整词云图大小失败:', e);
            }
        } else if (window.lastAnalysisData) {
            console.log('词云图实例不存在，尝试重新创建');
            initWordCloudChart(window.lastAnalysisData);
        }
        
        // 处理柱状图
        if (window.chartInstances && window.chartInstances.wordFreqBarChart) {
            try {
                window.chartInstances.wordFreqBarChart.resize();
                console.log('已调整词频柱状图大小');
            } catch (e) {
                console.error('调整词频柱状图大小失败:', e);
            }
        } else if (window.lastAnalysisData) {
            console.log('词频柱状图实例不存在，尝试重新创建');
            initWordFreqBarChart(window.lastAnalysisData);
        }
        
        // 如果有专门的词云图渲染函数，调用它
        if (typeof window.renderWordCloud === 'function') {
            setTimeout(window.renderWordCloud, 500);
        }
    }, 600); // 增加延迟确保DOM已完全更新
}

/**
 * 创建词频图表
 */
function createWordFreqCharts(data) {
    if (!data || !data.wordFreq) return;

    // 清理旧实例
    ['wordFreqBarChart', 'wordCloudChart'].forEach(id => {
        const container = document.getElementById(id);
        if (!container) return;

        try {
            const existing = echarts.getInstanceByDom(container);
            if (existing) existing.dispose();
        } catch (e) {
        }

        // 创建图表
        createSimpleChart(container, data.wordFreq);
    });
}

/**
 * 创建简单图表
 */
function createSimpleChart(container, data) {
    if (!container || !data) return null;

    // 确保容器尺寸
    container.style.width = '100%';
    container.style.height = '350px';

    try {
        const chart = echarts.init(container);
        const isBarChart = container.id === 'wordFreqBarChart';

        // 准备图表数据
        const words = data.slice(0, isBarChart ? 15 : 30).map(item => item.word);
        const values = data.slice(0, isBarChart ? 15 : 30).map(item => item.count);

        // 设置配置
        const option = isBarChart ?
            {
                title: {text: '高频词汇统计', left: 'center'},
                tooltip: {trigger: 'axis'},
                xAxis: {type: 'category', data: words},
                yAxis: {type: 'value'},
                series: [{data: values, type: 'bar'}]
            } :
            {
                title: {text: '词频分布', left: 'center'},
                tooltip: {trigger: 'item'},
                series: [{
                    type: 'pie',
                    radius: '60%',
                    data: words.map((word, i) => ({name: word, value: values[i]}))
                }]
            };

        chart.setOption(option);
        chart.resize();
        return chart;
    } catch (e) {
        console.error(`图表创建失败: ${e.message}`);
        return null;
    }
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

    // 当切换到词频统计区域时，确保图表正确显示
    if (sectionId === 'word-freq-section') {
        setTimeout(() => {
            initWordFreqDisplay();
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
 * 初始化网络爬虫功能
 * 添加网络爬虫相关UI元素和事件监听
 */
function initWebScraper() {
    console.log('初始化网络爬虫功能');
    
    // 创建链接输入组件
    const linkInputGroup = document.createElement('div');
    linkInputGroup.className = 'web-scraper-container mt-3';
    linkInputGroup.innerHTML = `
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h6 class="mb-0">
                    <i class="bi bi-link-45deg me-2"></i>从网页获取文本
                </h6>
                <button type="button" class="btn btn-sm btn-link text-decoration-none toggle-scraper-panel">
                    <i class="bi bi-chevron-down"></i>
                </button>
            </div>
            <div class="card-body scraper-panel">
                <div class="mb-3">
                    <label for="productUrl" class="form-label">输入商品链接（京东）:</label>
                    <div class="input-group">
                        <input type="url" class="form-control" id="productUrl" 
                            placeholder="https://item.jd.com/123456789.html">
                        <button class="btn btn-primary" type="button" id="scrapeButton">
                            <i class="bi bi-cloud-download me-1"></i>获取评论
                        </button>
                    </div>
                    <div class="form-text">系统将爬取商品评论并自动填入文本框</div>
                </div>
                <div id="scrapeStatus" class="d-none">
                    <div class="progress mb-2">
                        <div id="scrapeProgressBar" class="progress-bar progress-bar-striped progress-bar-animated" 
                            role="progressbar" style="width: 0%"></div>
                    </div>
                    <p class="small text-muted mb-0" id="scrapeStatusText">正在准备...</p>
                </div>
            </div>
        </div>
    `;
    
    // 获取文本输入区域
    const textTab = document.getElementById('textInput-tab');
    const textInput = document.getElementById('textInput');
    
    if (textTab && textInput) {
        // 插入到文本输入框后面
        textInput.parentNode.insertBefore(linkInputGroup, textInput.nextSibling);
        
        // 绑定面板切换事件
        const toggleButton = linkInputGroup.querySelector('.toggle-scraper-panel');
        const scraperPanel = linkInputGroup.querySelector('.scraper-panel');
        
        toggleButton.addEventListener('click', function() {
            const isVisible = scraperPanel.style.display !== 'none';
            scraperPanel.style.display = isVisible ? 'none' : 'block';
            toggleButton.querySelector('i').classList.toggle('bi-chevron-down', !isVisible);
            toggleButton.querySelector('i').classList.toggle('bi-chevron-up', isVisible);
        });
        
        // 绑定爬取按钮事件
        const scrapeButton = document.getElementById('scrapeButton');
        scrapeButton.addEventListener('click', function() {
            startScraping();
        });
        
        // 链接输入框添加回车触发爬取
        const productUrlInput = document.getElementById('productUrl');
        productUrlInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                startScraping();
            }
        });
    }
}

/**
 * 开始网络爬虫操作
 */
function startScraping() {
    const urlInput = document.getElementById('productUrl');
    const url = urlInput.value.trim();
    
    if (!url) {
        showError('请输入有效的商品链接');
        return;
    }
    
    // 验证URL格式
    if (!url.match(/^https?:\/\/(item\.)?(jd|jingdong)\.com\/\d+\.html/)) {
        showError('请输入有效的京东商品链接，例如：https://item.jd.com/123456789.html');
        return;
    }
    
    // 显示爬取状态
    const scrapeStatus = document.getElementById('scrapeStatus');
    const progressBar = document.getElementById('scrapeProgressBar');
    const statusText = document.getElementById('scrapeStatusText');
    
    scrapeStatus.classList.remove('d-none');
    progressBar.style.width = '10%';
    statusText.textContent = '正在连接到服务器...';
    
    // 禁用按钮
    const scrapeButton = document.getElementById('scrapeButton');
    scrapeButton.disabled = true;
    
    // 调用爬虫API
    window.webScraperModule.scrapeProductComments(url, 
        // 进度回调
        function(progress, message) {
            progressBar.style.width = `${progress}%`;
            statusText.textContent = message;
        },
        // 完成回调
        function(comments, error) {
            scrapeButton.disabled = false;
            
            if (error) {
                progressBar.classList.remove('bg-primary');
                progressBar.classList.add('bg-danger');
                statusText.textContent = `爬取失败: ${error}`;
                setTimeout(() => {
                    scrapeStatus.classList.add('d-none');
                    progressBar.classList.add('bg-primary');
                    progressBar.classList.remove('bg-danger');
                }, 3000);
                return;
            }
            
            progressBar.style.width = '100%';
            statusText.textContent = `成功获取 ${comments.length} 条评论数据`;
            
            // 填充到文本输入框
            const textInput = document.getElementById('textInput');
            textInput.value = comments.join('\n\n');
            
            // 3秒后隐藏进度条
            setTimeout(() => {
                scrapeStatus.classList.add('d-none');
                progressBar.style.width = '0%';
            }, 3000);
        }
    );
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
        item.addEventListener('click', function () {
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
    window.addEventListener('resize', function () {
        const activeSection = document.querySelector('.content-section.active');
        if (activeSection) {
            resizeChartsInSection(activeSection.id);
        }
    });

    // 设置标签页事件
    setupTabButtonsEvents();

    // 初始化网络爬虫功能
    initWebScraper();
}
