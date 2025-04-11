/**
 * 用户界面相关功能
 */

/**
 * 切换标签页
 * @param {string} view - 视图类型 ('list' 或 'chart')
 */
function switchTab(view) {
    const listView = document.getElementById('listView');
    const chartView = document.getElementById('chartView');
    const buttons = document.querySelectorAll('#sentenceResults .tab-button');

    if (view === 'list') {
        listView.style.display = 'block';
        chartView.style.display = 'none';
    } else {
        listView.style.display = 'none';
        chartView.style.display = 'block';
        setTimeout(() => {
            try {
                const pieChart = echarts.getInstanceByDom(document.getElementById('sentimentPieChart'));
                const barChart = echarts.getInstanceByDom(document.getElementById('sentimentBarChart'));
                const scatterChart = echarts.getInstanceByDom(document.getElementById('sentimentScatterChart'));
                if (pieChart) pieChart.resize();
                if (barChart) barChart.resize();
                if (scatterChart) scatterChart.resize();
            } catch (error) {
                console.error('调整图表大小失败:', error);
            }
        }, 0);
    }

    buttons.forEach(button => {
        button.classList.toggle('active',
            (view === 'list' && button.textContent === '列表视图') ||
            (view === 'chart' && button.textContent === '图表视图')
        );
    });
}

/**
 * 切换词频标签页
 * @param {string} view - 视图类型 ('tags' 或 'chart')
 */
function switchWordFreqTab(view) {
    const tagsView = document.getElementById('wordFreqTags');
    const chartsView = document.getElementById('wordFreqCharts');
    const buttons = document.querySelectorAll('#wordFreq .tab-button');

    if (view === 'tags') {
        tagsView.style.display = 'block';
        chartsView.style.display = 'none';
    } else {
        tagsView.style.display = 'none';
        chartsView.style.display = 'block';
        setTimeout(() => {
            try {
                const wordFreqBar = echarts.getInstanceByDom(document.getElementById('wordFreqBarChart'));
                const wordCloud = echarts.getInstanceByDom(document.getElementById('wordCloudChart'));
                if (wordFreqBar) wordFreqBar.resize();
                if (wordCloud) wordCloud.resize();
            } catch (error) {
                console.error('调整词频图表大小失败:', error);
            }
        }, 0);
    }

    buttons.forEach(button => {
        button.classList.toggle('active',
            (view === 'tags' && button.textContent === '标签视图') ||
            (view === 'chart' && button.textContent === '图表视图')
        );
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
}
