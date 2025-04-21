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
        
        // 给DOM一点时间完成布局
        setTimeout(() => {
            try {
                const pieChart = echarts.getInstanceByDom(document.getElementById('sentimentPieChart'));
                const barChart = echarts.getInstanceByDom(document.getElementById('sentimentBarChart'));
                const scatterChart = echarts.getInstanceByDom(document.getElementById('sentimentScatterChart'));
                
                if (pieChart) pieChart.resize();
                if (barChart) barChart.resize();
                if (scatterChart) scatterChart.resize();
                
                // 使用requestAnimationFrame确保在下一帧渲染时图表尺寸正确
                requestAnimationFrame(() => {
                    if (pieChart) pieChart.resize();
                    if (barChart) barChart.resize();
                    if (scatterChart) scatterChart.resize();
                });
            } catch (error) {
                console.error('调整图表大小失败:', error);
            }
        }, 100);
    }

    buttons.forEach(button => {
        button.classList.toggle('active',
            (view === 'list' && button.textContent === '列表视图') ||
            (view === 'chart' && button.textContent === '图表视图')
        );
    });
}

// 移除 switchWordFreqTab 函数，因为我们不再需要切换视图

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
        if (typeof isAdmin !== 'undefined' && isAdmin) {
            const adminSection = document.getElementById('admin-section');
            adminSection.classList.add('active');
            
            // 如果已定义fetchUsers函数，调用它获取用户列表
            if (typeof fetchUsers === 'function') {
                fetchUsers();
            }
        } else {
            // 非管理员用户，显示输入区域
            sectionId = 'input-section';
            document.getElementById('input-section').classList.add('active');
        }
    } else {
        // 普通区域，直接显示
        document.getElementById(sectionId).classList.add('active');
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
            // 管理员区域特殊处理
            if (item.dataset.section === 'admin-section' && 
                (!window.isAdmin || typeof window.isAdmin === 'undefined')) {
                console.log('非管理员用户，无法访问系统管理功能');
                return; // 非管理员用户点击无效
            }
            
            switchSection(item.dataset.section);
        });
    });
    
    // 监听窗口大小变化，重新调整当前活动区块的图表
    window.addEventListener('resize', function() {
        const activeSection = document.querySelector('.content-section.active');
        if (activeSection) {
            resizeChartsInSection(activeSection.id);
        }
    });
}
