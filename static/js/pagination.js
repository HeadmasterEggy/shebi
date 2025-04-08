/**
 * 分页相关功能
 */

/**
 * 更新显示内容
 */
function updateDisplay() {
    filteredSentences = allSentences.filter(sentence => {
        if (sentimentFilter === 'all') return true;
        return (sentimentFilter === 'positive' && sentence.sentiment === '积极') ||
               (sentimentFilter === 'negative' && sentence.sentiment === '消极');
    });

    document.getElementById('sentenceCount').textContent = `(${filteredSentences.length}条)`;

    const maxPage = Math.ceil(filteredSentences.length / sentencesPerPage);
    if (currentPage > maxPage && maxPage > 0) {
        currentPage = 1;
    }

    let startIndex = (currentPage - 1) * sentencesPerPage;
    let endIndex = Math.min(startIndex + sentencesPerPage, filteredSentences.length);
    let currentSentences = filteredSentences.slice(startIndex, endIndex);

    let sentenceResultsDiv = document.getElementById('sentenceList');
    let sentencesHtml = '';

    const cardClass = displayMode === 'compact' ? 'sentiment-card collapsed' :
                     (displayMode === 'expanded' ? 'sentiment-card expanded' : 'sentiment-card');

    currentSentences.forEach((sentence, index) => {
        const sentimentClass = sentence.sentiment === '积极' ? 'positive' : 'negative';
        const sentimentIcon = sentence.sentiment === '积极' ? '👍' : '👎';

        sentencesHtml += `
            <div class="${cardClass}" data-index="${startIndex + index}">
                <div class="sentiment-header" onclick="toggleCard(${startIndex + index})">
                    <span><strong>句子 ${startIndex + index + 1}</strong> ${sentimentIcon}</span>
                    <div>
                        <span class="sentiment-label ${sentimentClass}">${sentence.sentiment}</span>
                        <span class="toggle-icon">▼</span>
                    </div>
                </div>
                <div class="sentiment-content">${sentence.text}</div>
                <div class="sentiment-footer">
                    <div class="progress">
                        <div class="progress-bar bg-success" role="progressbar"
                            style="width: ${sentence.probabilities.positive.toFixed(2)}%">
                            正面 ${sentence.probabilities.positive.toFixed(2)}%
                        </div>
                        <div class="progress-bar bg-danger" role="progressbar"
                            style="width: ${sentence.probabilities.negative.toFixed(2)}%">
                            负面 ${sentence.probabilities.negative.toFixed(2)}%
                        </div>
                    </div>
                    <div>置信度: ${sentence.confidence.toFixed(2)}%</div>
                </div>
            </div>`;
    });

    sentenceResultsDiv.innerHTML = sentencesHtml || '<div class="alert alert-info">没有符合条件的句子</div>';

    document.getElementById('pageInfo').innerHTML =
        `第 ${currentPage} 页 / 共 ${Math.max(1, maxPage)} 页，共 ${filteredSentences.length} 条`;

    document.querySelector('.pagination li:first-child').classList.toggle('disabled', currentPage === 1);
    document.querySelector('.pagination li:last-child').classList.toggle('disabled',
        currentPage === maxPage || maxPage === 0);

    updatePageNumbers(maxPage);

    document.querySelectorAll('.word-freq-item').forEach((item, index) => {
        item.style.setProperty('--delay', index);
    });

    document.querySelectorAll('.chart-wrapper').forEach((wrapper, index) => {
        wrapper.style.animationDelay = `${index * 0.2}s`;
    });
}

/**
 * 切换页面
 * @param {number} delta - 页码变化量
 */
function changePage(delta) {
    let newPage = currentPage + delta;
    let maxPage = Math.ceil(filteredSentences.length / sentencesPerPage);

    if (newPage >= 1 && newPage <= maxPage) {
        currentPage = newPage;
        updateDisplay();
        document.getElementById('sentenceList').scrollIntoView({behavior: 'smooth'});
    }
}

/**
 * 跳转到指定页面
 * @param {number} page - 目标页码
 */
function goToPage(page) {
    let maxPage = Math.ceil(filteredSentences.length / sentencesPerPage);
    if (page >= 1 && page <= maxPage) {
        currentPage = page;
        updateDisplay();
        document.getElementById('sentenceList').scrollIntoView({behavior: 'smooth'});
    }
}

/**
 * 更新页码数字导航
 * @param {number} maxPage - 最大页数
 */
function updatePageNumbers(maxPage) {
    const pageNumbersContainer = document.getElementById('pageNumbers');
    let html = '';

    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(maxPage, startPage + 4);

    if (endPage - startPage < 4 && startPage > 1) {
        startPage = Math.max(1, endPage - 4);
    }

    if (startPage > 1) {
        html += `<div class="page-number" onclick="goToPage(1)">1</div>`;
        if (startPage > 2) {
            html += `<div class="page-number">...</div>`;
        }
    }

    for (let i = startPage; i <= endPage; i++) {
        html += `<div class="page-number ${i === currentPage ? 'active' : ''}" onclick="goToPage(${i})">${i}</div>`;
    }

    if (endPage < maxPage) {
        if (endPage < maxPage - 1) {
            html += `<div class="page-number">...</div>`;
        }
        html += `<div class="page-number" onclick="goToPage(${maxPage})">${maxPage}</div>`;
    }

    pageNumbersContainer.innerHTML = html;
}
