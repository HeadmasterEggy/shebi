/**
 * 共享变量和工具函数
 */

// 全局变量
let currentPage = 1;
let sentencesPerPage = 10;
let allSentences = [];
let filteredSentences = [];
let sentimentFilter = 'all';
let displayMode = 'normal';

/**
 * 显示错误信息
 * @param {string} message - 错误消息
 */
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    document.getElementById('loading').style.display = 'none';
}
