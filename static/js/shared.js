/**
 * 共享变量和工具函数
 */

/**
 * 把任意值转成可以安全拼进 innerHTML / HTML 属性的文本。
 *
 * 页面上展示的句子来自用户粘贴或上传的文本，用户名、文件名、历史记录标题也都是
 * 用户可控的。直接拼进模板字符串再赋给 innerHTML，一条内容为
 * <img src=x onerror=...> 的评论就能在当前登录用户的会话里执行脚本。
 */
function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// 全局变量
let currentPage = 1;
let sentencesPerPage = 5;
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
