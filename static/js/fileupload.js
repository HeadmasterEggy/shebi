/**
 * 文件上传相关功能
 */

// 当前上传的文件内容
let currentFileContent = '';
let currentFileName = '';

/**
 * 初始化文件上传相关事件监听
 */
function initFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const fileViewBtn = document.getElementById('fileViewBtn');
    const fileInfo = document.getElementById('fileInfo');
    const filePreview = document.getElementById('filePreview');
    const fileContent = document.getElementById('fileContent');
    const closePreview = document.getElementById('closePreview');

    // 监听文件选择事件
    fileInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (!file) {
            fileViewBtn.disabled = true;
            fileInfo.textContent = '';
            currentFileContent = '';
            currentFileName = '';
            return;
        }

        if (!file.name.toLowerCase().endsWith('.txt')) {
            showError('请上传 .txt 格式的文本文件');
            fileInput.value = '';
            return;
        }

        currentFileName = file.name;
        fileInfo.textContent = `已选择: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
        fileViewBtn.disabled = false;

        // 读取文件内容
        const reader = new FileReader();
        reader.onload = function (e) {
            currentFileContent = e.target.result;
        };
        reader.onerror = function () {
            showError('读取文件失败，请重试');
            currentFileContent = '';
        };
        reader.readAsText(file, 'UTF-8');
    });

    // 预览文件内容
    fileViewBtn.addEventListener('click', function () {
        if (!currentFileContent) {
            return;
        }

        fileContent.textContent = currentFileContent;
        filePreview.classList.remove('d-none');
    });

    // 关闭预览
    closePreview.addEventListener('click', function () {
        filePreview.classList.add('d-none');
    });
}

/**
 * 获取当前输入的文本（文本框输入或文件上传）
 * @returns {string} 输入文本
 */
function getInputText() {
    // 检查当前激活的输入标签页
    const textTabActive = document.getElementById('text-tab').classList.contains('active');

    if (textTabActive) {
        // 从文本框获取文本
        return document.getElementById('textInput').value.trim();
    } else {
        // 从上传的文件获取文本
        if (!currentFileContent) {
            showError('请先上传文件');
            return '';
        }
        return currentFileContent.trim();
    }
}

/**
 * 解析文件中的句子（每行作为一个句子）
 * @param {string} text 文件文本内容
 * @returns {string[]} 句子数组
 */
function parseSentencesFromFile(text) {
    if (!text) return [];

    // 按行分割，过滤空行
    return text.split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);
}

// 将文件功能暴露给全局，以便与其他模块集成
window.fileUploadModule = {
    getInputText,
    parseSentencesFromFile,
    getCurrentFileName: () => currentFileName
};
