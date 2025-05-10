/**
 * 网络爬虫模块 - 用于从各电商平台获取评论数据
 */

/**
 * 爬取产品评论
 * @param {string} url - 商品链接
 * @param {function} progressCallback - 进度回调函数(progress, message)
 * @param {function} completeCallback - 完成回调函数(comments, error)
 */
function scrapeProductComments(url, progressCallback, completeCallback) {
    // 更新进度
    progressCallback(10, '正在连接服务器...');
    
    // 发送爬取请求
    fetch('/api/scrape/product', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url }),
    })
    .then(response => {
        progressCallback(30, '正在获取数据...');
        
        if (!response.ok) {
            throw new Error(`服务器返回错误: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        progressCallback(70, '正在处理评论数据...');
        
        // 提取评论数据
        let comments = [];
        
        if (data.comments && Array.isArray(data.comments)) {
            comments = data.comments.map(comment => comment.text);
            
            // 如果是从缓存加载的，进度直接到90%
            if (data.cached) {
                progressCallback(90, `从缓存加载了 ${comments.length} 条评论`);
            } else {
                progressCallback(90, `成功爬取 ${comments.length} 条评论`);
            }
        } else {
            progressCallback(90, '未找到评论数据');
            comments = [];
        }
        
        // 调用完成回调
        setTimeout(() => {
            progressCallback(100, `处理完成，共 ${comments.length} 条评论`);
            completeCallback(comments);
        }, 500);
    })
    .catch(error => {
        console.error('爬取数据失败:', error);
        completeCallback([], error.message);
    });
}

// 导出模块
window.webScraperModule = {
    scrapeProductComments
};
