import json
import os
import time
import logging
from flask import Blueprint, request, jsonify
from DrissionPage import ChromiumPage

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建蓝图
scraper_bp = Blueprint('scraper', __name__, url_prefix='/api/scrape')

# 创建目录
os.makedirs("csv", exist_ok=True)
os.makedirs("txt", exist_ok=True)
os.makedirs("cache", exist_ok=True)

def extract_product_id(url):
    """从URL提取商品ID"""
    try:
        # 尝试从标准格式提取：https://item.jd.com/100008348542.html
        pid = url.split("/")[-1].split(".")[0]
        # 确保提取到的是数字ID
        if pid.isdigit():
            return pid
        
        # 尝试其他可能的格式，比如带参数的URL
        import re
        match = re.search(r'/(\d+)\.html', url)
        if match:
            return match.group(1)
        
        return None
    except Exception as e:
        logger.error(f"提取商品ID失败: {e}")
        return None

def save_comments_to_files(data_list, pid):
    """保存评论数据到csv和txt文件"""
    try:
        # 准备CSV数据
        csv_data = []
        comments_text = []
        
        for data in data_list:
            if isinstance(data, str):
                # 尝试解析JSON
                try:
                    data = json.loads(data)
                except:
                    continue
                    
            if "result" in data and "floors" in data["result"]:
                for floor in data["result"]["floors"]:
                    if floor.get("floorId") == "flo_11103":
                        for item in floor["data"]:
                            comment_info = item.get("commentInfo", {})
                            comment_data = comment_info.get("commentData", "")
                            processed_data = comment_data.replace("\n", " ").replace(",", "，")
                            if not processed_data:
                                continue
                                
                            csv_data.append([
                                comment_info.get("commentDate", ""),
                                processed_data,
                                comment_info.get("commentScore", ""),
                                comment_info.get("userNickName", ""),
                            ])
                            
                            comments_text.append(processed_data)

        # 保存到CSV
        if csv_data:
            import pandas as pd
            df = pd.DataFrame(
                csv_data,
                columns=["commentDate", "commentData", "commentScore", "userNickName"],
            )
            df.to_csv(f"csv/{pid}.csv", index=False, encoding="utf-8-sig")

            # 单独保存commentData到TXT
            txt_path = f"txt/{pid}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for comment in comments_text:
                    f.write(comment + "\n")
                    
            return comments_text
        return []
    except Exception as e:
        logger.error(f"保存评论数据失败: {e}")
        return []

@scraper_bp.route('/product', methods=['POST'])
def scrape_product():
    """爬取商品评论数据的API端点"""
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': '缺少商品URL'}), 400
            
        # 提取商品ID
        pid = extract_product_id(url)
        if not pid:
            return jsonify({'error': '无法从URL提取商品ID'}), 400
            
        # 检查缓存，避免重复爬取
        cache_file = f"cache/{pid}.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # 缓存最多保留1天
                if time.time() - cache_data.get('timestamp', 0) < 86400:
                    logger.info(f"使用缓存的评论数据: {pid}")
                    return jsonify({
                        'comments': cache_data.get('comments', []),
                        'cached': True
                    })
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
        
        logger.info(f"开始爬取商品评论: {url}")
        
        # 创建页面对象
        page = ChromiumPage()

        # 监听目标API
        page.listen.start("https://api.m.jd.com/client.action")

        # 打开商品页面
        page.get(url)
        logger.info("已打开商品页面")

        # 等待加载后点击"全部评价"按钮
        try:
            comment_btn = page.ele("#comment") or page.ele("text=商品评价")
            if comment_btn:
                comment_btn.click()
                logger.info("点击了商品评价按钮")
                time.sleep(2)
                
            # 尝试查找并点击"全部评价"按钮
            all_comments_btn = page.ele("text=全部评价")
            if all_comments_btn:
                all_comments_btn.click()
                logger.info("点击了全部评价按钮")
                time.sleep(2)
        except Exception as e:
            logger.warning(f"点击评论按钮失败: {e}")

        # 获取页面上已有的评论
        page.listen.wait(count=5, timeout=8)

        # 滚动并收集数据
        collected_data = []
        comments_count = 0
        
        # 最多滚动10次
        for i in range(8):
            cards = page.eles(".jdc-pc-rate-card") or page.eles(".comment-item")
            if cards:
                # 获取最后一个卡片并悬浮
                logger.info(f"找到{len(cards)}个评论卡片，悬浮最后一个")
                last_card = cards[-1]
                last_card.hover()
                last_card.scroll(800)
                time.sleep(2)
            
            # 获取监听数据
            resp = page.listen.wait(timeout=3)
            if resp is not None and hasattr(resp, 'response') and resp.response.body:
                collected_data.append(resp.response.body)
                comments_count += 1
                logger.info(f"收集到第{comments_count}批评论数据")
        
        # 关闭页面
        page.close()
        
        # 处理收集到的数据
        comments_list = save_comments_to_files(collected_data, pid)
        
        # 如果成功提取评论，保存到缓存
        if comments_list:
            cache_data = {
                'comments': [{'text': text} for text in comments_list],
                'timestamp': time.time()
            }
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False)
                logger.info(f"已缓存{len(comments_list)}条评论数据")
            except Exception as e:
                logger.warning(f"保存缓存失败: {e}")
        
        return jsonify({
            'comments': [{'text': text} for text in comments_list],
            'count': len(comments_list)
        })
        
    except Exception as e:
        logger.error(f"爬取数据失败: {e}")
        return jsonify({'error': f'爬取数据失败: {str(e)}'}), 500
