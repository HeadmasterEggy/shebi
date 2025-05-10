import json
import os
import time

from DrissionPage import ChromiumPage

# 创建目录
os.makedirs("csv", exist_ok=True)
os.makedirs("txt", exist_ok=True)

url = input("请输入商品链接：")
pid = url.split("/")[-1].split(".")[0]


def save_comments_to_files(data_list):
    """保存评论数据到csv和txt文件"""

    # 准备CSV数据
    csv_data = []
    for data in data_list:
        if "result" in data and "floors" in data["result"]:
            for floor in data["result"]["floors"]:
                if floor.get("floorId") == "flo_11103":
                    for data in floor["data"]:
                        comment_info = data.get("commentInfo", {})
                        comment_data = comment_info.get("commentData", "")
                        processed_data = comment_data.replace("\n", "").replace(
                            ",", "，"
                        )
                        if not processed_data:
                            continue
                        csv_data.append(
                            [
                                comment_info.get("commentDate", ""),
                                processed_data,  # 使用处理后的文本
                                comment_info.get("commentScore", ""),
                                comment_info.get("userNickName", ""),
                            ]
                        )

    # 保存到CSV
    if csv_data:
        import pandas as pd

        df = pd.DataFrame(
            csv_data,
            columns=["commentDate", "commentData", "commentScore", "userNickName"],
        )
        df.to_csv(f"csv/{pid}.csv", index=False, encoding="utf-8-sig")

        # 单独保存commentData到TXT
        with open(f"txt/{pid}.txt", "w", encoding="utf-8") as f:
            for row in csv_data:
                f.write(row[1] + "\n")


def main():
    # 创建页面对象，使用edge浏览器
    page = ChromiumPage()

    # 监听目标API
    page.listen.start("https://api.m.jd.com/client.action")

    # 打开商品页面
    page.get(url)

    # 等待加载后点击"全部评价"按钮
    page("#comment").ele("text=全部评价").click()
    time.sleep(3)

    page.listen.wait(count=7)

    # 滚动并收集数据
    collected_data = []
    for _ in range(20):
        cards = page.eles(".jdc-pc-rate-card")
        if cards:
            # 获取最后一个卡片并悬浮
            print("悬浮最后一个卡片")
            last_card = cards[-1]
            last_card.hover()
            last_card.scroll(800)
            time.sleep(5)
            print("滚动完成")

        # 获取监听数据
        resp = page.listen.wait(timeout=5)
        if resp is not None and hasattr(resp, "response") and resp.response.body:
            print("收集数据")
            collected_data.append(resp.response.body)
    print("收集完成")
    # 处理收集到的数据
    save_comments_to_files(collected_data)

    # 关闭页面
    # page.close()


if __name__ == "__main__":
    main()
