# -*- coding:utf-8 -*-

import pymongo
from pymongo.errors import PyMongoError
import random
import time
from math import ceil
from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import json
import argparse

"""
【1】教程文章：https://mp.weixin.qq.com/s/kcA-6WEHWQ-DOwxtWtYjWw
【2】使用 Selenium 爬取《Five Hundred Miles》 在网易云音乐歌曲的所有评论
【3】数据存储到 Mongo 数据库中
   
@Author monkey
@Date 2018-6-10
"""

MONGO_HOST = '127.0.0.1'
MONGO_DB = '163Music'
MONGO_COLLECTION = 'comments'

client = pymongo.MongoClient(MONGO_HOST, serverSelectionTimeoutMS=3000)
db_manager = client[MONGO_DB]

DEFAULT_JSONL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'comments.jsonl'))


def save_data_to_jsonl(data_list, path=DEFAULT_JSONL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for row in data_list:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
    print(f"[INFO] 已写入本地JSONL: {path} (+{len(data_list)})")

def login(brower):
    brower.get("https://music.163.com/")
    brower.execute_script("document.body.style.zoom='70%';")
    print("[DEBUG] 页面缩放已设置为70%")

    def _get_music_u_value():
        try:
            for c in brower.get_cookies():
                if c.get('name') == 'MUSIC_U':
                    return c.get('value')
        except Exception:
            return None
        return None

    before_music_u = _get_music_u_value()

    # 点击登录 
    login_button = brower.find_element(By.XPATH, "//div[contains(@class, 'm-tophead')]//a[text()='登录']")
    login_button.click()

    # 检测是否登录：等待 MUSIC_U cookie 出现或发生变化
    timeout = 180
    poll_interval = 0.5
    start = time.time()
    while time.time() - start < timeout:
        current_music_u = _get_music_u_value()
        if current_music_u and current_music_u != before_music_u:
            print("[INFO] 检测到 MUSIC_U cookie 变化，认为已登录")
            return
        time.sleep(poll_interval)

    print("[WARN] 等待登录超时：未检测到 MUSIC_U cookie 变化（可能未完成登录/被风控/验证码拦截）")
    

def start_spider(url, max_pages=None):
    """ 启动 Chrome 浏览器访问页面 """
    """
    # 从 Chrome 59 版本, 支持 Headless 模式(无界面模式), 即不会弹出浏览器
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    brower = webdriver.Chrome(chrome_options=chrome_options)
    """
    brower = webdriver.Chrome()
    login(brower)
    brower.get(url)
    # 等待 5 秒, 让评论数据加载完成
    time.sleep(5)
    # 页面嵌套一层 iframe, 必须切换到 iframe, 才能定位的到 iframe 里面的元素
    iframe = brower.find_element(By.CLASS_NAME, 'g-iframe')
    brower.switch_to.frame(iframe)
    # 增加一层保护, 拉动滚动条到底部
    js = "var q=document.documentElement.scrollTop=20000"
    brower.execute_script(js)

    # 获取【最新评论】总数
    new_comments = brower.find_elements(By.XPATH, "//h3[@class='u-hd4']")[1]

    max_page = get_max_page(new_comments.text)
    if max_pages is not None:
        try:
            max_pages_int = int(max_pages)
        except Exception:
            max_pages_int = None
        if max_pages_int is not None and max_pages_int > 0:
            max_page = min(max_page, max_pages_int)
    current = 1
    is_first = True
    while current <= max_page:
        print('正在爬取第', current, '页的数据')
        if current == 1:
            is_first = True
        else:
            is_first = False
        data_list = get_comments(is_first, brower)
        save_data_to_mongo(data_list)
        time.sleep(1)
        go_nextpage(brower)
        # 模拟人为浏览
        time.sleep(random.randint(8, 12))
        current += 1


def get_comments(is_first, brower):
    """ 获取评论数据 """
    items = brower.find_elements(By.XPATH, "//div[@class='cmmts j-flag']/div[@class='itm']")
    # 首页的数据中包含 15 条精彩评论, 20 条最新评论, 只保留最新评论
    if is_first:
        items = items[15: len(items)]

    data_list = []
    data = {}
    for each in items:
        # 用户 id
        userId = each.find_elements(By.XPATH, "./div[@class='head']/a")[0]
        userId = userId.get_attribute('href').split('=')[1]
        # 用户昵称
        nickname = each.find_elements(By.XPATH, "./div[@class='cntwrap']/div[1]/div[1]/a")[0]
        nickname = nickname.text
        # 评论内容
        content = each.find_elements(By.XPATH, "./div[@class='cntwrap']/div[1]/div[1]")[0]
        content = content.text.split('：')[1]  # 中文冒号
        # 点赞数
        like = each.find_elements(By.XPATH, "./div[@class='cntwrap']/div[@class='rp']/a[1]")[0]
        like = like.text
        if like:
            like = like.strip().split('(')[1].split(')')[0]
        else:
            like = '0'
        # 头像地址
        avatar = each.find_elements(By.XPATH, "./div[@class='head']/a/img")[0]
        avatar = avatar.get_attribute('src')

        data['userId'] = userId
        data['nickname'] = nickname
        data['content'] = content
        data['like'] = like
        data['avatar'] = avatar
        print(data)
        data_list.append(data)
        data = {}
    return data_list


def save_data_to_mongo(data_list):
    """ 一次性插入 20 条评论。
        插入效率高, 降低数据丢失风险
    """
    collection = db_manager[MONGO_COLLECTION]
    try:
        result = collection.insert_many(data_list, ordered=False)
        if result and result.inserted_ids:
            print('成功插入', len(result.inserted_ids), '条数据')
    except PyMongoError as e:
        print(f"[WARN] MongoDB 插入失败：{type(e).__name__}: {e}")
        # 更适合数据科学作业：至少先落盘，后续再用 pandas/duckdb/sqlite 处理
        save_data_to_jsonl(data_list)
    except Exception as e:
        print(f"[WARN] 插入数据出现未知异常：{type(e).__name__}: {e}")
        save_data_to_jsonl(data_list)


def go_nextpage(brower):
    """ 模拟人为操作, 点击【下一页】 """
    next_button = brower.find_elements(By.XPATH, "//div[@class='m-cmmt']/div[3]/div[1]/a")[-1]
    if next_button.text == '下一页':
        next_button.click()


def get_max_page(new_comments):
    """ 根据评论总数, 计算出总分页数 """
    print('=== ' + new_comments + ' ===')
    max_page = new_comments.split('(')[1].split(')')[0]
    # 每页显示 20 条最新评论
    offset = 20
    max_page = ceil(int(max_page) / offset)
    print('一共有', max_page, '个分页')
    return max_page


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Netease Music comments crawler (Selenium)')
    parser.add_argument('--song-id', type=str, default='27759600', help='网易云歌曲 ID（默认 27759600）')
    parser.add_argument('--pages', type=int, default=None, help='要爬取的页数上限（默认不限制，按网站总页数）')
    args = parser.parse_args()

    url = f'http://music.163.com/#/song?id={args.song_id}'
    start_spider(url, max_pages=args.pages)
