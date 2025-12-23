# -*- coding:utf-8 -*-

import json
import random

import requests
import time
import csv
import os


"""
【1】爬取网易云音乐歌曲的精彩评论
【2】教程文章：https://mp.weixin.qq.com/s/tMVu8dUepSPIvm3yCMUt1g
【3】爬取动态渲染页面(使用 ajax 加载数据)

@Author monkey
@Date 2018-6-6
"""


def start_spider(song_id):
    """ 评论数据采用 AJAX 技术获得, 下面才是获取评论的请求地址 """
    url = 'http://music.163.com/weapi/v1/resource/comments/R_SO_4_{}?csrf_token='.format(song_id)

    headers = {
        'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        'Origin': 'http://music.163.com',
        'Referer': 'http://music.163.com/song?id={}'.format(song_id),
    }

    formdata = {
        'params': 'ooxqBCDRgGLrhgocQjD3ETWKa33Jr1BFcRF691S0X9NbvzAYDZgNLCP3I6wVzZKWVsn9inDZTprbY4vM4cBK4nG5eHSZcyJVxzkrAb9ik5Ix1vbY94KgoGnAWzQeQQRXcTPNDs7/n0CysMtecT4Hxg==',
        'encSecKey': 'bb5aee4bb87bcba29df4b8f19b74e9dbfea9a2b3eeef5db34b41e5666dffb0d9d901065e2968eebd842c03ad9f081a0b1b9c12e7869bfbc25117b1e230ad982e7c839a1e257b22ed8cb49f63e6b599c40db344ea79b8f97e04243775364b8464cac14d09c50778a144e59c5a3fe5c81eb23281527958f4f3914adaf5b02ccabc',
    }

    response = requests.post(url, headers=headers, data=formdata)
    print('请求 [ ' + url + ' ], 状态码为 ')
    print(response.status_code)
    # get_hot_comments(response.text)
    # 将数据写到 CSV 文件中
    write_to_file(get_hot_comments(response.text))


def get_hot_comments(response):
    """ 获取精彩评论
    请求返回结果是 Json 数据格式, 使用 json.loads(response) 将其转化为字典类型, 就可以使用 key-value 形式获取值
    """
    data_list = []
    data = {}

    for comment in json.loads(response)['hotComments']:
        data['userId'] = comment['user']['userId']
        data['nickname'] = comment['user']['nickname']
        data['content'] = comment['content']
        data['likedCount'] = comment['likedCount']
        data_list.append(data)
        data = {}
    # print(data_list)
    return data_list


def write_to_file(datalist):
    print('开始将数据持久化……')
    file_name = 'hotcomments.csv'

    filednames = ['userId', 'nickname', 'content', 'likedCount']
    need_header = (not os.path.exists(file_name)) or (os.path.getsize(file_name) == 0)

    # 用 utf-8-sig：Excel/VS Code 打开中文更稳定（带 BOM）
    with open(file_name, 'a', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=filednames)

        if need_header:
            writer.writeheader()

        for data in datalist:
            print(data)
            writer.writerow({
                filednames[0]: data.get('userId'),
                filednames[1]: data.get('nickname'),
                filednames[2]: data.get('content'),
                filednames[3]: data.get('likedCount'),
            })

    print('成功将数据写入到 ' + file_name + ' 中！')


def get_song_id(url):
    """ 从 url 中截取歌曲的 id """
    song_id = url.split('=')[1]
    return song_id


def main():
    songs_url_list = [
        'https://music.163.com/#/song?id=2747400519',
        'https://music.163.com/song?id=2147432142'
    ]

    for each in songs_url_list:
        start_spider(get_song_id(each))
        time.sleep(random.randint(5, 8))


if __name__ == '__main__':
    main()

