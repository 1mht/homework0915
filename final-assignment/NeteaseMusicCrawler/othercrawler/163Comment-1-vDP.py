# 配置方法

# from DrissionPage import ChromiumOptions
# path = r'"C:\Program Files\Google\Chrome\Application\chrome.exe"'   # 电脑内Chrome可执行文件路径
# ChromiumOptions().set_browser_path(path).save()

# 可执行路径查找：右键点击桌面GoogleChrome快捷键--打开文件所在位置--复制Chrome.exe文件地址即可【Edge浏览器同理】
# DrissionPage只需要在第一次使用时配置浏览器即可。


# -*- coding：utf-8 -*-
# @author：筑基期摸鱼大师

# 导入包
from DrissionPage import Chromium
import csv
import time

# 1.启动浏览器，并获取获取最新的标签页对象。
driver = Chromium().latest_tab

# 2.监听数据包【F12开发者工具--网络--按关键字搜索数据包】
driver.listen.start('comments/get?csrf_token')

# 访问网站
driver.get('http://music.163.com/#/song?id=33599620')

# 3.创建csv文件以保存数据
filename = '聚集记忆的时间-第一次.csv'
file = open(filename, 'w', encoding='utf-8', newline='')
csv_writer = csv.DictWriter(file, fieldnames=[ '评论昵称', '评论内容',  '评论日期', '评论时间', '地区', "点赞人数", ])
csv_writer.writeheader()  # 写入表头

# 4.抓取数据并清洗返回
# 翻页设置
page = 1
while True:    

    print(f'正在采集第{page}页数据!')  
    
    # 等待数据包加载    
    r = driver.listen.wait(timeout=20)    
  
    # 获取数据包数据    
    json_data = r.response.body    
  
    # 提取评论所在列表    
    commentsList = json_data['data']["comments"]    
  
    # 使用for循环提取评论信息    
    for comment in commentsList:        
        comment_info = {
           '评论昵称': comment["user"]['nickname'],            
           '评论内容': comment['content'],            
           '评论日期': comment['timeStr'],            
           '评论时间':comment['time'],            
           "地区":comment["ipLocation"]['location'],            
           '点赞人数': comment['likedCount'],
           }        # 在循环中保存每条记录        
       
        csv_writer.writerow(comment_info)    
     
     
     # 翻页操作    
    next_btn = driver.ele('下一页')    
    total_page = next_btn.prev().text   # 下一页前一个元素，即总页数，字符串    
    if next_btn:        
        if int(total_page) == page :   # page等于234时停止程序,total_page数字不会变化,利用这点跳出while死循环            
            break            
            print('下一页不可点击,循环结束,所有数据已写入。')        
    else:            
        next_btn.click()            
        page = page + 1            
        time.sleep(2) # 避免爬得太快,反防爬,time.sleep一定要放在最后

file.close()   # 关闭并保存数据
print("数据采集完成,已写入csv文档！")
