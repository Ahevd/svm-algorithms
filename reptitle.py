#!/usr/bin/env python
# encoding: utf-8
import requests
import json
import re
import os
import time

'''
@author: zhangyangyang
@software: Pycharm
@file: reptitle.py
@time: 2019/4/18 19:13
@desc:爬去豆瓣电影网数据
'''

# 电影类型20种
movie_type = {'剧情', '喜剧', '动作', '爱情', '科幻', '动画', '悬疑', '惊悚', '恐怖', '犯罪',
              '音乐', '歌舞', '传记', '历史', '战争', '西部', '奇幻', '冒险', '灾难', '武侠'}


# 创建文件夹
def create_mkdir(mkdir_path):
    is_exists = os.path.exists(mkdir_path)
    if not is_exists:
        os.makedirs(mkdir_path)
        return True
    else:
        return False


def save_file(path, result):
    with open(path, 'a+', encoding='utf-8') as file:
        file.write(result)


# 1、保存各类电影的名称、url
def save_movie_url():
    for mt in movie_type:
        print(mt)
        save_path = 'E://python-data//' + mt + '.txt'
        file = open(save_path, "w", encoding='utf-8')
        movie_num = 10
        for num in range(0, 1):
            url = 'https://movie.douban.com/j/new_search_subjects?sort=U&range=0,10&tags=电影&start=' + str(
                movie_num) + '&genres=' + mt
            result = requests.get(url)
            data = result.json()['data']
            for item in data:
                item = item['title'] + '####' + item['url'] + '\r\n'
                file.write(item)
            movie_num = movie_num + 10
            print(movie_num)
            time.sleep(4)


# 2、根据电影url，保存电影简介
def save_content(movie_url, file_path):
    r = requests.get(movie_url)
    res_tr = r'<div class="indent" id="link-report">(.*?)</div>'
    m_tr = re.findall(res_tr, r.text, re.S | re.M)
    if len(m_tr) > 0:
        movie_desc = str(m_tr[0]).strip().replace('<br />', '').replace(' ', '').replace('\n', '')  # 简介去掉空格换行
        # 以电影名为txt文件名
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(movie_desc)
    else:
        with open('E://python-data//错误.txt', 'w', encoding='utf-8') as file1:
            file1.write(movie_url)


if __name__ == "__main__":
    # 1、获取电影名、简介URL
    save_movie_url()
    # 2、创建保存电影简介的文件夹
    # for mkdir_name in movie_type:
    #     create_mkdir('E://python-data//' + mkdir_name)  # 创建文件夹
    # 3、 读取txt文件，获得URL
    # for file_path in movie_type:
    #     with open('E://python-data4//' + file_path + '.txt', 'r', encoding='utf-8') as file:
    #         movie_list = file.readlines()
    #         for i in range(len(movie_list)):
    #             movie = movie_list[i]
    #             if movie == '\n':
    #                 continue
    #             movie = movie.rstrip('\n')
    #             movie_items = movie.split('####')
    #             movie_name = movie_items[0]
    #             movie_url = movie_items[1]
    #             save_path = 'E://python-data4//' + file_path + '//' + movie_name + '.txt'
    #             save_content(movie_url, save_path)
    #             print(movie_name)
    #             # 防止被封IP，延迟4秒
    #             time.sleep(4)
