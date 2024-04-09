import math
from collections import Counter

import numpy as np
import jieba
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

if __name__ == '__main__':
    book_list = [
        '白马啸西风', '碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣', '书剑恩仇录',
        '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑']
    content = ''
    for book_name in book_list:
        f = open("中文语料" + "/" + book_name + ".txt", "r", encoding='gbk', errors='ignore')
        book = f.read()
        content += book
    # 读取书籍文件
    content = content.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com\n', '')
    f = open("cn_stopwords.txt", "r", encoding='utf-8', errors='ignore')
    unprocessed_stopwords = f.readlines()
    stopwords = []
    for stopword in unprocessed_stopwords:
        stopwords.append(stopword.rstrip(stopword[-1]))
    effective_words = []
    for word in jieba.cut(content):
        if word not in stopwords and not word.isspace():
            effective_words.append(word)
    # 去除无用词
    counter = Counter(effective_words)
    sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    num = np.array([item[1] for item in sorted_counter])
    rank = np.array(range(1, len(num) + 1))

    # 统计词频
    plt.plot(np.log(num), np.log(rank))
    print(np.corrcoef(np.log(num), np.log(rank)))
    plt.xlabel('log(rank)')
    plt.ylabel('log(num)')
    plt.savefig('zipf\'s law.jpg')
