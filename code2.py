import math
from collections import Counter

import numpy as np
import jieba
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']



def entropy(effective_words):
    counter = Counter(list(effective_words))
    total_length = len(effective_words)
    word_entropy = 0
    for ch, num in counter.items():
        prob = num / total_length
        word_entropy -= prob * math.log2(prob)
    return word_entropy

def word_entropy(effective_words):
    counter = Counter(effective_words)
    total_length = len(effective_words)
    word_entropy = 0
    for ch, num in counter.items():
        prob = num / total_length
        word_entropy -= prob * math.log2(prob)
    return word_entropy

if __name__ == '__main__':
    book_list = [
        '白马啸西风', '碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣', '书剑恩仇录',
        '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑']
    content = ''
    for book_name in book_list:
        f = open("中文语料" + "/" + book_name + ".txt", "r", encoding='gbk', errors='ignore')
        book = f.read()
        # 读取书籍文件
        book = book.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com\n', '')
        f = open("cn_stopwords.txt", "r", encoding='utf-8', errors='ignore')
        unprocessed_stopwords = f.readlines()
        stopwords = []
        for stopword in unprocessed_stopwords:
            stopwords.append(stopword.rstrip(stopword[-1]))
        effective_words = []
        for word in jieba.cut(book):
            if word not in stopwords and not word.isspace():
                effective_words.append(word)
        # 去除无用词
        # print(word_entropy(effective_words))
        cleaned_text = ''
        for word in effective_words:
            cleaned_text += word
        print(entropy(cleaned_text))

