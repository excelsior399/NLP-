import jieba
import re
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
from gensim import corpora, models



def labelize(corpus_dict, num_paragraphs, k_value, type=0):
    result = []
    total_paragraphs_count = sum(len(corpus_dict[novel]) for novel in corpus_dict)
    paragraph_counts = {novel: len(paragraphs) for novel, paragraphs in corpus_dict.items()}
    for _ in range(num_paragraphs):
        novel = random.choices(list(corpus_dict.keys()),
                               weights=[count / total_paragraphs_count for count in paragraph_counts.values()], k=1)[0]
        paragraphs = corpus_dict[novel]
        paragraphs = re.split(r'\n\u3000\u3000', paragraphs)
        tokens = []
        while len(tokens) < k_value:
            paragraph = random.choice(paragraphs)
            if type == 0:
                tokens += list(paragraph)
            else:
                tokens += list(jieba.cut(paragraph))
        tokens = tokens[:k_value]
        result.append((tokens, novel))
    return result


def LDA(processed_data, num_topics=10):
    X = [item[0] for item in processed_data]  # 段落文本列表
    y = [item[1] for item in processed_data]  # 段落所属小说标签列表

    dictionary = corpora.Dictionary(X)
    lda_corpus_train = [dictionary.doc2bow(tmp_doc) for tmp_doc in X]
    lda = models.LdaModel(corpus=lda_corpus_train, id2word=dictionary, num_topics=num_topics)

    train_topic_distribution = lda.get_document_topics(lda_corpus_train)
    X_lda = np.zeros((len(X), num_topics))
    for i in range(len(train_topic_distribution)):
        tmp_topic_distribution = train_topic_distribution[i]
        for j in range(len(tmp_topic_distribution)):
            X_lda[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]

    classifier = RandomForestClassifier()
    scores = cross_val_score(classifier, X_lda, y, cv=10)
    print(scores.mean())

if __name__ == '__main__':
    book_list = [
        '白马啸西风', '碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣', '书剑恩仇录',
        '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑']
    dict = {}
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
        for stopword in stopwords:
            book.replace(stopword, '')
        dict[book_name] = book
    num_paragraphs = 1000
    num_topics = 100
    k_values = [20, 100, 500, 1000, 3000]
    k_value = k_values[4]
    processed_data = labelize(dict, num_paragraphs, k_value)
    LDA(processed_data, num_topics)