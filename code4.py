import jieba
from gensim.test.utils import get_tmpfile
from gensim.models import word2vec


if __name__ == '__main__':
    book_list = ['笑傲江湖']
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
        cleaned_text = ''
        for word in jieba.cut(book):
            if word not in stopwords and not word.isspace():
                effective_words.append(word)
                cleaned_text += word + ' '
        # 去除无用词
    with open('./cleaned_data.txt', 'w', encoding="utf-8") as f2:
        f2.write(cleaned_text)

    sentences = word2vec.LineSentence('./cleaned_data.txt')
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=5, vector_size=200, sg=1, epochs=50)
    # model.save("word2vec.model")
    model = word2vec.Word2Vec.load("word2vec.model")

    req_count = 10
    for key in model.wv.similar_by_word('东方不败', topn=100):
        print(key[0], key[1])
