'''
文本矩阵化：要求采用词袋模型且是词级别的矩阵化
https://blog.csdn.net/zh11403070219/article/details/88169237
'''
import jieba
import pandas as pd
import tensorflow as tf
from collections import Counter
# from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
 
 
# 读取停用词
def read_stopword(filename):
    stopword = []
    fp = open(filename, 'r',encoding='unicode_escape')
    for line in fp.readlines():
        stopword.append(line.replace('\n', ''))
    fp.close()
    return stopword
 
 
# 切分数据，并删除停用词
def cut_data(data, stopword):
    words = []
    for content in data['content']:
        word = list(jieba.cut(content))
        for w in list(set(word) & set(stopword)):
            while w in word:
                word.remove(w)
        words.append(' '.join(word))
    data['content'] = words
    return data
 
 
# 获取单词列表
def word_list(data):
    all_word = []
    for word in data['content']:
        all_word.extend(word)
    all_word = list(set(all_word))
    return all_word
 
 
# 计算文本向量
def text_vec(data):
    count_vec = CountVectorizer(max_features=300, min_df=2)
    count_vec.fit_transform(data['content'])
    fea_vec = count_vec.transform(data['content']).toarray()
    return fea_vec
 
 
if __name__ == '__main__':
    data = pd.read_csv('/home/renlei/dev/PyProjects/cnews/cnews.test.txt', names=['title', 'content'], sep='\t')  # (10000, 2)
    data = data.head(50)
 
    stopword = read_stopword('/home/renlei/dev/PyProjects/cnews/stopwords.txt')
    data = cut_data(data, stopword)
 
    fea_vec = text_vec(data)
    print(fea_vec)
 