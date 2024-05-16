import math
import re
import json
import jieba
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
# 实现Kmeans的类内距离排序

# 训练词向量
def train_word2vec_model(corpus, dim):
    sentences = []
    with open(corpus,encoding="utf8") as f :
        for line in f:
            sentences.append(jieba.lcut(line))
        model=Word2Vec(sentences,vector_size=dim,sg=1)
        model.save("model1.w2v")
    return model

# 加载需要分类的句子
def load_sentence():
    titles=set()
    with open("titles.txt",encoding="utf8") as f:
        for line in f:
            s = line.strip()  # 去掉左右两边多余的空格
            titles.add(" ".join(jieba.cut(s)))
    print("获取句子数量：", len(titles))
    return titles

# 将需要分类的文本转化成向量
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)  # 与子维度一样的0向量
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

# 计算两个向量的距离
def eculid_distance(vec1,vec2):
    return np.sqrt(np.sum(np.square(vec1-vec2)))

def main():

    model=train_word2vec_model("corpus.txt", 200)  # 第1步：训练词向量
    sentences=load_sentence()                                  # 第2步：加载需要分类的句子
    word_vector=sentences_to_vectors(sentences,model)          # 第3步：将需要分类的文本转化成向量
    # 第4步：用KMeans训练转化成的向量
    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters,n_init="auto")       # 定义一个kmeans计算类
    kmeans.fit(word_vector)           # 进行聚类训练
    sentence_label_dict = defaultdict(list)  # 字典
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)   # 把相同标签的句子放到一起如{key:[1,2,x]……
    # print(sentence_label_dict)

    density_dict = defaultdict(list)  # 距离字典
    for vector_index, label in enumerate(kmeans.labels_):
        center = kmeans.cluster_centers_[label]  # 该句子归属类的中心向量
        vector = word_vector[vector_index]  # 该句子的向量
        distance = eculid_distance(center, vector)
        density_dict[label].append(distance)
    for label, val in density_dict.items():
        density_dict[label] = np.mean(val)
    # print(density_dict)
    density_order = sorted(density_dict.items(), key=lambda x: x[1])
    print(density_order)

    m=0  # 打印距离最近的前5类
    for label, distance_avg in density_order:
        if m >4:
            break
        print("========================================================================")
        print("cluster %s,avg distance %f " % (label, distance_avg))
        sentence = sentence_label_dict[label]
        for i in sentence:
            print(i)
        m=m+1

if __name__ == "__main__":
    main()


