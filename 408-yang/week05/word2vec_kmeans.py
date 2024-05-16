# coding=utf-8
'''
    word2vec 进行kmeans 聚类
    对输入语料进行聚类分析
'''
from word2vec_train import load_word2vec_model
import jieba
import numpy as np
import math
from sklearn.cluster import KMeans
from collections import defaultdict


def load_sentences(input_path):
    sentence = set()
    with open(input_path,"r",encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            sentence.add(" ".join(jieba.lcut(line)))
    print("获取句子数量: ",len(sentence))
    return sentence


def sentence_to_vectors(sentences,model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector)
    return vectors

def euclidean_distance(vector1, vector2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(vector1, vector2)))

def cosine_distance(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cosine_distance = dot_product / (norm_vector1 * norm_vector2)
    return cosine_distance

def main(model_path,input_path):
    model = load_word2vec_model(model_path)
    # 加载句子，并进行分词
    sentences = load_sentences(input_path)
    # 将句子分词之后，从训练好的模型中加载每个词的词向量，以此作为句子的词向量
    vectors = sentence_to_vectors(sentences,model)

    # 进行聚类，
    # 首先要确定k值，随机选择k个质心，迭代循环进行聚类，借助于kmeans算法
    k = int(math.sqrt(len(sentences)))

    print('聚类数目为:' ,k)
    kmeans = KMeans(k).fit(vectors) #进行聚类计算
    
    labels = kmeans.labels_ 
    vector_label_dict = defaultdict(list)
    # 方法一，直接计算欧式距离
    for vector,label in zip(vectors,labels):
        k_center = kmeans.cluster_centers_[label]
        vector_label_dict[label].append(euclidean_distance(vector,k_center))

    distance = []
    for i in range(k):
        label_items = vector_label_dict[i]
        distance.append(sum(label_items) / len(label_items))
    
    #  找到所有最大值及其索引
    max_value = max(distance)
    max_indices = [index for index, value in enumerate(distance) if value == max_value]
    print('distance: ',distance)
    print('distance max index : ',max_indices)

    centroids = kmeans.cluster_centers_
    
    # 计算类内平均距离 方法二
    within_cluster_distances = []
    for i in range(k):
        # 计算类 i 中所有点到类中心的距离
        cluster_points = [vector_label_dict[label]]
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
        # 计算平均距离
        mean_distance = distances.mean()
        within_cluster_distances.append(mean_distance)

    # 打印类内平均距离
    print("类内平均距离：", within_cluster_distances)
    max_value = max(within_cluster_distances)
    max_indices = [index for index, value in enumerate(within_cluster_distances) if value == max_value]
    print('distance max index : ',)


if __name__ == "__main__" :
    model_path = './model/word2vec.pt'
    input_path = './data/titles.txt'
    main(model_path=model_path,input_path=input_path)