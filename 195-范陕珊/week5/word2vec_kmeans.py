#!/usr/bin/env python3  
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def cal_cos_distance(a, b):
    # Calculate the dot product of the vectors
    dot_product = np.dot(a, b)

    # Calculate the lengths of each vector
    a_length = np.linalg.norm(a)
    b_length = np.linalg.norm(b)

    # Calculate the cosine similarity using the formula
    return dot_product / (a_length * b_length)


def cal_distances(center, data):
    dis = 0
    for d in data:
        _dis_eu = np.linalg.norm(center - d)
        _dis_cos = cal_cos_distance(center, d)
        _dis = _dis_eu + _dis_cos
        print(f"dis{_dis} {center, d}")
        dis += _dis
    print(f"dis{dis}")
    return dis / len(data)


def get_sentencs_vectors_kmeans(sentences, vectors):
    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算
    vector_label_dict = defaultdict(list)
    sentences_label_dict = defaultdict(list)
    for vector, sentence, label in zip(vectors, sentences, kmeans.labels_):  # 取出句子和标签
        vector_label_dict[label].append(vector)  # 同标签的放到一起
        sentences_label_dict[label].append(sentence)  # 同标签的放到一起

    for label, sentences in sentences_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")
    class_dis_map = {}
    for idx, center in enumerate(kmeans.cluster_centers_):
        print(idx, center)
        label_vectors = vector_label_dict[idx]
        mean_dis = cal_distances(center, label_vectors)
        class_dis_map[mean_dis] = idx

    class_dis_sorted = sorted(class_dis_map.items())
    for idx, (dis, label) in enumerate(class_dis_sorted[::3]):
        sentences = sentences_label_dict[label]
        print(f"idx{idx},dis:{dis},label:{label},sentences:{sentences}")
    return sentences_label_dict, vector_label_dict, kmeans, class_dis_sorted


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    sentences = list(sentences)
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化
    sentences_label_dict, vector_label_dict, kmeans, class_dis_sorted \
        = get_sentencs_vectors_kmeans(sentences, vectors)

    print(f"class_dis_sorted{class_dis_sorted}")
    delete_center_idx = class_dis_sorted[-10:]
    print(f"delete_center_idx{delete_center_idx}")
    delete_all_sentences = []
    for _, delete_idx in delete_center_idx:
        print(len(sentences_label_dict[delete_idx]))
        delete_all_sentences.extend(sentences_label_dict[delete_idx])
    print(len(delete_all_sentences), vectors, type(vectors), type(delete_all_sentences))
    delete_all_sentences_idx = [sentences.index(sentence) for sentence in delete_all_sentences]
    print(f"delete_all_sentences_idx{delete_all_sentences_idx}")
    new_vectors = [v for idx, v in enumerate(vectors) if idx not in delete_all_sentences_idx]
    new_vectors = np.array(new_vectors)
    new_sentence = [s for idx, s in enumerate(sentences) if idx not in delete_all_sentences_idx]

    print(f"new_vectors{new_vectors.shape, len(delete_all_sentences)}")
    get_sentencs_vectors_kmeans(new_sentence, new_vectors)


if __name__ == "__main__":
    main()
