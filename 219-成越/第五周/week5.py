import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 无修改内容
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


# 无修改内容
def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


# 无修改内容
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


# 计算类内平均距离
def compute_average_distance(vectors, labels):
    label_vectors = defaultdict(list)
    for vector, label in zip(vectors, labels):
        label_vectors[label].append(vector)

    average_distances = {}
    for label, vectors in label_vectors.items():
        centroid = np.mean(vectors, axis=0)
        distances = [np.linalg.norm(vector - centroid) for vector in vectors]
        average_distances[label] = np.mean(distances)

    return average_distances


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = 60  # 设定较多数量的聚类类别
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    average_distances = compute_average_distance(vectors, kmeans.labels_)
    sorted_labels = sorted(average_distances, key=average_distances.get)

    num_classes_to_keep = 5
    for label in sorted_labels[:num_classes_to_keep]:
        print(f"Cluster {label} (Average Distance: {average_distances[label]:.4f}):")
        for sentence in [s for s, l in zip(sentences, kmeans.labels_) if l == label][:10]:
            print(sentence.replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()