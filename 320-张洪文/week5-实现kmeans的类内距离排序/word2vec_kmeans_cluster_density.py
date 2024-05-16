import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

"""
基于词向量和Kmeans进行文字标题聚类
"""
# 向量余弦距离
def cosine_distance(vec1, vec2):
    vec1 = vec1 / np.sqrt(np.sum(np.square(vec1)))  # A/|A|, 平方和再开平方
    vec2 = vec2 / np.sqrt(np.sum(np.square(vec2)))  # B/|B|
    return 1 - np.sum(vec1 * vec2)

# 欧式距离
def eculidean_distance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2), axis=1))

# 加载文本
def load_sentences(path):
    titles = []
    word_sentences = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            title = line.strip()
            sentence = " ".join(jieba.cut(title))
            if sentence not in word_sentences:
                word_sentences.append(sentence)
                titles.append(title)
    print("文本加载完成，原有标题: %d，分词后: %d" % (len(titles), len(word_sentences)))
    return titles, word_sentences

# 文本向量化
def sentences_to_vector(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split(" ")
        vector = np.zeros(model.vector_size)
        # 词向量加和求平均，作为句向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = Word2Vec.load("../model.w2v")  # 加载词向量模型
    titles, sentences = load_sentences("../titles.txt")  # 加载标题
    sentence_vectors = sentences_to_vector(sentences, model)  # 标题向量化

    # 进行Kmeans聚类
    n_clusters = int(np.sqrt(len(sentences)))  # 聚类数量
    print("当前聚类数量：%d" % n_clusters)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(sentence_vectors)  # Kmeans模型拟合：词向量

    # 计算kmeans中的类内平均距离
    cluster_centers = kmeans.cluster_centers_  # 聚类中心点
    labels = kmeans.labels_  # 聚类标签
    average_distances = defaultdict(float)   # 存储每个聚类的平均距离
    # 遍历每一个聚类
    for cluster in range(n_clusters):
        vector = sentence_vectors[labels == cluster]  # 获取当前聚类对应的句向量
        cluster_center = cluster_centers[cluster]     # 当前聚类中心点
        # 欧式距离计算: 一维数组，计算完成后数组中的每个元素表示一个点到中心点的欧式距离
        distance = eculidean_distance(vector, cluster_center)
        average_distance = np.mean(distance)  # 类内平均距离
        average_distances[cluster] = average_distance
    # 排序
    average_distance_sort = sorted(average_distances.items(), key=lambda x: x[1])
    for item in average_distance_sort:
        print("%d 类内平均距离: %s" % (item[0], item[1]))

    while True:
        query = input("请输入:")
        query = " ".join(jieba.cut(query))
        query_vec = sentences_to_vector([query], model)
        score = defaultdict(float)
        # 和每一个句向量计算余弦距离
        for index, vec in enumerate(sentence_vectors):
            dis = cosine_distance(query_vec[0], vec)
            score[index] = dis
        score_sort = sorted(score.items(), key=lambda x: x[1])
        print("查询结果:", titles[score_sort[0][0]])
        for index, d in score_sort[:3]:
            print("%s   余弦距离: %s" % (titles[index], d))


if __name__ == '__main__':
    main()
