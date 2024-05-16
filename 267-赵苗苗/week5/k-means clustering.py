"""
加载一个训练好的Word2Vec词向量模型。
从一个文本文件中加载句子。
将加载的句子转换为向量。
使用KMeans算法对句子向量进行聚类。
计算每个聚类的类内平均距离，并按照距离从大到小排序。
输出每个聚类的标签和一些代表性的句子。
"""

#聚类采用Kmeans算法，根据训练好的词向量模型进行聚类
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#根据模型文件路径加载训练好的模型
def load_word2vec_model(path):
    model=Word2Vec.load(path)
    return model

#从文件中加载句子并处理
def load_sentence(path):
    sentences=set()
    with open(path,encoding="utf8") as f:
        for line in f:
            sentence=line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：",len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences,model):
    vectors=[]
    for sentence in sentences:
        words=sentence.split()
        vector=np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector+=model.wv[word]
            except KeyError:
                vector+=np.zeros(model.vector_size)
        vectors.append(vector/len(words))
    return np.array(vectors)

#计算向量余弦距离
def cosine_distance(vec1,vec2):
    vec1=vec1/np.sqrt(np.sum(np.square(vec1)))
    vec2=vec2/np.sqrt(np.sum(np.square(vec2)))
    return np.sum(vec1*vec2)

#欧式距离
def eculid_distance(vec1, vec2):
    return np.sqrt((np.sum(np.square(vec1 - vec2))))

def main():
    model=load_word2vec_model("model.w2v")  #加载词向量模型
    sentences=load_sentence("titles.txt")     #加载所有标题
    vectors=sentences_to_vectors(sentences,model)  #将所有标题向量化

    n_clusters=int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：",n_clusters)  
    kmeans=KMeans(n_clusters)   #定义一个kmeans计算类
    kmeans.fit(vectors)   #进行聚类计算

    sentence_label_dict=defaultdict(list)
    for sentence,label in zip(sentences,kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)    #同标签的放到一起

    #计算类内距离
    density_dict=defaultdict(list)
    #遍历每个样本向量及其所属的类别标签
    for vector_index,label in enumerate(kmeans.labels_):  
        #获取当前样本向量和其所属类别的中心点
        vector=vectors[vector_index]   
        center=kmeans.cluster_centers_[label]
        #计算当前样本向量到其所属类别中心点的余弦距离
        distance=cosine_distance(vector,center)   
        density_dict[label].append(distance)
    # 计算每个类别的平均类内距离
    for label,distance_list in density_dict.items():
        density_dict[label]=np.mean(distance_list)
    # 按照平均类内距离对类别进行排序
    density_order=sorted(density_dict.items(),key=lambda x:x[1],reverse=True)

    #按照余弦距离顺序输出
    for label,distance_avg in density_order:
        print("cluster%s,avg distance%f:"% (label,distance_avg))
        sentences=sentence_label_dict[label]
        for i in range(min(10,len(sentences))):
            print(sentences[i].replace(' ',' '))
            print("--------")

if __name__ == '__main__':
    main()
