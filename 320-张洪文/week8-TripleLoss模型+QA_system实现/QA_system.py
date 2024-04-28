import json
import os.path
import jieba
import numpy as np
from collections import defaultdict
import torch
import itertools
from gensim.models import Word2Vec
from SimilarityFunction import edit_distance, jaccard_distance
from bm25 import BM25
from Text_Matching_Presentation_TripletLoss.config import Config
from Text_Matching_Presentation_TripletLoss.model import PresentationModel
from Text_Matching_Presentation_TripletLoss.loader import load_data, load_schema

class Prediction:
    def __init__(self, config, weights_path):
        self.config = config
        self.train_data = load_data(config["train_path"], config)  # 加载知识库
        self.model = PresentationModel(config)  # 加载模型
        self.model.load_state_dict(torch.load(weights_path))  # 加载权重
        self.vocab = self.train_data.dataset.vocab
        self.schema = {v: k for k, v in load_schema(config["schema_path"]).items()}
        self.index_questions = self.train_data.dataset.index_questions  # k:知识库中每一个问题的索引 v:问题在知识库中的标准问label
        self.knwb_vectors = None  # faq库向量化
        self.knwb_to_vector()  # 知识库问题向量初始化

    # 知识库问题向量化，为匹配做准备
    def knwb_to_vector(self):
        questions_list = list(itertools.chain(*self.train_data.dataset.knwb.values()))
        with torch.no_grad():  # 不计算梯度
            question_matrix = torch.stack(questions_list, dim=0)  # (n, max_len)  n 为问题总数
            # question_matrix = question_matrix.to(self.config["device"])
            self_knwb_vectors = self.model(question_matrix)  # shape = (n, hidden_size)
            # 向量归一化：v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self_knwb_vectors, dim=-1)

    def encode_sentence(self, text):
        encode_text = []
        if "chars" in self.config["vocab_path"]:  # 以字作为分词
            for char in text:
                encode_text.append(self.vocab.get(char, self.vocab["[UNK]"]))
        else:  # 否则以词作为分词
            for word in jieba.cut(text):
                encode_text.append(self.vocab.get(word, self.vocab["[UNK]"]))
        # 截断补全字符
        if len(encode_text) < self.config["max_len"]:
            encode_text.extend([0] * (self.config["max_len"] - len(encode_text)))
        else:
            encode_text = encode_text[:self.config["max_len"]]
        return encode_text

    def predict(self, question, top=3):
        input_ids = self.encode_sentence(question)
        input_ids = torch.LongTensor([input_ids])
        self.model.eval()  # 训练模式
        with torch.no_grad():  # 不计算梯度
            pred = self.model(input_ids)  # 问题向量化
        pred = torch.nn.functional.normalize(pred, dim=-1)  # 向量归一化
        # 计算预测结果: 通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
        # pred shape=(hidden_size,) knwb_vector shape=(n, hidden_size)
        results = torch.mm(pred.unsqueeze(0), self.knwb_vectors.T).squeeze()
        results = list(results)  # (n,)

        # 根据子问题的概率计算每个标准问的概率
        schema_prob = []
        index = 0
        for k, v in self.train_data.dataset.knwb.items():
            prob = np.mean(results[index: index + len(v)])
            schema_prob.append([self.schema[k], float(prob)])
            index += len(v)
        schema_prob = sorted(schema_prob, key=lambda x: x[1], reverse=True)
        return schema_prob[:top]

class QASystem:
    def __init__(self, know_base_path, algo, weights_path=None):
        """
        :param know_base_path: 知识库文件路径
        :param algo: 指定算法
        """
        # 加载对应算法
        self.target_questions = {}  # 保存标准问答对
        self.load_knowledge_base(know_base_path)  # 加载知识库
        self.algo = algo
        if algo == 'bm25':
            self.load_bm25()
        elif algo == "word2vec":
            self.load_word2vec()
        elif algo == "language_model":
            self.model = Prediction(Config, weights_path)

    def load_knowledge_base(self, know_base_path):
        with open(know_base_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = json.loads(line.strip())
                questions, target = text['questions'], text['target']
                self.target_questions[target] = questions

    def load_bm25(self):
        self.corpus = defaultdict(list)
        for target, questions in self.target_questions.items():
            for question in questions:
                self.corpus[target] += jieba.lcut(question)
        self.bm25 = BM25(self.corpus)

    def sentence_vector(self, sentence):
        vector = np.zeros(self.w2v_model.vector_size)
        words = jieba.lcut(sentence)
        # 所有词的向量加和求平均，作为句子向量
        count = 0
        for word in words:
            if word in self.w2v_model.wv:
                count += 1
                vector += self.w2v_model.wv[word]  # 向量累加
        vector = vector / count  # 求平均
        # 向量归一化，方便做cos距离
        vector = vector / np.sqrt(np.sum(np.square(vector)))
        vector2 = vector / np.linalg.norm(vector)
        return vector

    def load_word2vec(self):
        """
        加载预训练的词向量模型
        词向量的训练需要一定时间，我们可以读取训练好的模型 (注意如果数据集更换了，应当重新训练)
        #然，也可以收集一份大量的通用的语料，训练一个通用词向量模型。一般少量数据来训练效果不会太理想
        """
        if os.path.isfile('data/model.w2v'):
            self.w2v_model = Word2Vec.load('model.w2v')
        else:
            # 准备训练语料
            corpus = []
            for questions in self.target_questions.values():
                for question in questions:
                    corpus.append(jieba.lcut(question))
            # 训练模型：调用第三方库模型
            self.w2v_model = Word2Vec(corpus, vector_size=100, min_count=1)
            self.w2v_model.save('model.w2v')  # 保存模型

        # 借助词向量模型，将知识库中的问题向量化
        self.target_vectors = defaultdict(list)
        for target, questions in self.target_questions.items():
            for question in questions:
                self.target_vectors[target].append(self.sentence_vector(question))
            np.array(self.target_vectors[target])

    # 接收用户问题，返回最相似的问题集
    def query(self, user_query):
        results = []
        if self.algo == "edit_distance":
            for target, questions in self.target_questions.items():
                scores = [edit_distance(user_query, question) for question in questions]
                score = max(scores)
                results.append([target, score])
        elif self.algo == "jaccard_distance":
            for target, questions in self.target_questions.items():
                scores = [jaccard_distance(user_query, question) for question in questions]
                score = max(scores)
                results.append([target, score])
        elif self.algo == "bm25":
            words = jieba.lcut(user_query)
            results = self.bm25.get_scores(words)
        elif self.algo == "word2vec":
            query_vector = self.sentence_vector(user_query)
            for target, vectors in self.target_vectors.items():
                cos = query_vector.dot(np.array(vectors).transpose())
                results.append([target, np.mean(cos)])
        elif self.algo == "language_model":
            return self.model.predict(user_query)
        else:
            assert "没有找到对应算法！！！"
        sort_results = sorted(results, key=lambda x: x[1], reverse=True)
        return sort_results[:3]  # 返回排名前三的序列


if __name__ == '__main__':
    # qas = QASystem("data/train.json", "bm25")
    # qas = QASystem("data/train.json", "edit_distance")
    # qas = QASystem("data/train.json", "jaccard_distance")
    weight_path = "./Text_Matching_Presentation_TripletLoss/model_30.pth"
    qas = QASystem("data/train.json", "language_model", weight_path)

    # que = "我想要重置一下固话密码"
    # result = qas.query(que, weight_path)
    # print(result)
    while True:
        question = input("请输入问题：")
        res = qas.query(question)
        print("命中问题：", res)
        print("-----------")

