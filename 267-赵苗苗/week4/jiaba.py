import jieba

#词典，每个词后方存储的是其词频
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#构建中文分词的有向无环图
def calc_dag(sentence):
    DAG={}  #用来存储DAG有向无环图
    N=len(sentence)
    for k in range(N):   #遍历每个句子，对每个字符进行处理
        tmplist=[]       #存储与当前位置k对应的可能的词或词组的结束位置
        i=k
        frag=sentence[k]
        while i<N:
            if frag in Dict:
                tmplist.append(i)
            i+=1
            frag=sentence[k:i+1]
        if not tmplist:
            tmplist.append(k)
        DAG[k]=tmplist
    return DAG

#将DAG中的信息解码（还原）出来，用文本展示出所有的切分方式
class DAGDecode:
    #通过两个队列来实现
    def __init__(self, sentence):
        self.sentence = sentence
        self.DAG=calc_dag(sentence)
        self.length=len(sentence)
        self.unfinish_path=[[]]  #保存带解码序列的队列
        self.finish_path=[]      #保存解码完成的序列的队列
    
    #对已经进行部分解码的路径进行扩展，以完成整个句子的解码过程
    def decode_next(self,path):
        path_length=len("".join(path))
        if path_length==self.length: #已完成解码
            self.finish_path.append(path)  #直接加入
            return
        candidates=self.DAG[path_length]  #未完成解码
        new_paths=[]
        for candidate in candidates:
            new_paths.append(path+[self.sentence[path_length:candidate+1]])
        self.unfinish_path+=new_paths  #放入带解码队列
        return
    #递归调用序列解码过程
    def decode(self):
        while self.unfinish_path!=[]:
            path=self.unfinish_path.pop()  #从待解码队列中取出一个序列
            self.decode_next(path)  #使用该序列进行解码


sentence = "经常有意见分歧"
dd=DAGDecode(sentence)
dd.decode()
print(calc_dag(sentence))  #输出可能结束位置
print(dd.finish_path)  #输出全切分结果