'''
对话系统
输入一个句子，跳转到下一个节点，
重听功能：
'''
import json
import pandas as pd
import re
from dataclasses import dataclass,field
from typing import List,Dict 

@dataclass
class Memory:
    avaliable_nodes:  List[str]= field(default_factory=list)  # 使用dict的默认工厂函数
    user_query:str= field(default="")
    response:str= field(default="")
    hit_node_id:int= field(default=-1)
    hit_score: float= field(default=0)
    missing_slot: str = field(default="")
    policy:str =field(default="")
    slot: Dict[str,List[str]]= field(default_factory=dict)  # 使用dict的默认工厂函数
    history: List[str]= field(default_factory=list)  # 历史对话数据

REPEAT_NODE_ID = 'scenario-clothes_node5'

class DialougeSystem(object):
    def __init__(self,scene_path,template_path) -> None:
        self.scene_path =scene_path
        self.template_path = template_path
        self.all_nodes_info={}
        self.slot_info = {}
        
        self.load()

    def load(self):
        # 节点信息 节点id:节点信息
        self.all_nodes_info={}
        self.load_scenario(self.scene_path)
        # 加载词槽模版 词槽名称： [query,value]
        self.slot_info = {}
        self.load_template(self.template_path)
    
    def load_scenario(self,file_name):
        parts = file_name.rsplit('/', 1)
        # 如果有分割结果，取最后一个元素，否则整个字符串
        result = parts[-1] if len(parts) > 1 else file_name
        scenario_name = result.replace(".json", "")
        # 加载场景脚本
        with open(file_name,"r",encoding="utf8") as f:
            for node_info in json.load(f):
                node_id = node_info["id"]
                node_id = scenario_name + "_"+node_id
                if "childnode" in node_info:
                    node_info["childnode"] = [scenario_name+"_"+child_node_id for child_node_id in node_info["childnode"]]
                self.all_nodes_info[node_id] = node_info
        return 
    
    def load_template(self,file_name):
        slot_template = pd.read_excel(file_name,sheet_name='Sheet1')
        for idx,row in slot_template.iterrows():
             self.slot_info[row["slot"]] = [row["query"], row["values"]]
        return 


    def run(self,user_query,memory):
        if len(memory.avaliable_nodes)==0:
            memory.avaliable_nodes.append("scenario-clothes_node1")
            memory.avaliable_nodes.append(REPEAT_NODE_ID)
        memory.user_query = user_query
        memory = self.nlu(memory) # 自然语言理解
        memory = self.ds(memory) # 对话管理
        memory = self.nlg(memory) #自然语言生成
        memory.history.append({"user_query":user_query,"response":memory.response})
        return memory

    def nlu(self,memory):
        # 自然语言理解 
        # 意图识别 + 槽位抽取
        memory = self.intent_recognition(memory)
        memory = self.slot_filling(memory)
        return memory

    def ds(self,user_memory):
        # 对话状态跟踪
        user_memory = self.dst(user_memory)
        # 策略管理
        user_memory = self.dpo(user_memory)
        return user_memory
    

    def dst(self,memory):
        # 检查命中节点对应位置的槽位信息是否齐备
        for slot in self.all_nodes_info[memory.hit_node_id].get('slot',[]):
            if memory.slot.get(slot,'') == '':
                # 词槽不满足
                memory.missing_slot = slot
                return memory
        memory.missing_slot = None
        return memory

    def dpo(self,memory):
        # 策略  如果缺少槽位，反问；如果不缺，回答
        if memory.hit_node_id == REPEAT_NODE_ID:
            memory.policy = 'repeat'
            return memory
        if memory.missing_slot is not None:
            slot = memory.missing_slot
            memory.policy = 'ask'
            # 留在本节点
            # avaliable_nodes 不需要变
            # memory.avaliable_nodes = memory.hit_node_id
        else:
            memory.policy = 'answer'
            # 开放子节点
            memory.avaliable_nodes = self.all_nodes_info[memory.hit_node_id].get("childnode", [])
        return memory
            
    def nlg(self,memory):
        # 根据policy 做出选择
        if memory.policy =='repeat':
            if(len(memory.history))==0:
                memory.response = "你好"
            else:
                memory.response = memory.history[-1].get("response","")
        elif memory.policy == 'ask':
            slot = memory.missing_slot
            ask_sentence,_ = self.slot_info[slot]
            memory.response = ask_sentence
        elif memory.policy == 'answer':
            response = self.all_nodes_info[memory.hit_node_id]['response']
            # response中的词槽替换
            response = self.replace_slot_in_response(memory,response)
            memory.response = response
        return memory

    def replace_slot_in_response(self,memory,response):
        slots = self.all_nodes_info[memory.hit_node_id].get('slot',[])
        for slot in slots:
            response = re.sub(slot,memory.slot.get(slot,""),response)
        return response

    def slot_filling(self,memory):
        # 判断当前是哪个节点，需要哪些slot,正则匹配用户话术中的词槽
        user_query = memory.user_query
        slot_list = self.all_nodes_info[memory.hit_node_id].get("slot",[])
        # self.slot_info[row["slot"]] = [row["query"], row["values"]]
        # return 
        for slot in slot_list:
            _,candidates = self.slot_info[slot]
            search_result = re.search(candidates,user_query)
            if search_result:
                memory.slot[slot] = search_result.group()
        return memory


    def intent_recognition(self,memory):
        # 意图识别 规则、分类任务，选出分值最高的节点.和用户输入句子分数最高的意图
        hit_score = -1 
        hit_node_id = None
        for node_id in memory.avaliable_nodes:
            node_info = self.all_nodes_info[node_id]
            score= self.cal_intent_score(memory,node_info)
            if score > hit_score:
                hit_score = score
                hit_node_id = node_id
        memory.hit_node_id = hit_node_id
        memory.hit_score = hit_score
        return memory
    
    def cal_intent_score(self,memory,node_info):
        #计算意图识别的分数 计算用户输入与当前节点的相似分数，当前节点可能是多意图的，取最高分
        user_query = memory.user_query
        intent_list = node_info["intent"]
        all_scores = []
        for intent in intent_list:
            score = self.sentence_similarity(user_query,intent)
            all_scores.append(score)
        return max(all_scores)
    
    def sentence_similarity(self,str1,str2):
        # jaccard距离
        score = len(set(str1) & set(str2))/len(set(str1)|set(str2))
        return score


if __name__ == "__main__":
    scene_path = './scenario/scenario-clothes.json'
    template_path = './scenario/slot_fitting_templet.xlsx'
    dialog_system = DialougeSystem(scene_path,template_path)
    print(dialog_system.all_nodes_info)
    print(dialog_system.slot_info)
    # memory={
    #     "avaliable_nodes":[],
    #     "user_query":"",
    #     "response":"",
    #     "hit_node_id":None,
    #     "hit_score":-1,
    #     'missing_slot':None,
    #     'policy':None,
    #     # slot 每一个词槽有哪些词
    # }
    memory = Memory()
    while True:
        user_query = input("请输入：")
        memory = dialog_system.run(user_query,memory)
        print(memory.response)
        print("--------------------")
        print(memory)
        print("========================")