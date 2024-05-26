import json
import os
import pandas
import re

'''
基于场景脚本的对话
memory中的字段
avaliable_node 当前可以访问的节点
hit_node_id 命中的节点id
hit_score 命中的节点分值
user_query 用户输入
system_response 系统回复
missing_slot 缺少的槽位
policy  当前策略  ask_slot 缺槽位反问 answer 回答 pardon重听

'''

class DialogSystem:
    def __init__(self) -> None:
        self.short_memory = {}
        self.long_memory = {}
        self.dialog_history = []
        
        self.load_data()
        
    def load_data(self):
        #加载场景脚本
        self.load_scenario()
        #加载槽位信息
        self.load_slot_fitting()
        
        
        
    def load_slot_fitting(self):
        self.slot_fitting = {} #槽位信息 槽位名称为key [query,values]为value 
        self.slot_fitting_df = pandas.read_excel('slot_fitting_templet.xlsx',sheet_name='Sheet1')
        for index,row in self.slot_fitting_df.iterrows():
            self.slot_fitting[row['slot']] = [row['query'],row['values']]
        print('槽位信息：',self.slot_fitting)
    def load_scenario(self):
        self.scenario_script = {} #id为key 节点信息为value
        
        file_path = os.path.join(os.path.dirname(__file__), 'scenario-买衣服.json')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            scenario_name = file_path.split(os.path.sep)[-1].split('.')[0]
            for node_info in json.load(f):
                node_id = node_info['id']
                node_id = scenario_name + '-' + node_id
                if 'childnode' in node_info:
                    node_info['childnode'] = [scenario_name + '-' + child_id for child_id in node_info['childnode']]
                self.scenario_script[node_id] = node_info
        
        print('脚本信息：', self.scenario_script)
    def respond(self, user_query: str) -> str:
        if len(self.short_memory.get('avaliable_node',[])) == 0:
            #对用户开放初始节点
            self.short_memory['avaliable_node'] = ['scenario-买衣服-node1']
        self.short_memory['user_query'] = user_query
        #加入简单的重听功能
        if self.pardon():
            self.short_memory['policy'] = 'pardon'
            self.nlg()
        else:


            self.nlu()
            self.dst()
            self.dpo()
            self.nlg()

        return self.short_memory['system_response']
        
    def nlu(self):
        '''意图识别，槽位抽取'''
        self.intent_recognition()
        self.slot_filling()
        
    def intent_recognition(self):
        '''意图识别,选出分值最高的节点'''
        hit_score = -1
        hit_node_id = None
        for node_id in self.short_memory['avaliable_node']:
            node_info = self.scenario_script[node_id]
            score = self.calc_intent_score(node_info)
            
            if score > hit_score:
                hit_score = score
                hit_node_id = node_id
        self.short_memory['hit_node_id'] = hit_node_id
        self.short_memory['hit_score'] = hit_score
        
    def calc_intent_score(self, node_info):
        '''计算节点的意图分值'''
        

        user_query  = self.short_memory['user_query'] 

        
        intent_list = node_info['intent']
        all_scores = []
        
        for intent in intent_list:
            score = self.sentence_similarity(user_query, intent)
            print('(',user_query,',',intent,')-->分值:',score)
            all_scores.append(score)
        return max(all_scores)
    def pardon(self):

        return self.short_memory.get('system_response',None) and self.short_memory['user_query'] in ['再说一遍','pardon','能再说一遍吗','没听清','重听']

    def sentence_similarity(self, sentence1, sentence2):
        '''使用jaccard距离'''

        score = len(set(sentence1) & set(sentence2))/len(set(sentence1) | set(sentence2))
        return score
            
    def slot_filling(self):
        '''槽位填充'''
        user_query = self.short_memory['user_query']
        slot_list = self.scenario_script[self.short_memory['hit_node_id']].get('slot',[])
        for slot in slot_list:
            _,candidates = self.slot_fitting[slot]
            search_result = re.search(candidates,user_query)
            if search_result:
                self.short_memory[slot] = search_result.group()
        
    def dst(self):
        '''检查命中节点对应的槽位信息是否齐备'''
        for slot in self.scenario_script[self.short_memory['hit_node_id']].get('slot',[]):
            if slot not in self.short_memory or self.short_memory[slot] == None: #如果缺槽位
                self.short_memory['missing_slot'] = slot
                return
        self.short_memory['missing_slot'] = None
    def dpo(self):
        '''如果缺槽位，反问
            如果槽位不缺，回答
        '''
        if self.short_memory['missing_slot']:
            self.short_memory['policy'] = 'ask_slot'
            #留在本节点 available_node不变
            self.short_memory['avaliable_node'] = [self.short_memory['hit_node_id']]
            
        else:
            self.short_memory['policy'] = 'answer'
            #开放子节点
            self.short_memory['avaliable_node'] = self.scenario_script[self.short_memory['hit_node_id']].get('childnode',[])
            
    def nlg(self):
        '''根据policy生成回复'''
        if self.short_memory['policy'] == 'ask_slot':
            missing_slot = self.short_memory['missing_slot']
            response,candidates = self.slot_fitting[missing_slot]
            self.short_memory['system_response'] = response
        elif self.short_memory['policy'] == 'answer':
            #根据命中节点找到response
            raw_reponse = self.scenario_script[self.short_memory['hit_node_id']]['response']
            #替换成memory中的槽位
            response = self.replace_slot_in_response(raw_reponse)
            self.short_memory['system_response'] = response
            self.short_memory.pop('policy')
        elif self.short_memory['policy'] == 'pardon':
            return
            
    def replace_slot_in_response(self, raw_reponse):
        for slot in self.scenario_script[self.short_memory['hit_node_id']].get('slot',[]):
            raw_reponse = re.sub(slot,self.short_memory[slot],raw_reponse)
        return raw_reponse
        
        
    
if __name__ == '__main__':
    dialog_system = DialogSystem()
    
    user_query = input("输入: ")
    
    
    while True:
        if user_query == "exit":
            break
        else:
            response = dialog_system.respond(user_query)
            print('===============')
            print(dialog_system.short_memory)
            print(response)
            user_query = input("输入: ")