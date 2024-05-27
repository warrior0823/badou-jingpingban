


import json
import re

import pandas





'''
    基于一个模板，做一个问答
'''

class DialogSystem:
    def __init__(self):
        #加载数据
        self.all_node_info = {} #以id 为key ， 节点信息为value
        self.slot_info = {}     #以key 为槽位名称，value 是[query ,values]
        self.load()

    def load(self):
        self.load_scenario('scenario-买衣服.json')
        self.load_slot('slot_fitting_templet.xlsx')

    def load_scenario(self, path):
        #加载场景
        scenario_name =  path.replace('.json' , '')
        with open(path, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                itemId = scenario_name + '_'+item['id']
                self.all_node_info[itemId] = item
                if 'childnode' in item :
                    item['childnode'] = [scenario_name+'_'+i for i in item['childnode']]
                self.all_node_info[itemId] = item
        print('self.all_node_info:',self.all_node_info)
        return

    def load_slot(self , path):
        #加载槽位
        df = pandas.read_excel(path)
        for  index , row in df.iterrows():
            self.slot_info[row['slot']] = [row['query'],row['values']]
        print('self.slot_info:',self.slot_info)
        return

    def run(self, user_input, memory):
        if len(memory) ==  0:
            memory["avaliable_nodes"] = ["scenario-买衣服_node1"]
        memory['user_input'] = user_input
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory

    def nlu(self, memory):
        #意图识别 + 语义槽填充
        memory = self.to_intention(memory)
        memory = self.to_slot_filling(memory)
        return memory

    def to_intention(self, memory):
        # 意图识别, 选出分值最高的节点
        hit_score = -1
        hit_node_id = None
        #循环可使用的节点
        for node_id  in memory['avaliable_nodes']:
            node_info = self.all_node_info[node_id]
            score = self.intention_score(memory , node_info)
            if score > hit_score:
                hit_score = score
                hit_node_id = node_id
        memory['hit_node_id'] = hit_node_id
        memory['hit_score'] = hit_score
        return memory


    def intention_score(self , memory , node_info):
        input_content = memory['user_input']
        node_list = node_info['intent']
        # 字符串匹配选分值最高的
        all_scores = []
        for node in node_list:
            score = self.sentence_similarity(input_content, node)
            all_scores.append(score)
        return max(all_scores)

    def sentence_similarity(self, sentence1, sentence2):
        # 计算两个句子的相似度
        return len(set(sentence1) & set(sentence2)) / len(set(sentence1) | set(sentence2))

    def to_slot_filling(self, memory):
        # 槽位抽取
        slot_list =  self.all_node_info[memory['hit_node_id']].get('slot' , [])
        for item_slot in slot_list:
            key , value = self.slot_info[item_slot]
            search_result =  re.search(value,memory['user_input'])
            if  search_result is not None:
                memory[item_slot] = search_result.group()
        return memory


    def dst(self, memory):
        #记录节点
        self.node_num(memory)
        # 检查命中节点对应的槽位信息是否齐备
        slot_list = self.all_node_info[memory['hit_node_id']].get('slot', [])

        for item_slot in slot_list:
            if item_slot not in memory or memory[item_slot] == '': #槽位缺失
                memory['missing_slot'] = item_slot
                return memory
        memory['missing_slot']= None
        return memory

    def node_num(self , memory):
        #新增一个参数 ，记录 hit_node_id
        if 'node_id_show_num' not in memory:
            memory['node_id_show_num'] =[memory['hit_node_id']]
        else:
            if  memory['hit_node_id'] != memory['node_id_show_num'][0]:
                memory['node_id_show_num'] =[ memory['hit_node_id']]
            else:
                memory['node_id_show_num'].append(memory['hit_node_id'])


    def dpo(self, memory):
        # 如果缺少槽位：反问；如果不缺槽位，回答
        if memory['missing_slot'] is not None:
            memory['policy'] = 'ask'
            # 留在本节点
            memory['avaliable_nodes'] = [memory['hit_node_id']]
        else:
            memory['policy']= 'answer'
            # 开放子节点
            memory['avaliable_nodes'] = self.all_node_info[memory['hit_node_id']].get('childnode', [])
        return memory

    def nlg(self, memory):
        # 根据policy做选择
        if memory['policy'] == 'ask':
            # 反问
            slot =  memory['missing_slot']
            key , value = self.slot_info[slot]
            memory['response'] = key

        elif memory['policy'] == 'answer':
            # 回答
            response = self.all_node_info[memory['hit_node_id']]['response']
            response = self.replace_slot_in_response(memory , response)
            memory['response'] = response
        return memory

    def replace_slot_in_response(self , memory , response):
        slot_list = self.all_node_info[memory['hit_node_id']].get('slot', [])
        for slot in slot_list:
            response = response.replace(slot, memory[slot])
        return response

if __name__ == '__main__':
    dialogSystem  = DialogSystem()
    memory = {}
    while True:
        user_input = input("输入：")
        result = dialogSystem.run(user_input, memory)
        print(memory["response"])
        print(memory)
        print("===============")
        if len(memory['node_id_show_num']) == 5:
            print("对话结束")
            break
        # if user_input != '重说':
        #     result = dialogSystem.run(user_input, memory)
        #     print(memory["response"])
        #     memory['retry'] = memory['response']
        #     print(memory)
        #     print("===============")
        # else:
        #     print(memory['retry'])
'''
    num : 2

'''