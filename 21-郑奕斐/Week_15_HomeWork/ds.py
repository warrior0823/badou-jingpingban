"""
基于场景对话的脚本

history中的关键字段
user_query 用户当前输入
missing_slot 当前欠缺槽位
current_node 当前轮命中ID


"""
import pandas as pd
import json
import re
from config import config

class Dialouge_System:
    def __init__(self) -> None:
        self.scenario = config['scenes']
        self.available_scenario_list = [scenario['name'] for scenario in self.scenario]
        self.scenario_path = None
        self.slot_fitting_path = None
        
    def load(self):
        #加载场景
        self.all_nodes_info = {}
        self.loading_scenario(self.scenario_path) #id 为 key, node_info为value

        #语义槽填充
        self.slot_filling_info = {} #slot 为key, [query, values]为value
        self.load_fitting(self.slot_fitting_path)

    def loading_scenario(self, scenario_path) -> list:
        scenario_name = scenario_path.replace('.json', '')
        with open(scenario_path, 'r', encoding='utf-8') as f:
            for node in json.load(f):
                node_id = node['id']
                node_id = scenario_name + '_' + node_id
                if 'childnode' in node:
                    node['childnode'] = [scenario_name + '_' + child_node_id for child_node_id in node['childnode']]
                self.all_nodes_info[node_id] = node
        return

    def load_fitting(self, slot_fitting_path):
        slot = pd.read_excel(slot_fitting_path, sheet_name='Sheet1')
        for _, row in slot.iterrows():
            slot_name = row['slot']
            self.slot_filling_info[slot_name] = [row['query'], row['values']]

    def run(self, user_query, history):
        if history is None:
            history = {}
        choice_scenario = self.scenario_choice(user_query)
        for i in range(len(self.scenario)):
            if self.scenario[i]['name'] == choice_scenario:
                self.scenario_path = self.scenario[i]['path']
                self.slot_fitting_path = self.scenario[i]['slot_fitting_path']
        self.load() #加载数据
        history['available_node'] = [next(iter(self.all_nodes_info.keys()))]
        history['user_query'] = user_query
        # 检查用户请求中是否包含重听关键词
        if self.check_repeat_keyword(user_query):
            return self.repeat_last_response(history)
        history = self.NLU(history)
        history = self.DM(history)
        history = self.NLG(history)
        history['last_user_query'] = user_query #保存用户上一轮查询结果

        return history

    def check_repeat_keyword(self, user_query):
        # 检查用户请求中是否包含重听关键词
        repeat_keywords = ["再次", "重复", "复述"]  # 可以根据实际情况添加其他关键词
        for keyword in repeat_keywords:
            if keyword in user_query.lower():
                return True
        return False
    
    def repeat_last_response(self, history):
        # 重复上一轮对话的内容
        if 'last_user_query' in history and 'response' in history:
            repeated_response = f"User: {history['last_user_query']}\nSystem: {history['response']}"
            return repeated_response
        else:
            return "I'm sorry, I don't have anything to repeat."

    def scenario_choice(self, user_query):
        score_list = []
        for scenario in self.available_scenario_list:
            score = self.sentence_similarity(user_query, scenario)
            score_list.append(score)
        max_score = max(score_list)
        index = score_list.index(max_score)
        return self.available_scenario_list[index]
    
    def sentence_similarity(self, string1, string2):
        #使用jaccard距离
        similarity = len(set(string1) & set(string2)) / len(set(string1) | set(string2))
        return similarity
        
    def NLU(self, history):
        #意图识别 + 槽位抽取
        user_query = history['user_query']
        history = self.intend_detection(user_query, history)
        history = self.slot_fitting(user_query, history)
        return history
    
    def intend_detection(self, user_query, history):
        #意图识别
        score_list = []
        max_score = -1
        cur_node_id = None
        for node_id in history['available_node']:
            node_info = self.all_nodes_info[node_id]
            score = self.sentence_similarity(user_query, node_info)
            score_list.append(score)
            if score > max_score:
                max_score = score
                cur_node_id = node_id
        history['current_node'] = cur_node_id
        return history

    def slot_fitting(self, user_query, history):
        #槽位抽取
        slot_list = self.all_nodes_info[history["current_node"]].get("slot", [])
        for slot in slot_list:
            _, candidates = self.slot_filling_info[slot]
            search_result = re.search(candidates, user_query)
            if search_result is not None:
                history[slot] = search_result.group()
        return history

    def DM(self, history):
        #检查所有槽位是否均填充
        history = self.dst(history)
        #如果缺少槽位, 反问;如果不缺少, 回答
        history = self.dpo(history)
        return history

    def dst(self, history):
        #检查所有槽位是否均填充
        for slot in self.all_nodes_info[history["current_node"]].get("slot", []):
            if slot not in history or history[slot] == '':
                history['missing_slot'] = slot
                return history
        history['missing_slot'] = None
        return history
    
    def dpo(self, history):
        #如果不缺少槽位, 回答
        if history['missing_slot'] is None:
            history['policy'] = 'answer'
            #开放子节点，用于下一轮对话
            history["avaliable_nodes"] = self.all_nodes_info[history["hit_node_id"]].get("childnode", [])
        else:
            history['policy'] = 'question'
            history["avaliable_nodes"] = [history['current_node']]
        return history
    
    def NLG(self, history):
        if history['policy'] == 'answer':
            #根据命中节点找response
            response = self.all_nodes_info[history["hit_node_id"]]["response"]
            #把槽位替换成memory中真实值
            response = self.replace_slot_in_response(history, response)
            history["response"] = response
        elif history['policy'] == 'question':
            #找到missing_slot
            slot = history["missing_slot"]
            ask_sentence, _ = self.slot_filling_info[slot]
            history["response"] = ask_sentence
        return history
    
    def replace_slot_in_response(self, history, response):
        #把response中的槽位替换成memory中真实值
        #找到命中节点的槽位
        slots = self.all_nodes_info[history["hit_node_id"]].get("slot", [])
        for slot in slots:
            response = re.sub(slot, history[slot], response)        
        return response

if __name__ == "__main__":
    dialouge_system = Dialouge_System()
    history= dialouge_system.run("我想买短袖衣服", {})
    print(history)
    history = {}
    while True:
        user_query = input("输入：")
        memory = dialouge_system.run(user_query, history)
        print(history["response"])
        print(history)
        print("===============")