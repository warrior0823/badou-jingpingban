# coding:utf8

'''
基于场景脚本的对话系统

memory中的关键字段
avaliable_nodes  当前可以访问的节点
hit_node_id  当前轮命中的节点id
hit_score   当前轮命中节点的得分
user_query  用户当轮输入
missing_slot 当前欠缺的槽位
policy   当前策略，策略2种 ask 和 answer
'''

import re
import os
import json
import pandas


class DialougeSystem:
    def __init__(self):
        self.all_nodes_info = {}  # id为key ，节点信息为value
        self.slot_info = {}  # id为槽位名称，value为[query, value]
        self.init_memeory = {}
        self.load()  #完成所有数据资源加载
        #self.database_path = "database.json"

    def load(self):
        # 加载场景脚本
        self.load_scenario("scenario-买衣服.json")
        # 加载填槽模板
        self.load_template("slot_fitting_templet.xlsx")

    def load_scenario(self, file_name):
        scenario_name = file_name.replace(".json", "")
        # 加载场景脚本
        with open(file_name, "r", encoding="utf8") as f:
            for node_info in json.load(f):
                node_id = node_info["id"]
                node_id = scenario_name + "_" + node_id
                if "childnode" in node_info:
                    node_info["childnode"] = [scenario_name + "_" + child_node_id for child_node_id in node_info["childnode"]]
                self.all_nodes_info[node_id] = node_info
        self.init_memeory["avaliable_nodes"] = ["scenario-买衣服_node1"]
        return

    def load_template(self, file_name):
        # 加载填槽模板
        self.slot_template = pandas.read_excel(file_name, sheet_name="Sheet1")
        for index, row in self.slot_template.iterrows():
            self.slot_info[row["slot"]] = [row["query"], row["values"]]
        return

    def run(self, user_query, memory):
        if len(memory) == 0:
            memory = self.init_memeory
        memory["user_query"] = user_query
        self.check_user_query_is_valid(memory)
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory

    def check_user_query_is_valid(self, memory):
        count = 0
        while len(memory["user_query"]) <= 0:
            count += 1
            if count > 3:
                print("对不起，您输入的太少了，请重新输入！")
                memory["user_query"] = input(":")
                count = 0
            else:
                print("请再说一遍！")
                memory["user_query"] = input(":")
        return

    def nlu(self, memory):
        # 意图识别 + 槽位抽取
        memory = self.intent_recognition(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self, memory):
        # 意图识别, 选出分值最高的节点
        hit_score = -1
        hit_node_id = None
        for node_id in memory["avaliable_nodes"]:
            node_info = self.all_nodes_info[node_id]
            score = self.calc_intent_score(memory, node_info)
            if score > hit_score:
                hit_score = score
                hit_node_id = node_id
        memory["hit_node_id"] = hit_node_id
        memory["hit_score"] = hit_score
        return memory

    def calc_intent_score(self, memory, node_info):
        # 计算意图识别的分数
        user_query = memory["user_query"]
        intent_list = node_info["intent"]
        # 字符串匹配选分值最高的
        all_scores = []
        for intent in intent_list:
            score = self.sentence_similarity(user_query, intent)
            all_scores.append(score)
        return max(all_scores)

    def sentence_similarity(self, string1, string2):
        #使用jaccard距离
        score = len(set(string1) & set(string2)) / len(set(string1) | set(string2))
        return score

    def slot_filling(self, memory):
        # 槽位抽取
        user_query = memory["user_query"]
        slot_list = self.all_nodes_info[memory["hit_node_id"]].get("slot", [])
        for slot in slot_list:
            _, candidates = self.slot_info[slot]
            search_result = re.search(candidates, user_query)
            if search_result is not None:
                memory[slot] = search_result.group()
        return memory


    def dst(self, memory):
        # 检查命中节点对应槽位信息是否齐备
        for slot in self.all_nodes_info[memory["hit_node_id"]].get("slot", []):
            if slot not in memory or memory[slot] == "": # 如果缺槽位
                memory["missing_slot"] = slot
                return memory
        memory["missing_slot"] = None
        return memory


    def dpo(self, memory):
        # 如果缺少槽位，反问；如果不缺槽位，回答
        if memory["missing_slot"] is not None:
            slot = memory["missing_slot"]
            memory["policy"] = "ask"
            # 留在本节点
            memory["avaliablie_nodes"] = [memory["hit_node_id"]]
        else:
            memory["policy"] = "answer"
            # 开放子节点
            memory["avaliablie_nodes"] = self.all_nodes_info[memory["hit_node_id"]].get("childnode", [])
            # 在数据库中查找对应的服装
            memory["database_name"] = "database.json"
            #self.take_action(memory)
        return memory

    def take_action(self, memory):
        # 打开并读取JSON文件
        database_name = memory["database_name"]
        # type_ = memory["type_"]
        # color = memory["color"]
        # size = memory["size"]
        type_ = 'shirt'
        color = 'red'
        size = 'M'
        with open(database_name, "r", encoding="utf8") as f:
            clothing_list = json.load(f)
        # 遍历服装列表，寻找匹配的服装
        for clothing in clothing_list:
            if clothing['type'] == type_ and clothing['color'] == color and clothing['size'] == size:
                return clothing
            else:
                # 如果没有找到匹配的服装，返回None
                print("No matching clothing found.")
                return None


    def nlg(self, memory):
        # 根据policy做选择
        if memory["policy"] == "ask":
            # 找到missing_slot
            slot = memory["missing_slot"]
            ask_sentence, _ = self.slot_info[slot]
            memory["response"] = ask_sentence
        elif memory["policy"] == "answer":
            # 根据命中节点找response
            response = self.all_nodes_info[memory["hit_node_id"]]["response"]
            # 把槽位替换成memory中真实值
            response = self.replace_slot_in_response(memory, response)
            memory["response"] = response
        return memory

    def replace_slot_in_response(self, memory, response):
        # 把response中的槽位替换成memory中真实值
        # 找到命中节点的槽位
        slots = self.all_nodes_info[memory["hit_node_id"]].get("slot", [])
        for slot in slots:
            response = re.sub(slot, memory[slot], response)
        return response


if __name__ == "__main__":
    dialouge_system = DialougeSystem()
    # print(dialouge_system.all_nodes_info)
    # print(dialouge_system.slot_info)
    # print(dialouge_system.run("我要买衣服", {}))

    memory = {}
    while True:
        user_query = input("输入:")
        memory = dialouge_system.run(user_query, memory)
        print(memory["response"])
        print(memory)
        print("================")

