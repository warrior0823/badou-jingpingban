# coding: utf-8
"""
基于场景脚本的对话系统
"""

import re
import pandas
import json

class DialogueSystem:
    def __init__(self):
        self.load()     # 完成所有资源加载

    def load(self):
        self.all_nodes_info = {}    # id为key，节点信息为value
        # 加载场景脚本
        self.load_scenario("scenario-买衣服.json")
        self.slot_info = {}        # id为槽位名称，value是[query, value]
        # 加载填槽文件
        self.load_template("slot_fitting_templet.xlsx")

    def load_scenario(self, file_path):
        scenario_name = file_path.replace(".json", "")
        with open(file_path, "r", encoding="utf-8") as f:
            for node_info in json.load(f):
                node_id = node_info["id"]
                node_id = scenario_name + "_" + node_id
                if "childnode" in node_info:
                    node_info["childnode"] = [scenario_name + "_" + child_node_id for child_node_id in node_info["childnode"]]
                self.all_nodes_info[node_id] = node_info
        # print(self.all_nodes_info)
        return

    def load_template(self, file_path):
        self.slot_template = pandas.read_excel(file_path, sheet_name="Sheet1")
        for index, row in self.slot_template.iterrows():
            self.slot_info[row["slot"]] = [row["query"], row["values"]]
        # print("slot info: \n", self.slot_info)
        return

    def run(self, user_query, memory):
        if len(memory) == 0:
            memory["available_nodes"] = ["scenario-买衣服_node1"]
        memory["user_query"] = user_query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory

    def nlu(self, memory):
        # 意图识别 + 槽位抽取
        memory = self.intent_recognition(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self, memory):
        # 意图识别
        hit_score = -1
        hit_node_id = None
        for node_id in memory["available_nodes"]:      # 当前可访问节点
            node_info = self.all_nodes_info[node_id]
            score = self.calc_intent_score(memory, node_info)
            if score > hit_score:
                hit_score = score
                hit_node_id = node_id
        memory["hit_node_id"] = hit_node_id
        memory["hit_score"] = hit_score
        return memory

    def calc_intent_score(self, memory, node_info):
        user_query = memory["user_query"]
        intent_list = node_info["intent"]
        all_scores = []
        for intent in intent_list:
            score = self.sentence_similarity(user_query, intent)
            all_scores.append(score)
        return max(all_scores)

    def sentence_similarity(self, sentence1, sentence2):
        return len(set(sentence1) & set(sentence2)) / len(set(sentence1) | set(sentence2))

    def slot_filling(self, memory):
        # 槽位抽取
        user_query = memory["user_query"]
        slot_list = self.all_nodes_info[memory["hit_node_id"]].get("slot", [])
        for slot in slot_list:
            # print("slot is: ", slot)
            _, candidates = self.slot_info[slot]
            # print(candidates, user_query)
            search_result = re.search(candidates, user_query)
            if search_result is not None:
                memory[slot] = search_result.group()
        # print("memory: ", memory)
        return memory

    def dst(self, memory):
        # 对话状态跟踪，检查命中节点对应的槽位信息是否齐备
        for slot in self.all_nodes_info[memory["hit_node_id"]].get("slot", []):
            # print(self.all_nodes_info)
            if slot not in memory or memory[slot] == "":
                memory["missing_slot"] = slot
                return memory
        memory["missing_slot"] = None
        return memory

    def dpo(self, memory):
        # 如果缺少槽位：反问；如果不缺槽位：回答
        if memory["missing_slot"] is not None:
            slot = memory["missing_slot"]
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory["hit_node_id"]]
        else:
            memory["policy"] = "answer"
            memory["available_nodes"] = self.all_nodes_info[memory["hit_node_id"]].get("childnode", [])
        return memory

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


if __name__ == '__main__':
    dialogue_system = DialogueSystem()
    memory = {}  # 对话状态跟踪
    # print(memory)
    while True:
        user_query = input("请输入：")
        dialogue_system.run(user_query, memory)
        print("回答：", memory["response"])
        print(memory)