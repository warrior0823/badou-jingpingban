"""
基于场景脚本的对话系统
memory中的关键字段
available_nodes 当前可以访问的节点
hit_node_id 当前轮命中的节点id
hit_score 当前轮命中节点的得分
user_query 用户当轮输入
missing_slot 当前欠缺的槽位
policy 策略：如果缺槽位：反问 如果槽位不缺：回答 ask answer repetition
"""

import re
import os
import json
import pandas


class DialougeSystem:
    def __init__(self) -> None:
        self.load()  # 完成所有的资源加载

    def load(self):
        # 加载场景脚本
        self.all_nodes_info = {}  # id为key，节点信息为value
        self.load_scenario("scenario-买衣服.json")
        # 加载填槽模版
        self.slot_info = {}  # id为槽位名称 values是[query,value]
        self.load_template("slot_fitting_templet.xlsx")

    def load_scenario(self, path):
        scenario_name = path.replace(".json", "")
        # 加载场景脚本
        with open(path, "r", encoding="utf-8") as f:
            for node_info in json.load(f):
                node_id = node_info["id"]
                node_id = scenario_name + "_" + node_id
                if "childnode" in node_info:
                    node_info["childnode"] = [
                        scenario_name + "_" + chlidnode
                        for chlidnode in node_info["childnode"]
                    ]
                self.all_nodes_info[node_id] = node_info
        return

    def load_template(self, path):
        # 加载填槽模版
        self.template = pandas.read_excel(path, sheet_name="Sheet1")
        for index, row in self.template.iterrows():
            slot_name = row["slot"]
            query = row["query"]
            value = row["values"]
            self.slot_info[slot_name] = [query, value]
        return

    def run(self, user_query, memory):
        if len(memory) == 0:
            memory["avaliable_nodes"] = ["scenario-买衣服_node1"]
            memory["history"] = []
        memory["user_query"] = user_query
        if user_query == "重听":
            memory["policy"] = "repetition"
        else:
            memory = self.nlu(memory)
            memory = self.dst(memory)
            memory = self.dpo(memory)
        memory = self.nlg(memory)

        # 保存当前对话轮次
        if user_query != "重听":
            memory["history"].append((user_query, memory["response"]))
        return memory

    def nlu(self, memory):
        # 意图识别+槽位抽取
        memory = self.intent_recognition(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self, memory):
        # 意图识别,选出分值最高的节点
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

    def sentence_similarity(self, sentence1, sentence2):
        # 使用jaccard距离
        score = len(set(sentence1) & set(sentence2)) / len(
            set(sentence1) | set(sentence2)
        )

        return score

    def slot_filling(self, memory):
        # 槽位抽取
        user_query = memory["user_query"]
        slot_list = self.all_nodes_info[memory["hit_node_id"]].get("slot", [])
        for slot in slot_list:
            _, candidate = self.slot_info[slot]
            search_result = re.search(candidate, user_query)
            if search_result is not None:
                memory[slot] = search_result.group()
        return memory

    def dst(self, memory):
        # 检查命中节点对应的槽位信息是否齐备
        hit_node_id = memory["hit_node_id"]
        node_info = self.all_nodes_info[hit_node_id]
        slot_list = node_info.get("slot", [])
        for slot in slot_list:
            if slot not in memory or memory[slot] == "":  # 如果缺槽位
                memory["missing_slot"] = slot
                return memory
        memory["missing_slot"] = None
        return memory

    def dpo(self, memory):
        # 策略：如果缺槽位：反问 如果槽位不缺：回答
        if memory["missing_slot"] is not None and memory["user_query"] != "重听":
            slot = memory["missing_slot"]
            memory["policy"] = "ask"
            # 留在本节点
            memory["avaliable_nodes"] = [memory["hit_node_id"]]
        elif memory["user_query"] == "重听":
            memory["policy"] = "repetition"
        else:
            memory["policy"] = "answer"
            # 开放子节点
            memory["avaliable_nodes"] = self.all_nodes_info[memory["hit_node_id"]].get(
                "childnode", []
            )
        return memory

    def nlg(self, memory):
        # 根据policy做选择
        policy = memory["policy"]
        if policy == "ask":
            # 找到missing_slot
            slot = memory["missing_slot"]
            ask_sentence, _ = self.slot_info[slot]
            memory["response"] = ask_sentence
        elif memory["policy"] == "answer":
            # 根据节点找到response
            response = self.all_nodes_info[memory["hit_node_id"]]["response"]
            # 把槽位替换成memoty中真实值
            response = self.replace_slot_in_response(memory, response)
            memory["response"] = response
        elif policy == "repetition":
            # 重听功能
            if memory["history"]:
                last_user_query, last_response = memory["history"][-1]
                memory["response"] = (
                    f"上一个问题: {last_user_query}, 上一个回复: {last_response}"
                )
            else:
                memory["response"] = "没有什么可以重复的内容."
        return memory

    def replace_slot_in_response(self, memory, response):
        # 把槽位替换成memory中真实值
        # 找到命中节点的槽位
        slot_list = self.all_nodes_info[memory["hit_node_id"]].get("slot", [])
        for slot in slot_list:
            if slot in memory:
                response = re.sub(slot, memory[slot], response)
        return response


if __name__ == "__main__":
    dialouge_system = DialougeSystem()
    memory = {}
    while True:
        user_query = input("输入：")
        memory = dialouge_system.run(user_query, memory)
        print(memory["response"])
        print(memory)
        print("============================")
