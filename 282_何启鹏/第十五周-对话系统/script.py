import json
import pandas
import re
"""
多轮对话系统
读取场景脚本
"""

class DLSystem:
    def __init__(self):
        self.load()

    def load(self):
        self.all_node_info = {}
        self.load_scenario("./scenario/scenario-买衣服.json")
        # self.load_scenario("./scenario/scenario-看电影.json")
        self.load_slot_values()
        self.permanent_ndoe = ["买衣服_node5"]  # 加入重复节点
        self.response = []  # 保存每一次的输出字符
    def load_scenario(self, path):
        sc_name = path.split("-")[1].split(".")[0]
        with open(path, encoding='utf8') as f:
            content = json.loads(f.read())
            for node in content:
                if "childnode" in node:
                    node["childnode"] = [sc_name + "_" + n for n in node["childnode"]]
                node_name = sc_name + "_" + node["id"]
                self.all_node_info[node_name] = node

    def load_slot_values(self):
        self.all_slot_info = {}
        df = pandas.read_excel("./scenario/slot_fitting_templet.xlsx")
        for i in range(len(df)):
            slot = df["slot"][i]
            query = df["query"][i]
            values = df["values"][i]
            self.all_slot_info[slot] = [query, values]

    def get_intent(self,memory):
        max_score = -1
        hit_node = None
        for node_name in memory["available_nodes"] + self.permanent_ndoe:
            node = self.all_node_info[node_name]
            score = self.get_node_score(node, memory)
            if score > max_score:
                hit_node = node_name
                max_score = score
        memory["hit_node"] = hit_node
        memory["score"] = score
        return memory

    #文本匹配
    def similar_score(self, string1, string2):
        return len(set(string1) & set(string2)) / len(set(string2) | set(string2))

    def get_node_score(self, node, memory):
        res = []
        for intent in node["intent"]:
            score = self.similar_score(intent, memory["query"])
            res.append(score)
        return max(res)

    def get_slot(self, memory):
        hit_node = memory["hit_node"]
        for slot in self.all_node_info[hit_node].get("slot", []):
            pattern = self.all_slot_info[slot][1]
            if re.search(pattern, memory["query"]):
                value = re.search(pattern, memory["query"]).group()
                memory[slot] = value
        return memory

    def nlu(self, memory):
        memory = self.get_intent(memory)
        memory = self.get_slot(memory)
        return memory

    #对话状态跟踪
    def dst(self, memory):
        memory["repeat"] = False
        hit_node = memory["hit_node"]
        for slot in self.all_node_info[hit_node].get("slot", []):
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None
        if self.all_node_info[hit_node]["intent"][0] == "重复一遍" :
            memory["repeat"] = True
        return memory

    def action(self, memory):
        #pass
        return memory

    #
    def pm(self, memory):
        if memory.get("repeat") :
            memory["policy"] = "repeat"  # 播报response部分
        else:
            if memory.get("require_slot") is None:
                memory = self.action(memory)
                hit_node = memory["hit_node"]
                childs = self.all_node_info[hit_node].get("childnode", [])
                memory["available_nodes"] = childs
                memory["policy"] = "reply"  # 播报response部分
            else:
                memory["policy"] = "ask"  # 询问slot部分
                memory["available_nodes"] = [memory["hit_node"]]
        return memory

    def nlg(self, memory):
        if memory["policy"] == "repeat":
            memory["response"] = self.response[-1]  # 如果为repaet就返回上次输出结果
        else:
            if memory["policy"] == "ask":
                slot = memory["require_slot"]
                memory["response"] = self.all_slot_info[slot][0]
                self.response.append(memory["response"])
            else:
                hit_node = memory["hit_node"]
                res = self.all_node_info[hit_node]["response"]
                for slot in self.all_node_info[hit_node].get("slot", []):
                    res = re.sub(slot, memory[slot], res)
                memory["response"] = res
                self.response.append(res)
        return memory

    def run(self, query, memory):
        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.pm(memory)
        memory = self.nlg(memory)
        return memory

if __name__ == "__main__":
    dl = DLSystem()
    print(dl.all_node_info)
    print(dl.all_slot_info)
    print(dl.response)
    # query = "我想买短袖衣服"
    # print(query)
    memory = {"available_nodes":["买衣服_node1"]}
    # memory = dl.run(query, memory)
    # print(memory["response"] 重复一遍)
    print()
    # print(memory)
    while True:
        query = input("请输入：")
        memory = dl.run(query, memory)
        print(memory["response"])
        print(memory)
        print()
        # print(dl.response)