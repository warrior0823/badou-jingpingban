import re
import json
import pandas

"""

对话系统

"""

class DialogueSystem:
    def __init__(self):
        self.load()
        self.available_nodes = ["买衣服_node1"]

    def load(self):
        self.all_slot_info={}
        self.load_excel_templet("./scenario/slot_fitting_templet.xlsx")
        self.all_node_info={}
        self.load_scenario("./scenario/scenario-买衣服.json")

    def load_excel_templet(self,path):
        df = pandas.read_excel(path)
        for i in range(len(df)):
            slot = df['slot'][i]
            query = df['query'][i]
            values = df['values'][i]
            # 问题和 槽位信息
            self.all_slot_info[slot] = [query, values]

    def load_scenario(self,path):
        # 场景名称
        scen_name = path.split('/')[-1].split("-")[1].split('.')[0]
        with open(path,encoding='utf-8') as f:
            # 读成python的数据结构
            data = json.loads(f.read())
            for node_info in data:
                node_name = node_info["id"]
                # 判断node_info 中是否有childnode ，字典可以直接询问是否存在键，但无法直接访问是否存在某个值
                if "childnode" in node_info:
                    node_info["childnode"] = [scen_name +"_"+ name for name in node_info["childnode"]]
                node_name = scen_name + "_" + node_name
                self.all_node_info[node_name] = node_info

    def get_intent(self,memory):
        score = -1
        hit_node =None
        for node in self.available_nodes:
            intent_score = self.get_intent_score(memory,node)
            if intent_score >score:
                score =intent_score
                hit_node = node

        memory["score"] = score
        memory["hit_node"] = hit_node
        return memory

    # 节点意图识别
    def get_intent_score(self,memory,node):
        scores=[]
        for intent in self.all_node_info[node]["intent"]:
            score = self.sentence_match(intent,memory["query"])
            scores.append(score)
        return max(scores)

    #  文本匹配算法
    def sentence_match(self,intent,query):
        return  len(set(intent)&set(query)) / len(set(intent)|set(query))

    # 槽位抽取
    def get_slot(self,memory):
        # 根据获得的节点，遍历槽位信息
        for slot in self.all_node_info[memory["hit_node"]].get("slot",[]):
            # 获得槽位
            pattern = self.all_slot_info[slot][1]
            # 搜索到一个就结束，re.search()方法扫描整个字符串，并返回第一个成功的匹配。如果匹配失败，则返回None。
            match = re.search(pattern, memory["query"])

            if match is not None:
                memory[slot] =  match.group()
        return memory

    def nlu(self,memory):
        memory = self.get_intent(memory)
        memory = self.get_slot(memory)
        return memory

    # 判断当前的槽位是否填满
    def dst(self,memory):
        for slot in self.all_node_info[memory["hit_node"]].get("slot", []):
            if slot not in memory:
                memory["require_slot"] = slot
                return memory

        memory["require_slot"] = None
        return memory

    def take_action(self,memory):
        return memory

    # 策略选择
    def pm(self,memory):
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            memory = self.take_action(memory)
            self.available_nodes = self.all_node_info[memory["hit_node"]].get("childnode")
        else:
            memory["policy"] = "ask"
            self.available_nodes = [memory["hit_node"]]
        return memory

    def nlg(self,memory):
        if memory["policy"] == "reply":
            response = self.all_node_info[memory["hit_node"]].get("response")
            response = self.fill_templet(memory,response)
        else:
            slot = memory["require_slot"]
            response = self.all_slot_info[slot][0]

        memory["response"] = response
        return memory

    def fill_templet(self,memory,response):
        for slot in self.all_node_info[memory["hit_node"]].get("slot",[]):
            response = re.sub(slot, memory[slot],response)
        return response

    def run(self,query,memory):

        memory["query"] = query
        # 自然语言理解
        memory = self.nlu(memory)
        # 对话状态的跟踪  dialogue state tracking
        memory = self.dst(memory)
        # 策略选择 policy making
        memory = self.pm(memory)
        # 自然语言生成
        memory = self.nlg(memory)
        return memory


if __name__ =="__main___":
    dl = DialogueSystem()
    query = "我想买衣服"

    print(dl.all_node_info)
    memory={}
    print(dl.all_slot_info)
    while True:
        query = input("用户输入(按q退出）: ")
        if query =="q" or query in "再见":
            print("系统回答：谢谢光临")
            break
        memory = dl.run(query,memory)
        print(memory)
        print("系统回答：",memory["response"])
        print()
