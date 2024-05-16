# Week14_Homework
## 目录
- 1. 本地环境准备
- 2. 本地运行

### 本地环境准备
系统: Mac OS
- 尝试使用brew安装Neo4j，报错
```
    /opt/homebrew/var/homebrew/linked libpng is not a valid keg
```
- 改用官网版本(Linux/Mac Executable) 
Link:https://neo4j.com/deployment-center/ 
21-郑奕斐/Week_14_HomeWork/Screenshot 2024-05-16 at 08.28.04.png
- 解压完成文件
- ![alt text](<Screenshot 2024-05-16 at 08.33.15.png>)

- 配套JAVA版本安装(https://www.oracle.com/cn/java/technologies/downloads/#jdk22-mac)
![alt text](<Screenshot 2024-05-16 at 08.31.29.png>)

### 本地运行
- 1. 运行终端进入Neo4j文件夹下的bin文件
```unix
    cd (neo4j路径/bin)
```
- 2. 输入 ./neo4j start 激活neo4j本地化
- 如何确保neo4j已正常激活？
![alt text](<Screenshot 2024-05-16 at 08.40.38.png>)
- 打开浏览器，将红线部分粘贴至网址框
![alt text](<Screenshot 2024-05-16 at 08.45.09.png>)
***初始登陆用户名&密码均为 neo4j 进入后重新设置***
- Neo4j环境准备完毕
![alt text](<Screenshot 2024-05-16 at 08.49.31.png>)

- 3. Python文件数据库导入
库准备 & 代码调试(修改部分代码)
**build_graph.py**
```Python 
    from neo4j import GraphDatabase
    #连接图数据库
    graph = GraphDatabase.driver("bolt://localhost:7687",auth=("neo4j","********"))
    #执行建表脚本
    with graph.session() as session:
        session.run(cypher)
```

**graph_qa_base_on_sentence_match.py**
```Python
#将带有token的模板替换成真实词
    #string:%ENT1%和%ENT2%是%REL%关系吗
    #combination: {"%ENT1%":"word1", "%ENT2%":"word2", "%REL%":"word"}
    def replace_token_in_string(self, string, combination):
        ic(string)
        ic(combination)
        for key, value in combination.items():
            string = string.replace(key, str(value)) -> #避免value值为空类型为dict报错
        return string
#对外提供问答接口
for templet, cypher, score, answer in templet_cypher_score:
    -> #由于graph代码调整，这里需要做配套调整
            with self.graph.session() as session: 
                graph_search_result = session.run(cypher).data()
```
- 完成截图
![alt text](<Screenshot 2024-05-16 at 08.59.29.png>)

- neo4j 关闭
```
./neo4j stop
```
