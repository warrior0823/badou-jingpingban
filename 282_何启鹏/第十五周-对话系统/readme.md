## 加入重听机制

```
请输入：我要买衣服
您想买长袖、短袖还是半截袖
{'available_nodes': ['买衣服_node1'], 'query': '我要买衣服', 'hit_node': '买衣服_node1', 'score': 0.0, 'repeat': False, 'require_slot': '#服装类型#', 'policy': 'ask', 'response': '您想买长袖、短袖还是半截袖'}

请输入：长袖
您喜欢什么颜色
{'available_nodes': ['买衣服_node1'], 'query': '长袖', 'hit_node': '买衣服_node1', 'score': 0.0, 'repeat': False, 'require_slot': '#服装颜色#', 'policy': 'ask', 'response': '您喜欢什么颜色', '#服装类型#': '长袖'}

请输入：黄色
您想要多尺寸
{'available_nodes': ['买衣服_node1'], 'query': '黄色', 'hit_node': '买衣服_node1', 'score': 0.0, 'repeat': False, 'require_slot': '#服装尺寸#', 'policy': 'ask', 'response': '您想要多尺寸', '#服装类型#': '长袖', '#服装颜色#': '黄'}

请输入：重复一遍
您想要多尺寸
{'available_nodes': ['买衣服_node1'], 'query': '重复一遍', 'hit_node': '买衣服_node5', 'score': 1.0, 'repeat': True, 'require_slot': None, 'policy': 'repeat', 'response': '您想要多尺寸', '#服装类型#': '长袖', '#服装颜色#': '黄'}
```

