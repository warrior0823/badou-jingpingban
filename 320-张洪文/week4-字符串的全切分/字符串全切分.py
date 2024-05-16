Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

# 动态规划
def dp_all_cut(sentence, dicts):
    n = len(sentence)
    # dp[i] 存储从索引 i 到结尾的所有切分方式
    dp = [[] for _ in range(n+1)]
    dp[n] = [[]]

    for i in range(n-1, -1, -1):
        for j in range(i+1, n+1):
            word = sentence[i:j]
            if word in dicts:
                for path in dp[j]:
                    dp[i].append([word] + path)
    return dp[0]

# 递归
def recursion_all_cut(sentence, dicts):
    target = []

    def _all_cut_helper(sentence, start, path):
        if start == len(sentence):
            if path:
                target.append(path)
            return

        for word in Dict:
            if sentence.startswith(word, start):
                _all_cut_helper(sentence, start + len(word), path + [word])

    _all_cut_helper(sentence, 0, [])
    return target


if __name__ == '__main__':
    # 待切分文本
    text = "经常有意见分歧"

    # result = dp_all_cut(text, Dict)
    result = recursion_all_cut(text, Dict)

    print("全切分结果：")
    for res in result:
        print(res)
