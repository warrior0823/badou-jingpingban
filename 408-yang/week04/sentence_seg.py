# encoding=utf-8

# 参考 https://liweiwei1419.github.io/leetcode-solution-blog/leetcode-problemset/backtracking/0140-word-break-ii.html#%E2%80%9C%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92-%E5%9B%9E%E6%BA%AF%E2%80%9D%E6%B1%82%E8%A7%A3%E5%85%B7%E4%BD%93%E5%80%BC%EF%BC%88python-%E4%BB%A3%E7%A0%81%E3%80%81java-%E4%BB%A3%E7%A0%81%EF%BC%89
# https://blog.csdn.net/ibelieve8013/article/details/103133725
#词典，每个词后方存储的是其词频，仅为示例，也可自行添加


from typing import List
from collections import deque

dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

class Solution1:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        size = len(s)
        # 题目中说非空字符串，以下 assert 一定通过
        assert size > 0

        # 预处理，把 wordDict 放进一个哈希表中
        word_set = {word for word in wordDict}
        # print(word_set)

        # 状态：以 s[i] 结尾
        # 这种状态定义很常见
        dp = [False for _ in range(size)]

        dp[0] = s[0] in word_set

        # print(dp)

        # 使用 r 表示右边界，可以取到
        # 使用 l 表示左边界，也可以取到
        for r in range(1, size):
            # Python 的语法，在切片的时候不包括右边界
            # 如果整个单词就直接在 word_set 中，直接返回就好了
            # 否则把单词做分割，挨个去判断
            if s[:r + 1] in word_set:
                dp[r] = True
                continue

            for l in range(r):
                # dp[l] 写在前面会更快一点，否则还要去切片，然后再放入 hash 表判重
                if dp[l] and s[l + 1: r + 1] in word_set:
                    dp[r] = True
                    # 这个 break 很重要，一旦得到 dp[r] = True ，循环不必再继续
                    break
        res = []
        # 如果有解，才有必要回溯
        if dp[-1]:
            queue = deque()

            self.__dfs(s, size - 1, wordDict, res, queue, dp)
        return res

    def __dfs(self, s, end, word_set, res, path, dp):
        # print('刚开始', s[:end + 1])
        # 如果不用拆分，整个单词就在 word_set 中就可以结算了
        if s[:end + 1] in word_set:
            path.appendleft(s[:end + 1])
            res.append(' '.join(path))
            path.popleft()

        for i in range(end):
            if dp[i]:
                suffix = s[i + 1:end + 1]
                if suffix in word_set:
                    path.appendleft(suffix)
                    self.__dfs(s, i, word_set, res, path, dp)
                    path.popleft()



class Solution:
    def wordBreak(self, s: str, dict: set)-> List[str]:
        dp = [False] * len(s)
        for i in range(len(s)):
            for j in range(i, len(s)):
                dp[i] = dp[i] or self.match(s[i:j+1], dict)
        print('dp ' ,dp)
        self.output(dp, s, len(s) - 1)
        return self.result

    def match(self, s: str, dict: set) -> bool:
        print('s ',s)
        return s in dict

    def output(self, dp, s, i):
        if i == -1 :
            print('mystring: ',' '.join(reversed(self.mystring)))
            self.result.append(' '.join(reversed(self.mystring)))
        else:
            for k in range(i, -1, -1):
                # print('k ',k) 

                if dp[k]:
                    if s[k:i+1] in dict:
                        self.mystring.append(s[k:i+1])
                        self.output(dp, s, k-1)
                        self.mystring.pop()

    def __init__(self):
        self.result = []
        self.mystring = []
        self.dp = None

# # 使用示例
# solution = Solution()
# s = "经常有意见分歧"
# result = solution.wordBreak(s, dict)
# # dict = {"leet", "code"}
# print(len(result))
# for item in result:
#     print(item+"\n")
        
# 输出根据字典能够切分出的所有切分方式
def split_sentence(sentence,dict):
    result = []
    mystring = []
    dp = [0] * len(sentence)
    for i in range(len(sentence)):
        for j in range(i,len(sentence)):
            if sentence[i:j+1] in dict:
                dp[i]= 1
    output(result,mystring, dp,sentence,len(sentence)-1)
    return result

def output(result,mystring, dp,sentence,index):
    if index == -1:
        result.append(' '.join(reversed(mystring)))
    else:
        for k in range(index, -1, -1):
            if dp[k]:
                if sentence[k:index+1] in dict:
                    mystring.append(sentence[k:index+1])
                    output(result, mystring, dp, sentence, k-1)
                    mystring.pop()
    return result

if __name__ =="__main__":
    sentence = "经常有意见分歧"
    result = split_sentence(sentence,dict=dict) 
    for item in result:
        print(item+"\n")
        