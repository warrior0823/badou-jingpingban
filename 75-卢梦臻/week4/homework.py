# _*_ coding: UTF-8 _*_
# @Time : 2024/4/16 9:21
# @Author : Yujez
# @File : 0-homeword_v4_dynamic
# @Project : intro_to_ml
'''
动态规划实现
从0位开始遍历
'''
Dict = {"经常":0.1,
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
sentence = "经常有意见分歧"

final_result=[]
temp_list=[]
def all_cut(sentence):
    if sentence=="":
        final_result.append(temp_list.copy())
        return
    for target in range(len(sentence)):
        window=sentence[:target+1]
        if window in Dict:
            temp_list.append(window)
            all_cut(sentence[target+1:])
            temp_list.pop()
if __name__ == '__main__':
    all_cut(sentence)
    print(len(final_result))
