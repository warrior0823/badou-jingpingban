import csv

import pandas as pd
import numpy as np


def get_top(data: pd.DataFrame, match_key, match_num, ascending):
    top_values = data.sort_values(by=match_key, ascending=ascending).head(match_num)
    if ascending:
        sort_str = f"根据{match_key}排序从低到高第"
    else:
        sort_str = f"根据{match_key}排序从高到低第"
    for idx, t in enumerate(top_values.values):
        print(
            f"{sort_str}{idx + 1}名模型结果，模型类型：{t[0]}, 准确率：{t[-4]}  ,召回率：{t[-3]},loss:{t[-2]},精确率：{t[-1]} , epoch: {t[1]}, 优化器 :{t[2]} 学习率: {t[3]}, "
            f"池化方式:{t[6]}，训练所用时间：{t[7]} ,hidden_size:{t[4]},batch_size:{t[5]}\n")


def get_mean(data: pd.DataFrame, match_keys, match_values):
    this_d = data
    pre_str = ""
    for idx, match_key in enumerate(match_keys):
        match_value = match_values[idx]
        this_d = this_d[this_d[match_key] == match_value]
        pre_str += f"{match_key} 为 {match_value}，"
    print(f"当{pre_str}模型准确率平均为{this_d['acc'].mean()}\n")
    print(f"当{pre_str}模型召唤率平均为{this_d['call_back'].mean()}\n")
    print(f"当{pre_str}模型loss平均为{this_d['loss'].mean()}\n")
    print(f"当{pre_str}模型精确率平均为{this_d['precision'].mean()}\n")


with open("output/res.csv") as fr:
    d = pd.read_csv(fr)
    print(f"所有模型类型 {d['model_type'].unique()}\n")
    # middle_acc=d['acc'].
    print(f"所有模型准确率均值：{d['acc'].mean()} \n")

    get_top(d, 'acc', 3, False)
    get_top(d, 'acc', 3, True)

    get_top(d, 'call_back', 3, False)
    get_top(d, 'call_back', 3, True)

    get_top(d, 'precision', 3, False)
    get_top(d, 'precision', 3, True)

    get_top(d, 'time_value', 3, False)
    get_top(d, 'time_value', 3, True)
    for model_type in d['model_type'].unique():
        get_mean(d, match_keys=['model_type'], match_values=[model_type])
        get_mean(d, match_keys=['model_type', 'opt'], match_values=[model_type, 'sgd'])
        get_mean(d, match_keys=['model_type', 'opt'], match_values=[model_type, 'adam'])
