#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-25
# @Contact    : qichun.tang@bupt.edu.cn
'''离线计算相似度特征'''
import os

import numpy as np
import pandas as pd
from joblib import load

train = pd.read_csv('data_format1/train_format1.csv')
test = pd.read_csv('data_format1/test_format1.csv')
train.pop('label')
test.pop('prob')
N = train.shape[0]
all_data = pd.concat([train, test], axis=0)
res = all_data.copy()
pk2vcr = load("data/pk2vcr2.pkl")
core_ids = ['user_id', 'merchant_id']
item_ids = ['brand_id', 'item_id', 'cat_id']
for pk in core_ids:
    columns = pk2vcr[pk].columns[1:]
    all_data = all_data.merge(pk2vcr[pk], "left", pk)
    all_data.rename(columns=dict(zip(
        [f"{x}_vectors" for x in item_ids],
        [f"{pk}_{x}_vectors" for x in item_ids],
    )), inplace=True)
for id_ in item_ids:
    A = np.array(all_data[f"user_id_{id_}_vectors"].tolist())
    B = np.array(all_data[f"merchant_id_{id_}_vectors"].tolist())
    res[f"{id_}_similarity"] = np.sum(A * B, axis=1) / \
                               (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
all_data.to_pickle("data/items_vectors2.pkl")
res.to_pickle("data/similarity_features2.pkl")
os.system('google-chrome https://ssl.gstatic.com/dictionary/static/sounds/oxford/ok--_gb_1.mp3')
