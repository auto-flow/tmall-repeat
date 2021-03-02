#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-23
# @Contact    : qichun.tang@bupt.edu.cn
from collections import defaultdict
from functools import partial

import pandas as pd
from joblib import dump

user_log: pd.DataFrame = pd.read_pickle("user_log.pkl")[['user_id', 'merchant_id', 'action_type']]

G = defaultdict(lambda: defaultdict(int))
entity2id = {}


def get_id_of_entity(id, prefix):
    id = f"{prefix}{id}"
    if id not in entity2id:
        entity2id[id] = len(entity2id)
    return entity2id[id]


uid = partial(get_id_of_entity, prefix="u")
mid = partial(get_id_of_entity, prefix="m")

# 遍历
for i, (user_id, merchant_id, action_type) in user_log.iterrows():
    G[uid(user_id)][mid(merchant_id)] += 1

# 输出
data = []
for u in G.keys():
    Gu = G[u]
    for v, Guv in Gu.items():
        data.append([u, v, Guv])

pd.DataFrame(data).to_csv('data/graph_data.csv', header=False, index=False)
# 对字典进行序列化
dump(entity2id, "data/entity2id.pkl")
# 因为含有lambda，所以无法直接序列化
# 结点规模是40w
dump(dict(G), "data/graph_data.pkl")
