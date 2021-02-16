#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-25
# @Contact    : qichun.tang@bupt.edu.cn
from collections import Counter

import gc
import pandas as pd

user_log: pd.DataFrame = pd.read_pickle('data/user_log.pkl')
print(user_log)
core_ids = ['user_id', 'merchant_id']
item_ids = ['brand_id', 'item_id', 'cat_id']


def value_counts_ratio(seq):
    n_seq = len(seq)
    count = Counter(seq)
    return {k: v / n_seq for k, v in count.items()}


for pk in core_ids:
    df = user_log.groupby(pk).agg(
        dict(zip(item_ids, [value_counts_ratio] * len(item_ids)))).\
        reset_index()
    df.to_pickle(f'data/{pk}_value_counts_ratio.pkl')
    del df
    gc.collect()
