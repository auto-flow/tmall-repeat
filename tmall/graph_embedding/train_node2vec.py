#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-24
# @Contact    : qichun.tang@bupt.edu.cn
import csrgraph as cg
from joblib import dump
from nodevectors import Node2Vec

G = cg.read_edgelist("data/graph_data.csv", directed=False, sep=',')
node2vec = Node2Vec(threads=6, n_components=100, w2vparams=dict(workers=12))
node2vec.fit(G)
print(node2vec)
dump(node2vec, "data/node2vec.pkl")
