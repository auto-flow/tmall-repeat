#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-24
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import ExtraTreesClassifier
from joblib import dump
from joblib import load
import os

boruta = BorutaPy(
    ExtraTreesClassifier(max_depth=5, n_jobs=4),
    n_estimators='auto', max_iter=1000, random_state=0, verbose=2)

train = load('data/train2.pkl')
train.fillna(0, inplace=True)
train[np.isinf(train)] = 0
y = train.pop('label')
boruta.fit(train, y)
dump(boruta, 'data/boruta3.pkl')
os.system('google-chrome https://ssl.gstatic.com/dictionary/static/sounds/oxford/ok--_gb_1.mp3')
print(boruta)
