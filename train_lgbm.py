#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-22
# @Contact    : qichun.tang@bupt.edu.cn
import os

import pandas as pd
# from imblearn.ensemble import BalancedBaggingClassifier
from joblib import load
from lightgbm import LGBMClassifier

from tmall.bagging import BalancedBaggingClassifier

train, y, test = load('data/all_data.pkl')
# merchant_w2v: pd.DataFrame = load('data/merchant_w2v.pkl')
# train = pd.merge(train, merchant_w2v, 'left', 'merchant_id')
# test = pd.merge(test, merchant_w2v, 'left', 'merchant_id')
highR_cols = [
    'purchase-merchant_id-user_id-mostlike',
    'merchant_id-user_id-mostlike',
    'merchant_id-item_id-mostlike',
    'user_id'
]
# cat_encoder = CatBoostEncoder(random_state=0, cols=highR_cols).fit(train, y)
# train = cat_encoder.transform(train)
# test = cat_encoder.transform(test)
# train.drop(highR_cols, axis=1, inplace=True)
# test.drop(highR_cols, axis=1, inplace=True)
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
categorical_feature = [
    c for i, c in enumerate(train.columns)
    if c.endswith('merchant_id') or c.endswith('mostlike') and c.split('-')[-2] != 'user_id'
]
feature_name = train.columns.tolist()

# train[cat_features] = train[cat_features].astype(int)
# test[cat_features] = test[cat_features].astype(int)

sample_weight = load('data/sample_weights.pkl').mean(1)[:train.shape[0]]
# train_hid, test_hid = load('data/hidden_features.pkl')
# hid_cols = [f"hidden_{i}" for i in range(20)]
# train[hid_cols] = train_hid
# test[hid_cols] = test_hid
# merchant_w2v = pd.read_pickle('data/merchant_n2v.pkl')
# user_w2v = pd.read_pickle('data/user_n2v.pkl')
# merchant_w2v_col = merchant_w2v.columns.tolist()[1:]
# user_w2v_col = user_w2v.columns.tolist()[1:]
# train.drop(user_w2v_col + merchant_w2v_col, axis=1, inplace=True)
# test.drop(user_w2v_col + merchant_w2v_col, axis=1, inplace=True)
print(train.shape)


class MyLGBMClassifier(LGBMClassifier):
    def fit(self, X, y,
            sample_weight=None):
        super(MyLGBMClassifier, self).fit(
            X, y, feature_name=feature_name, categorical_feature=categorical_feature)


gbm = LGBMClassifier(random_state=None, silent=False, learning_rate=0.04, n_estimators=1000)  # n_estimators=100
bc = BalancedBaggingClassifier(
    gbm, random_state=0, n_estimators=50, n_jobs=1,
    oob_score=True,  # warm_start=True
)

prediction = pd.read_csv('data_format1/test_format1.csv')
prediction.pop('prob')

model = bc.fit(train, y, sample_weight=sample_weight)
# dump(model, "data/catboost.pkl")
y_pred = bc.predict_proba(test)
prediction['prob'] = y_pred[:, 1]
prediction.to_csv('predictions/prediction3.csv', index=False)
os.system('google-chrome https://ssl.gstatic.com/dictionary/static/sounds/oxford/ok--_gb_1.mp3')
print(bc)
