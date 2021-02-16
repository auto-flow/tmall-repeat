#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-22
# @Contact    : qichun.tang@bupt.edu.cn
import warnings
from collections import Counter

import gc
import numpy as np
import pandas as pd
from joblib import dump
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
# spark 变量类型：https://spark.apache.org/docs/latest/sql-ref-datatypes.html
from pyspark.sql.types import ByteType, ShortType, IntegerType, LongType
from pyspark.sql.types import FloatType, DoubleType
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import TimestampType

from stat_feat.fesys_pyspark import FeaturesBuilder


def get_schema_from_df(df: pd.DataFrame):
    convert = {
        "int32": IntegerType(),
        "int64": LongType(),
        "int16": ShortType(),
        "int8": ByteType(),
        "float8": FloatType(),
        "float16": FloatType(),
        "float32": FloatType(),
        "float64": DoubleType(),
    }
    fields = []
    for col_name, dtype in zip(df.columns, df.dtypes):
        dtype_name = dtype.name
        spark_type = convert.get(dtype_name, TimestampType())
        fields.append(StructField(col_name, spark_type, True))
    return StructType(fields)


# from fesys2 import FeaturesBuilder

warnings.filterwarnings("ignore")

user_log = pd.read_pickle('data/user_log.pkl')
user_log['label'] = user_log['label'].astype('int8')  # 之前的操作忘了这步

spark = SparkSession.builder \
    .appName("tmall") \
    .config("master", "local[*]") \
    .enableHiveSupport() \
    .getOrCreate()
sc = spark.sparkContext
sqlContest = SQLContext(sc)
print('loading data to spark dataframe ...')
schema = get_schema_from_df(user_log)
user_log = sqlContest.createDataFrame(user_log.iloc[:10000, :], schema=schema)  # 采样
# user_log = sqlContest.createDataFrame(user_log, schema=schema)
gc.collect()
print('done')

feat_builder = FeaturesBuilder(user_log)
# udf
median_udf = udf(lambda values: float(np.median(values)), FloatType())
median_udf.__name__ = "median"
timediff_udf = udf(lambda x: (max(x) - min(x)).days, IntegerType())
timediff_udf.__name__ = 'timediff'
mostlike_udf = udf(lambda x: Counter(x).most_common(1)[0][0] if len(x) else 0, IntegerType())
mostlike_udf.__name__ = 'mostlike'


def freq_stat_info(seq):
    cnt = Counter(seq)
    size = len(seq)
    freq = [v / size for v in cnt.values()]
    return np.min(freq), np.mean(freq), np.max(freq), np.std(freq)


freq_stat_info_names = ["freq_min", "freq_mean", "freq_max", "freq_std"]
# rebuy_ranges = list(range(1, 11, 1))
rebuy_ranges = list(range(1, 2, 1))

merchant_item_ids = ['merchant_id', 'item_id', 'brand_id', 'cat_id']
item_ids = ['item_id', 'brand_id', 'cat_id']
user_feats = ['age_range', 'gender']
# cross_feats = [['user_id', 'merchant_id']]
core_ids = ['user_id', 'merchant_id']
indicator2action_type = {
    'click': 0,
    'add_car': 1,
    'purchase': 2,
    'favorite': 3,
}
for indicator in ["purchase", None]:
    if indicator is not None:
        action_type = indicator2action_type[indicator]
        feat_builder.core_df = user_log.filter(user_log['action_type'] == action_type)
    else:
        feat_builder.core_df = user_log
    # ==============================================
    print('计算用户和商铺的复购次数（复购率用UDF算）')
    if indicator is not None:
        for rebuy_times in rebuy_ranges:
            rebuy_udf = udf(lambda x: sum([cnt for cnt in Counter(x).values() if cnt > rebuy_times]),
                            IntegerType())
            rebuy_udf.__name__ = f"rebuy{rebuy_times}"
            feat_builder.buildCountFeatures('user_id', 'merchant_id', dummy=False,
                                            agg_funcs=[rebuy_udf], prefix=indicator)
            feat_builder.buildCountFeatures('merchant_id', 'user_id', dummy=False,
                                            agg_funcs=[rebuy_udf], prefix=indicator)
    # =============================================
    print('【商家】与用户的【年龄，性别】两个特征的交互')
    for pk in merchant_item_ids:
        feat_builder.buildCountFeatures(pk, user_feats, prefix=indicator,
                                        agg_funcs=['mean', 'max', 'min', median_udf, 'std', 'var', 'nunique'])
    # =============================================
    print('构造开始终止时间特征')
    for pk in core_ids + [core_ids]:
        feat_builder.buildCountFeatures(pk, "time_stamp_int", dummy=False, agg_funcs=["min", "max"],
                                        prefix=indicator)
    # =============================================
    print('对频率分布的统计特征')
    for pk in core_ids:
        target = [id_ for id_ in core_ids + item_ids if id_ != pk]
        feat_builder.buildCountFeatures(
            pk, target, dummy=False,
            multi_out_agg_funcs=[(freq_stat_info_names, freq_stat_info)],
            prefix=indicator)
    feat_builder.buildCountFeatures(
        core_ids, item_ids, dummy=False,
        multi_out_agg_funcs=[(freq_stat_info_names, freq_stat_info)],
        prefix=indicator)
    # =============================================
    print('【商家，商品，品牌，类别】与多少【用户】交互过（去重）')
    for pk in merchant_item_ids:
        feat_builder.buildCountFeatures(pk, 'user_id', dummy=False, agg_funcs=['nunique'], prefix=indicator)
    # =============================================
    print('【用户】与多少【商家，商品，品牌，类别】交互过（去重）')
    feat_builder.buildCountFeatures('user_id', merchant_item_ids, dummy=False, agg_funcs=['nunique'], prefix=indicator)
    print('【商家】,【用户，商品】与多少【商品，品牌，类别】交互过（去重）')
    for pk in ['merchant_id'] + [core_ids]:
        feat_builder.buildCountFeatures(pk, item_ids, dummy=False, agg_funcs=['nunique'],
                                        prefix=indicator)
    # =============================================
    if indicator is None:
        print('【用户，商家，商品，品牌，类别, 。。。】的【action_type】统计 （行为比例）')
        for pk in ['user_id', 'merchant_id'] + [core_ids]:
            feat_builder.buildCountFeatures(pk, 'action_type', agg_funcs=['nunique'], prefix=indicator)
    # =============================================
    print('【用户，商家，【用户，商家】】每个【月，星期】的互动次数,  持续时间跨度')
    for pk in ['user_id', 'merchant_id'] + [core_ids]:
        feat_builder.buildCountFeatures(pk, ['month', 'weekday'], agg_funcs=['nunique'], prefix=indicator)
        feat_builder.buildCountFeatures(pk, ['time_stamp'], dummy=False, agg_funcs=[timediff_udf], prefix=indicator)
    # =============================================
    print('最喜欢特征')
    all_features = ['user_id'] + ['month', 'weekday'] + merchant_item_ids
    if indicator is None:
        all_features.append('action_type')
    for feat_a in ['user_id', 'merchant_id']:
        targets = [feat_b for feat_b in all_features if feat_b != feat_a]
        feat_builder.buildCountFeatures(feat_a, targets, dummy=False, agg_funcs=[mostlike_udf], prefix=indicator)
    prefix = ""
    if indicator is not None:
        prefix = f"{indicator}-"
    print('用户在商铺的出现比例, 以及相反')
    feat_builder.addOperateFeatures(f'{prefix}users_div_merchants',
                                    f"lambda x: x['{prefix}user_id-cnt'] / x['{prefix}merchant_id-cnt']")
    feat_builder.addOperateFeatures(f'{prefix}merchants_div_users',
                                    f"lambda x: x['{prefix}user_id-cnt'] / x['{prefix}merchant_id-cnt']")
    print('用户和商铺的复购率')
    if indicator:
        for rebuy_times in rebuy_ranges:
            feat_builder.addOperateFeatures(f'{prefix}user_rebuy{rebuy_times}_ratio',
                                            f"lambda x: x['{prefix}user_id-merchant_id-rebuy{rebuy_times}'] / x['{prefix}user_id-cnt']")
            feat_builder.addOperateFeatures(f'{prefix}merchant_rebuy{rebuy_times}_ratio',
                                            f"lambda x: x['{prefix}merchant_id-user_id-rebuy{rebuy_times}'] / x['{prefix}merchant_id-cnt']")
    print('finish', indicator)

feat_builder.reduce_mem_usage()
del feat_builder.core_df
dump(feat_builder, "data/feat_builder.pkl")
# 打印出来的总特征数不准，因为有些主键对应的表用不上
print("总特征数：", feat_builder.n_features)
