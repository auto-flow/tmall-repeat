#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-21
# @Contact    : qichun.tang@bupt.edu.cn
import warnings
from typing import List, Dict, Union, Tuple, Callable, Optional

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame


def get_dummies(df, columns, prefix=None):
    if not isinstance(columns, list):
        columns = [columns]
    for column in columns:
        dst_df = df.select(column).distinct().toPandas()
        dst_df = dst_df.sort_values(by=column)
        for dst in dst_df[column]:
            if prefix is None:
                col_name = f"{column}_{dst}"
            else:
                col_name = f"{prefix}_{dst}"
            df = df.withColumn(col_name, F.when(df[column] == dst, 1).otherwise(0))
    return df

# 尽量能和pandas的规则对应上
window_func = {
    "max": F.max,
    "min": F.min,
    "mean": F.avg,
    "avg": F.avg,
    "count": F.count,
    "nunique": F.countDistinct,
    "var": F.var_samp,
    "std": F.stddev,
}

from utils import reduce_mem_usage

warnings.filterwarnings("ignore")


class FeaturesBuilder():
    def __init__(self, core_df):
        self.core_df: DataFrame = core_df
        self.pk2df: Dict[str, DataFrame] = {}  # primaryKeyToDataFrame
        self.op_feats: List[Tuple[str, Callable]] = []

    @property
    def n_features(self):
        # todo
        res = 0
        for pk, df in self.pk2df.items():
            res += df.shape[1] - len(pk)
        res += len(self.op_feats)
        return res

    def reduce_mem_usage(self):
        # todo
        for pk in self.pk2df:
            self.pk2df[pk] = reduce_mem_usage(self.pk2df[pk])

    def buildCountFeatures(
            self,
            primaryKey: Union[List[str], str],
            countValues: Union[List[str], str, None],
            countPK=True,
            dummy=True,
            ratio=True,
            agg_funcs=None,  # 注意， 如果countValues为离散特征或计数特征请谨慎使用，所以默认为None
            multi_out_agg_funcs: Optional[List[Tuple[List[str], Callable]]] = None,
            prefix=None
            # descriptions=None # 如果不为空，长度需要与countValues一致
    ):
        if isinstance(primaryKey, str):
            primaryKey = [primaryKey]
        if isinstance(countValues, str):
            countValues = [countValues]
        # 如果不存在主键对应的DF，创建新的
        t_pk = tuple(primaryKey)
        if t_pk not in self.pk2df:
            df = self.core_df[primaryKey].drop_duplicates().sort(primaryKey)
            self.pk2df[t_pk] = df
        # 主键列名
        pk_col = "-".join(primaryKey)
        if prefix:
            pk_col = f"{prefix}-{pk_col}"
        # 根据规则对参数进行校验
        if not countValues:
            dummy = False
            agg_funcs = None
        if dummy == False or countPK == False:
            ratio = False
        # 先对主键进行统计
        pk_cnt_col = f"{pk_col}-cnt"
        if countPK and pk_cnt_col not in self.pk2df[t_pk].columns:
            pk_cnt_df = self.core_df.groupby(primaryKey).count().withColumnRenamed("count", pk_cnt_col)
            self.pk2df[t_pk] = self.pk2df[t_pk].join(pk_cnt_df, how='left', on=primaryKey)
        # 对countValues进行处理
        if not countValues:
            countValues = []
        # for循环，对每个要计算的列进行统计
        for countValue in countValues:
            # 共现列名（fixme: 在dummy中会被删除）
            pk_val_col = f"{pk_col}-{countValue}"
            # 对聚集函数进行处理
            agg_args = []
            if agg_funcs is not None:
                for agg_func in agg_funcs:
                    if isinstance(agg_func, str):
                        agg_func_ = (window_func[agg_func])
                        agg_col_ = (agg_func)
                        agg_obj_ = (countValue)
                    else:
                        agg_func_ = (agg_func)
                        agg_col_ = (agg_func.__name__)
                        agg_obj_ = (F.collect_list(countValue))
                    new_name = f"{pk_val_col}-{agg_col_}"
                    agg_args.append(agg_func_(agg_obj_).alias(new_name))

            if agg_args:
                df_agg = self.core_df.groupby(primaryKey).agg(*agg_args)
                # 将除0产生的nan替换为0
                df_agg = df_agg.fillna(0)
                self.pk2df[t_pk] = self.pk2df[t_pk].join(df_agg, how='left', on=primaryKey)
            # if multi_out_agg_funcs:
            #     for names, func in multi_out_agg_funcs:
            #         df_mo_agg = self.core_df.groupby(primaryKey).agg({countValue: func}).reset_index()
            #         for i, name in enumerate(names):
            #             df_mo_agg[f"{pk_val_col}-{name}"] = df_mo_agg[countValue].apply(lambda x: x[i])
            #         df_mo_agg.pop(countValue)
            dummy_columns = []
            if dummy:
                # todo
                # 对values计数，得到dummy特征
                pk_val_cnt_df = self.core_df.groupby(primaryKey + [countValue]).count(). \
                    withColumnRenamed("count", pk_val_col)
                pk_cnt_df_dummy = get_dummies(pk_val_cnt_df, columns=[countValue], prefix=pk_val_col)
                dummy_columns = pk_cnt_df_dummy.columns[len(primaryKey) + 2:]  # pk , countValue , pk_val_col
                for column in dummy_columns:
                    pk_cnt_df_dummy = pk_cnt_df_dummy.withColumn(
                        column,
                        pk_cnt_df_dummy[column] * pk_cnt_df_dummy[pk_val_col]
                    )
                pk_cnt_df_dummy = pk_cnt_df_dummy.drop(countValue)
                pk_cnt_df_dummy = pk_cnt_df_dummy.drop(pk_val_col)
                columns = pk_cnt_df_dummy.columns
                pk_cnt_df_dummy = pk_cnt_df_dummy.groupby(primaryKey).sum()
                for pk in primaryKey: # 删掉原主键（新增了sum(pk)），呆的一笔
                    pk_cnt_df_dummy = pk_cnt_df_dummy.drop(pk)
                cur_cols = pk_cnt_df_dummy.columns
                # todo:  更好的重命名方法
                for cur, new in zip(cur_cols, columns):
                    pk_cnt_df_dummy = pk_cnt_df_dummy.withColumnRenamed(cur, new)
                self.pk2df[t_pk] = self.pk2df[t_pk].join(pk_cnt_df_dummy, how='left', on=primaryKey)
            if ratio and dummy_columns:
                df = self.pk2df[t_pk]
                ratio_columns = [f"{dummy_column}-ratio" for dummy_column in dummy_columns]
                for ratio_column, dummy_column in zip(ratio_columns, dummy_columns):
                    # 新建一个ratio columns
                    df = df.withColumn(
                        ratio_column,
                        df[dummy_column] / df[pk_cnt_col]
                    )
                    # 将除0产生的nan替换为0
                self.pk2df[t_pk] = df
        self.pk2df[t_pk] = self.pk2df[t_pk].fillna(0)
        self.pk2df[t_pk].cache()  # 先缓存再transform
        self.pk2df[t_pk].take(1)  # dummy transform

    def addOperateFeatures(
            self,
            new_feature_name: str,
            df_apply_func: Union[Callable, str]
    ):
        # 记得处理nan
        self.op_feats.append([new_feature_name, df_apply_func])

    def outputFeatures(self, base_df: pd.DataFrame, apply_op=True):
        # todo
        df = base_df
        pk_list = list(self.pk2df.keys())
        pk_list.sort()
        for pk in pk_list:
            df = df.merge(self.pk2df[pk], 'left', on=pk)
        if apply_op:
            self.applyOperateFeatures(df)
        return df

    def applyOperateFeatures(self, base_df: pd.DataFrame):
        for name, func in self.op_feats:
            if isinstance(func, str):
                func = eval(func)
            base_df[name] = func(base_df)
        return base_df
