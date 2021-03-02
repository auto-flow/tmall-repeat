#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-03-02
# @Contact    : qichun.tang@bupt.edu.cn
import datetime
import warnings

import gc
import pandas as pd

from tmall.utils import reduce_mem_usage, get_data_path

warnings.filterwarnings("ignore")


def read_csv(file_name, num_rows):
    return pd.read_csv(file_name, nrows=num_rows)

data_path=get_data_path()

# num_rows = 200 * 10000  # 1000条测试代码使用
num_rows = None

# 读入数据，内存压缩
train_file = f'{data_path}/data_format1/train_format1.csv'
test_file = f'{data_path}/data_format1/test_format1.csv'

user_info_file = f'{data_path}/data_format1/user_info_format1.csv'
user_log_file = f'{data_path}/data_format1/user_log_format1.csv'

train_data = reduce_mem_usage(read_csv(train_file, num_rows))
test_data = reduce_mem_usage(read_csv(test_file, num_rows))
user_info = reduce_mem_usage(read_csv(user_info_file, num_rows))
# 处理缺失值
user_info['age_range'][pd.isna(user_info['age_range'])] = 0
user_info['gender'][pd.isna(user_info['gender'])] = 2
user_info[['age_range', 'gender']] = user_info[['age_range', 'gender']].astype('int8')
user_info.to_pickle('user_info.pkl')
user_log = reduce_mem_usage(read_csv(user_log_file, num_rows))
user_log.rename(columns={'seller': 'merchant_id'})

del test_data['prob']
all_data = train_data.append(test_data)
all_data = all_data.merge(user_info, 'left', on=['user_id'], how='left')
gc.collect()
# seller_id 与 训练测试集不匹配
user_log.rename(columns={'seller_id': 'merchant_id'}, inplace=True)
# user_log 的 brand_id存在空值，用0填充
user_log['brand_id'][pd.isna(user_log['brand_id'])] = 0
user_log['brand_id'] = user_log['brand_id'].astype('int16')
# 引入用户画像信息到用户日志中
user_log = pd.merge(user_info, 'left', user_log, on='user_id')
# 把月和天的信息抽取出来
# pandas做这件事特别慢
# user_log['month'] = user_log['time_stamp'].apply(lambda x: int(f"{x:04d}"[:2])).astype('int8')
# user_log['day'] = user_log['time_stamp'].apply(lambda x: int(f"{x:04d}"[2:])).astype('int8')
user_log['month'] = (user_log['time_stamp'] // 100).astype('int8')
user_log['day'] = (user_log['time_stamp'] % 100).astype('int8')
# 查一下是星期几的
user_log['time_stamp'] = user_log['time_stamp'].apply(lambda x: datetime.datetime.strptime(f'2016{x:04d}', '%Y%m%d'))
user_log['weekday'] = user_log['time_stamp'].apply(lambda x: x.weekday()).astype('int8')
# 加入标签特征
train = pd.read_csv('data_format1/train_format1.csv')
user_log = user_log.merge(train, 'left', ['user_id', 'merchant_id'])
user_log['label'].fillna(-1, inplace=True)
user_log['time_stamp_int']=(user_log['month']*100+user_log['day']).astype('int16')
# 保存
user_log.to_pickle(f'{data_path}/user_log.pkl')
