#coding:utf-8

import os

import pandas as pd


def get_file_dict(file_dir, postfix, specify_dir_depth=0, key_remain_length=1):
    '''
    获取指定目录下指定后缀的文件字典
    :param file_dir: 文件目录
    :param postfix: 文件后缀
    :param specify_dir_depth: 指定目录深度，0表示不指定
    :param key_remain_length: 关键目录剩余层数
    :return: 文件字典
    '''
    file_dict = {}
    for root, _, files in os.walk(file_dir):
        if specify_dir_depth == 0 or root.count(os.sep) == file_dir.count(os.sep) + specify_dir_depth:
            for filename in files:
                if filename.endswith(postfix):
                    key_parts = root.split(os.sep)[-key_remain_length:]
                    key = os.path.join(*key_parts)
                    if key not in file_dict:
                        file_dict[key] = []
                    file_dict[key].append(filename)
    return file_dict

def get_baseline_config_dict():
    baseline_path = os.path.join(os.path.dirname(__file__), '..', '..', 'baselines')
    # 遍历baselines目录下的所有.py文件，组织为dict，key为目录名，value为文件名。只遍历第一层子目录。
    baseline_config_dict = get_file_dict(baseline_path, '.py', specify_dir_depth=1, key_remain_length=1)
    return baseline_config_dict


def get_ckpt_config_dict():
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
    ckpt_config_dict = get_file_dict(ckpt_path, '.pt', specify_dir_depth=3, key_remain_length=3)
    return ckpt_config_dict

def load_dataframe(input_data: list):
    df = pd.DataFrame(input_data)
    df_index = pd.to_datetime(df[0].values, format='%Y-%m-%d %H:%M:%S').to_numpy()
    df = df[df.columns[1:]]
    df.index = pd.Index(df_index)
    df = df.astype('float32')
    return df

if __name__ == '__main__':
    # d = get_baseline_config_dict()
    d = get_ckpt_config_dict()
    for k, v in d.items():
        print(f'{k}: {v}')
