# -*- codeing = utf-8 -*-
# @Time : 2022-09-01 11:00
# @Author : 张超然
# @File : building_features.py
# @Software: PyCharm
import warnings
warnings.filterwarnings("ignore")

import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import exists
from os.path import join
from joblib import Parallel, delayed

from idea1_features.emotional.loading_emotional_lexicons import emotional_lexicons
from idea1_features.sentiment.loading_sentiment_lexicons import sentiment_lexicons
from idea1_features.morality.morality import MORALITY_class
from idea1_features.imageability.imageability import imageability_class
from idea1_features.hyperbolic.hyperbolic import hyperbolic_class
import os
# config
np.random.seed(0)
tqdm.pandas()
current_path = os.path.dirname(__file__)
# 构造普通的清理函数
def clean_regex(text, keep_dot=False, split_text=False):

    try:
        text = re.sub(r'((http|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=;%&:/~+#-]*[\w@?^=%&;:/~+#-])?)', ' ',
                      text)
        text = re.sub(r'[^ ]+\.com', ' ', text)
        text = re.sub(r'(\d{1,},)?\d{1,}(\.\d{1,})?', '', text)
        text = re.sub(r'’', '\'', text)
        text = re.sub(r'[^A-Za-z\'. ]', ' ', text)
        text = re.sub(r'\.', '. ', text)
        text = re.sub(r'\s{2,}', ' ', text)

        text = re.sub(r'(\.\s)+', '.', str(text).strip())
        text = re.sub(r'\.{2,}', '.', str(text).strip())
        text = re.sub(r'(?<!\w)([A-Z])\.', r'\1', text)

        text = re.sub(r'\'(?!\w{1,2}\s)', ' ', text)

        text = text.split('.')
        if keep_dot:
            text = ' '.join([sent.strip() + ' . ' for sent in text])
        else:
            text = ' '.join([sent.strip() for sent in text])

        text = text.lower()
        return text.split() if split_text else text
    except:
        text = 'empty text'
        return text.split() if split_text else text

# 自定义转换器transformer，fit函数是返回构造器本身，transform函数对传入的data做相应的处理
class append_split_3D(BaseEstimator, TransformerMixin):
    def __init__(self, segments_number=20, max_len=50, mode='append'):
        self.segments_number = segments_number
        self.max_len = max_len
        self.mode = mode
        self.appending_value = -5.123

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        if self.mode == 'append':
            self.max_len = self.max_len - data.shape[2]                    # shape【2】输出的是（3，20，8）的8?
            appending = np.full((data.shape[0], data.shape[1], self.max_len), self.appending_value)  # 都填充上 -5.123？
            new = np.concatenate([data, appending], axis=2)
            # print(data.shape)            # (3, 20, 8)
            # print(appending.shape)       # (3, 20, 42)
            # print(new.shape)             # (3, 20, 50)
            return new
        elif self.mode == 'split':
            tmp = []
            for item in range(0, data.shape[1], self.segments_number):
                tmp.append(data[:, item:(item + self.segments_number), :])

            tmp = [item[item != self.appending_value].reshape(data.shape[0], self.segments_number, -1) for item in tmp]
            # (3, 20, 8)
            new = np.concatenate(tmp, axis=2)
            return new
        else:
            print('Error: Mode value is not defined')
            exit(1)

class segmentation(BaseEstimator, TransformerMixin):

    def __init__(self, n_jobs=1, segments_number=20):
        self.n_jobs = n_jobs
        self.segments_number = segments_number

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        # print("segmentation", data
        out = []
        for sentence in data:
            tmp = np.array_split(sentence, self.segments_number)                          # 即把一个句子划分成几段
            tmp = [np.sum(item, axis=0) / sentence.shape[0] for item in tmp]              # 最高维度,每个单词对应的八个列分别相加/句子长度
            out.append(tmp)
        out = np.array(out)                 # (3,20,8)        3个句子，每个句子分为20段，每段都是8个情感维度
        return out

class segmentation_text(BaseEstimator, TransformerMixin):

    def __init__(self, n_jobs=1, segments_number=20):
        self.n_jobs = n_jobs
        self.segments_number = segments_number

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        data = Parallel(n_jobs=1, backend="multiprocessing", prefer="processes") \
            (delayed(clean_regex)(sentence, keep_dot=False, split_text=True) for sentence in tqdm(data, desc='Text Segmentation'))
        if isinstance(data, list):
            data = np.array([np.array(sent) for sent in data])
        out = []
        for sentence in data:
            try:
                tmp = np.array_split(sentence, self.segments_number)
                tmp = ' . '.join([' '.join(item.tolist()) for item in tmp])
            except:
                print()
            out.append(tmp)
        return out

# 返回句子的情感特征
class emotional_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1, model_name='', representation='frequency'):
        self.path = path
        self.n_jobs = n_jobs
        self.model_name = model_name
        self.representation = representation

    def fit(self, X, y=None):
        return self

    def error_representation(self):
        print('\n\nError: check the value of the variable "representation".')
        exit(1)

    def transform(self, data):
        # 拿到句子中的每个单词是否以及具体代表了哪种情感，用类似于one-hot来表示, 这里先把惊讶的情绪变成了两倍试试
        flag = data["flag"]
        data = data["content"]
        print(data.head())
        file_name = current_path + '/processed_files/features/emotional_feature111s_{}_{}_{}.npy'.format(self.model_name, self.representation, flag)
        print(file_name)
        if exists(file_name):
            features = np.load(file_name, allow_pickle=True).tolist()
        else:
            data = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")\
                (delayed(clean_regex)(sentence, False, True) for sentence in tqdm(data, desc='Cleaning text'))

            emo = emotional_lexicons(path=join(self.path, 'emotional'))
            loop = tqdm(data)
            loop.set_description('Building emotional_features ({})'.format(self.representation))
            # ['i', "don't", 'to', 'xsdf']
            # ['she', 'can', 'want', 'to', 'be', 'witt']
            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes") \
                (delayed(emo.frequency if self.representation == 'frequency' else emo.intensity if self.representation == 'intensity' else self.error_representation())
                 (sentence) for sentence in loop)

            features = [np.array(item) for item in features]
            print(111111111111111)
            print(features)
            # np.save(file_name, features)
        return features

class sentiment_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1, model_name=''):
        self.path = path
        self.n_jobs = n_jobs
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        flag = data["flag"]
        data = data["content"]
        file_name = current_path + '/processed_files/features/sentiment_features111_{}_{}.npy'.format(self.model_name, flag)
        if exists(file_name):
            features = np.load(file_name, allow_pickle=True).tolist()
        else:
            data = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes") \
                (delayed(clean_regex)(sentence, False, True) for sentence in tqdm(data, desc='Cleaning text'))

            senti = sentiment_lexicons(path=join(self.path, 'sentiment'))
            loop = tqdm(data)
            loop.set_description('Building sentiment_features')

            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(delayed(senti.score)(sentence) for sentence in loop)
            features = [np.array(item) for item in features]
            # np.save(file_name, features)
        return features

class morality_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1, model_name=''):
        self.path = path
        self.n_jobs = n_jobs
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        flag = data["flag"]
        data = data["content"]
        file_name = current_path + '/processed_files/features/morality_features111_{}_{}.npy'.format(self.model_name, flag)
        if exists(file_name):
            features = np.load(file_name, allow_pickle=True).tolist()
        else:
            data = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes") \
                (delayed(clean_regex)(sentence, False, True) for sentence in tqdm(data, desc='Cleaning text'))

            lex = MORALITY_class(path=join(self.path, 'morality'))
            loop = tqdm(data)
            loop.set_description('Building Morality_features')

            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(delayed(lex.score)(sentence) for sentence in loop)
            features = [np.array(item) for item in features]
            # np.save(file_name, features)
        return features

class imageability_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1, model_name=''):
        self.path = path
        self.n_jobs = n_jobs
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        flag = data["flag"]
        data = data["content"]
        file_name = current_path + '/processed_files/features/imageability_features111_{}_{}.npy'.format(self.model_name, flag)
        if exists(file_name):
            features = np.load(file_name, allow_pickle=True).tolist()
        else:
            data = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes") \
                (delayed(clean_regex)(sentence, False, True) for sentence in tqdm(data, desc='Cleaning text'))

            lex = imageability_class(path=join(self.path, 'imageability'))
            loop = tqdm(data)
            loop.set_description('Building Imageability_features')

            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(delayed(lex.score)(sentence) for sentence in loop)
            features = [np.array(item) for item in features]
            # np.save(file_name, features)
        return features

class hyperbolic_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1, model_name=''):
        self.path = path
        self.n_jobs = n_jobs
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        flag = data["flag"]
        data = data["content"]
        file_name = current_path + '/processed_files/features/hyperbolic_features111_{}_{}.npy'.format(self.model_name, flag)
        if exists(file_name):
            features = np.load(file_name, allow_pickle=True).tolist()
        else:
            data = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes") \
                (delayed(clean_regex)(sentence, False, True) for sentence in tqdm(data, desc='Cleaning text'))

            lex = hyperbolic_class(path=join(self.path, 'hyperbolic'))
            loop = tqdm(data)
            loop.set_description('Building Hyperbolic_features')

            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(delayed(lex.score)(sentence) for sentence in loop)
            features = [np.array(item) for item in features]
            # np.save(file_name, features)
        return features


def manual_features(path='', n_jobs=1, model_name='', segments_number=20, emo_rep='frequency'):
    print("进入到了管道")
    manual_feats = Pipeline([
        ('FeatureUnion', FeatureUnion([
            ('1', Pipeline([
                ('emotional_features', emotional_features(path=path, n_jobs=n_jobs, model_name=model_name, representation=emo_rep)),
                ('segmentation', segmentation(n_jobs=n_jobs, segments_number=segments_number)),
                ('append', append_split_3D(segments_number=segments_number, max_len=50, mode='append')),
            ])),
            ('2', Pipeline([
                ('sentiment_features', sentiment_features(path=path, n_jobs=n_jobs, model_name=model_name)),
                ('segmentation', segmentation(n_jobs=n_jobs, segments_number=segments_number)),
                ('append', append_split_3D(segments_number=segments_number, max_len=50, mode='append')),
            ])),
            ('3', Pipeline([
                ('morality_features', morality_features(path=path, n_jobs=n_jobs, model_name=model_name)),
                ('segmentation', segmentation(n_jobs=n_jobs, segments_number=segments_number)),
                ('append', append_split_3D(segments_number=segments_number, max_len=50, mode='append')),
            ])),
            ('4', Pipeline([
                ('imageability_features', imageability_features(path=path, n_jobs=n_jobs, model_name=model_name)),
                ('segmentation', segmentation(n_jobs=n_jobs, segments_number=segments_number)),
                ('append', append_split_3D(segments_number=segments_number, max_len=50, mode='append')),
            ])),
            ('5', Pipeline([
                ('hyperbolic_features', hyperbolic_features(path=path, n_jobs=n_jobs, model_name=model_name)),
                ('segmentation', segmentation(n_jobs=n_jobs, segments_number=segments_number)),
                ('append', append_split_3D(segments_number=segments_number, max_len=50, mode='append')),
            ])),
        ], n_jobs=1)),
        ('split', append_split_3D(segments_number=segments_number, max_len=50, mode='split'))
    ])
    return manual_feats


if __name__ == '__main__':
    # df = pd.DataFrame([{'content': "I don't to xsdf"},
    #                    {'content': "she can want to be witt"},
    #                    {'content': "I can abacus"}])
    # res = manual_features(n_jobs=4).fit_transform(df)
    # print(res.shape)
    # print(type(res))
    # print(res.tolist()[0])                     #    20 * 24
    # x = pd.Series(res.tolist())
    # print('')

    current_path = os.path.dirname(__file__)
    print(current_path)