# -*- codeing = utf-8 -*-
# @Time : 2022-09-01 11:20
# @Author : 张超然
# @File ： loading_emotional_lexicons.py
# @Software: PyCharm

import warnings, operator
warnings.filterwarnings("ignore")
import pandas as pd
from os.path import join
import os
current_path = os.path.dirname(__file__)
class emotional_lexicons:

    def __init__(self, path):
        self.lexicons_path = path

        # NRC, plutchik
        self.nrc = pd.read_csv(current_path + r'\\nrc.txt', sep='\t', names=["word", "emotion", "association"])
        self.nrc = self.nrc.pivot(index='word', columns='emotion', values='association').reset_index()
        del self.nrc['positive']
        del self.nrc['negative']

        # NRC (intensity), plutchik
        self.nrc_intensity = pd.read_csv(current_path + r'\\NRC-AffectIntensity-Lexicon.txt', sep='\t', names=["word", "score", "emotion"], skiprows=1)
        self.nrc_intensity = self.nrc_intensity.pivot(index='word', columns='emotion', values='score').reset_index()
        self.nrc_intensity.fillna(value=0, inplace=True)



    def frequency(self, sentence):
        words = []
        # print(self.nrc.head())        # 这里是 词典表，即每个单词 的8种情绪（0，1表示）
        for word in sentence:
            try:
                result = self.nrc[self.nrc.word == str(word)].values.tolist()[0][1:]
                # 如果句子中的单词存在于该词典中，即返回[00001000]
            except:
                result = [0, 0, 0, 0, 0, 0, 0, 0]
                pass
            words.append(result)

        if len(words) == 0:
            words.append([0, 0, 0, 0, 0, 0, 0, 0])
        return words

    def intensity(self, sentence):
        words = []
        for word in sentence:
            try:
                result = self.nrc_intensity[self.nrc_intensity.word == str(word)].values.tolist()[0][1:]
            except:
                result = [0, 0, 0, 0]
                pass
            words.append(result)
        if len(words) == 0:
            words.append([0, 0, 0, 0])
        return words

if __name__ == '__main__':
    print("当前路径 -  %s" % os.getcwd())
    current_path = os.path.dirname(__file__)
    print(current_path)
    path = 'nrc.txt'
    f = open(path)
    line = f.readline().strip()  # 读取第一行
    f.close()
    df = pd.read_csv(path,  sep='\t', names=["word", "emotion", "association"])
    print(df.head())