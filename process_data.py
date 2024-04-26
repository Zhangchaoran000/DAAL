
import pickle
import numpy as np
import torch
import random
from transformers import BertModel, BertTokenizer
from affect_features.building_features import manual_features
import pandas as pd

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)


class Vocab:
    UNK = '[UNK]'
    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                 w = word.strip('\n')
                 self.stoi[w] = i
                 self.itos.append(w)
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)

def build_vocab(vocab_path):
    '''
    利用Vocab得到Bert的词汇表
    :param vocab_path:
    :return:
    '''
    return Vocab(vocab_path)

def add_event_label(filepath_list):
    # 初始化的时候给全部文件添加一列事件标签列
    for i, filepath in enumerate(filepath_list):
        insert_list = []
        df = pd.read_csv(filepath, encoding='unicode_escape')
        for index, row in df.iterrows():  # 遍历数据表 , 插入类别标签i
            insert_list.append(i)
        df['event_label'] = insert_list
        df.to_csv(filepath, index=False)
    pass

def add_if_marked_label(filepath_list,mark):
    # 1是只这条新闻被标注过，0是没有被标注过
    for i, filepath in enumerate(filepath_list):
        insert_list = []
        df = pd.read_csv(filepath, encoding='unicode_escape')
        for index, row in df.iterrows():  # 遍历数据表 , 插入类别标签i
            insert_list.append(mark)
        df['if_marked_label'] = insert_list
        df.to_csv(filepath, index=False)
    pass

def modify_if_marked_label(add_data):
    # 相当于把目标域，选择出来的有标记的数据的mark标记位置为1, add_data即为每次选择出来的数据, 类型是dataframe
    add_data['if_marked_label'] = add_data['if_marked_label'].map({0 : 1})
    print("已经完成了取样", add_data.head())
    pass

def read_data(filepath):
    # 读取文件，转化成为dataframe,并把标签列转为str便于数据清洗
    df = pd.read_csv(filepath, encoding='unicode_escape',dtype={'label': str})
    # 删除标签的缺省值
    df = df.dropna(subset=["label"])

    # 删除label中 不等于 0/1的行
    d = df[(df['label'] != '0') & (df['label'] != '1') & (df['label'] != '1.0') & (df['label'] != '0.0')]       # d既是要删除的数据

    df = df.drop(d.index)
    # 去重
    df = df.drop_duplicates()
    #df_clear = df.drop(df[df['label'] != 0].index)
    return df

# 构建自己训练材料的字典
class LoadSingleSentenceClassificationDataset:
    def __init__(self,
                 vocab_path = "./bert_model/vocab.txt",         # Bert词汇表的文件
                 tokenizer = None,
                 batch_size=64,
                 max_sen_len=54,                 # 以每个batch中最长样本长度为标准
                 max_position_embeddings=512,
                 pad_index=0,
                 ):
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = self.vocab['[PAD]']
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.batch_size = batch_size
        self.max_position_embeddings = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len

    def data_process(self, filepath_list, flag):
        # 从五个文件中先将数据全部取出来，然后统一处理，封装成列表，再把列表划分，返回一个一个data
        global df_ter
        global df_test
        # global df_train
        # global df_test
        if flag == "train1":
        # 4个源域数据 + 目标域数据的0.1
            for i, filepath in enumerate(filepath_list):
                if i == 0:
                    df_tmp = read_data(filepath)
                    df_train = df_tmp
                else:
                    df_tmp = read_data(filepath)
                    df_train = pd.concat([df_train, df_tmp], axis=0)
            print("源域的全部数据长度：", len(df_train))
            df_train = pd.concat([df_train, df_ter], axis=0)                 # 最后再拼接上抽出的目标与数据
            print("train文件最后", len(df_train))
            df_train = df_train.reset_index(drop=True)
            df_train.to_excel('./tmp_data/process_data/train_data.xlsx', index=False)
            return df_train
        elif flag == "train2":
            df_train = pd.read_excel('./tmp_data/process_data/train_data.xlsx')
            df_test = pd.read_excel('./tmp_data/process_data/test_data.xlsx')
            df_pool = pd.read_excel('./tmp_data/process_data/pool_data.xlsx')
            df_train2 = pd.concat([df_train, df_pool], axis=0)
            df_train2 = df_train2.reset_index(drop=True)
            print("train2文件最后有", len(df_train2))
            df_train2.to_excel('./tmp_data/process_data/train2_data.xlsx', index=False)
            return df_train2
        elif flag == "pool":
            # 相当于目标域的 1-0.25-0.1 = 0.65 作为候选选择数据
            df_pool = []
            for i, filepath in enumerate(filepath_list):
                if i == 0:
                    df_tmp = read_data(filepath)
                    df_pool = df_tmp
                else:
                    df_tmp = read_data(filepath)
                    df_pool = pd.concat([df_pool, df_tmp], axis=0)
                df_pool = pd.concat([df_pool, df_test, df_test]).drop_duplicates(keep=False)  # 这一行相当于从目标域数据集中删去固定的0.25的数据
                df_ter = df_pool.sample(frac=0.25, random_state=123, axis=0)                  # 原来是0.125
                df_pool = pd.concat([df_pool, df_ter, df_ter]).drop_duplicates(keep=False)
                modify_if_marked_label(df_ter)          # 这里要先删除，再调整，不然找不到交集
                df_pool = df_pool.reset_index(drop=True)
                print("pool文件最后有", len(df_pool))
                df_pool.to_excel('./tmp_data/process_data/pool_data.xlsx', index=False)
                return df_pool
        elif flag == "test":
            df_test = []
        # 从目标域中拿到0.25的初始数据集
            for i, filepath in enumerate(filepath_list):
                if i == 0:
                    df_tmp = read_data(filepath)
                    df_test = df_tmp
                else:
                    df_tmp = read_data(filepath)
                    df_test = pd.concat([df_test, df_tmp], axis=0)
            print("目标域的全部数据长度：", len(df_test))
            # df_ter = df_test.sample(frac=0.1, random_state=123, axis=0)                                    # 取出来样本
            # df_test = pd.concat([df_test, df_ter, df_ter]).drop_duplicates(keep=False)   # 删去后得到真正的test样本
            df_test = df_test.sample(frac=0.2, random_state=123, axis=0)                  # 这里先抽出来0.25的样本作为不动的测试集
            print("test文件最后有", len(df_test))                                            # 再把样本的标记值给改掉
            df_test = df_test.reset_index(drop=True)
            df_test.to_excel('./tmp_data/process_data/test_data.xlsx', index=False)
            return df_test

    def to_bert_input(self, df, df_index, flag):
        text = []
        mask = []
        eve_label = []
        label = []
        if_marked_label = []
        affection = []
        max_len = 0
        affection = self.to_BiGRU_input(df, flag)
        for i in range(0,len(df)):
            # 拿到文本数据 和 标签 和 事件类别标签 和 是否有标注的标签
            s, l, e, il = df.iloc[i,2], df.iloc[i,1], df.iloc[i,3],df.iloc[i,4]
            # 构建每个句子的CLS + 词汇对应数字 + SEP，一条构成了tmp
            tmp = [self.CLS_IDX]
            tmp += [self.vocab[token] for token in self.tokenizer.tokenize(s)]
            # 长截短补
            if len(tmp) > self.max_sen_len - 1:                 # 31
                tmp = tmp[:(self.max_sen_len - 1)]          # 此时有31个字符向量
                tmp += [self.SEP_IDX]                       # 此时正好够32个
            else:
                tmp += [self.SEP_IDX]                       # 先补上sep，再补pad
                tmp = tmp + [self.PAD_IDX for _ in range(self.max_sen_len - len(tmp))]
            attn_mask = [1 if num != 0 else 0 for num in tmp]       # 注意力机制编码 [1,1,1,0,0,0]
            tensor_ = torch.tensor(tmp, dtype=torch.long)           # 文本信息
            m = torch.tensor(attn_mask, dtype=torch.long)           # 注意力信息
            text.append(tensor_)
            mask.append(m)

            # l = torch.tensor(int(float(l)), dtype=torch.long)              # 加入标签
            l = int(float(l))
            l_numpy = np.array(l)
            l_tensor = torch.from_numpy(l_numpy).long()
            label.append(l_tensor)

            # event_label = torch.tensor(int(e), dtype=torch.long)          # 加入类别信息
            e = int(e)
            e_numpy = np.array(e)
            e_tensor = torch.from_numpy(e_numpy).long()
            eve_label.append(e_tensor)

            #il = torch.tensor(int(il), dtype=torch.long)                  # 加入是否有标注的标签
            il = int(il)
            il_numpy = np.array(il)
            il_tensor = torch.from_numpy(il_numpy).long()
            if_marked_label.append(il_tensor)

            max_len = max(max_len, tensor_.size(0))

        if flag == "source":
            df = pd.read_excel('./tmp_data/process_data/train_data.xlsx')
            df.insert(loc=2, column='affection', value=affection)
            df.to_excel('./tmp_data/process_data/train_data.xlsx', index=False)
        elif flag == "source_extend":
            df = pd.read_excel('./tmp_data/process_data/train2_data.xlsx')
            df.insert(loc=2, column='affection', value=affection)
            df.to_excel('./tmp_data/process_data/train2_data.xlsx', index=False)
        elif flag == "source_pool":
            df = pd.read_excel('./tmp_data/process_data/pool_data.xlsx')
            df.insert(loc=2, column='affection', value=affection)
            df.to_excel('./tmp_data/process_data/pool_data.xlsx', index=False)
        elif flag == "destination":
            df = pd.read_excel('./tmp_data/process_data/test_data.xlsx')
            df.insert(loc=2, column='affection', value=affection)
            df.to_excel('./tmp_data/process_data/test_data.xlsx', index=False)

        # text相当于是[[一句话中每个单词的对应数字]，[...]]
        # affection相当于是[ [[每个单词的24个向量],[第2个单词]] ,[[],[]] ]
        # 将affection转成tensor张量

        affection = np.array(affection)
        affection = torch.from_numpy(affection)  # dtype=torch.float64
        data = {
            "text": text,
            "mask": mask,
            "affection": affection,   # len(data["affection"]) = 4689, len(data["affection"][0]) = 8, len(data["affection"][0][0])=24
            "label": label,
            "event_label": eve_label,
            "if_marked_label": if_marked_label,
            "data_index": df_index
        }
        print(1)
        return data

    def to_BiGRU_input(self, df, flag):
        # 拿到每句话的文本后，将其对应转化为24维的向量,返回到上一个函数中，一起封装成为data形式
        print("先来拿到BiGRU的信息")
        print(flag)
        n_jobs = 1
        dataset = 'Rumor'
        segments_number = 8               # 不是10个单词，而是指 把每个句子分成了10段，每段用24维的向量来表示
        emo_rep = 'frequency'             # 分段数和先前的提取特征是不影响的，因为它是先提取了每个单词的特征，再综合成了一个句子几段
        content_features = []
        data = {"content": df['content'], "flag": flag }
        content_features = manual_features(n_jobs=n_jobs, path='idea1_features', model_name=dataset,
                                    segments_number=segments_number, emo_rep=emo_rep).transform(data)
        print(content_features.shape)                   # (4689, 10, 24)
        content_features = content_features.tolist()
        return content_features

    def to_bert_input_new(self, df, df_index):
        # 解决在代码训练的过程中，把文件里面的文本转换为bert，情感信息不要接着训练了
        text = []
        mask = []
        eve_label = []
        label = []
        if_marked_label = []
        affection_list = []
        max_len = 0
        affection = df['affection'].tolist()                 # 为什么这里直接没有affection这两列了
        for i in range(0,len(df)):
            # 拿到文本数据 和 标签 和 事件类别标签 和 是否有标注的标签
            s, l, e, il = df.iloc[i,3], df.iloc[i,1], df.iloc[i,4],df.iloc[i,5]
            # 构建每个句子的CLS + 词汇对应数字 + SEP，一条构成了tmp
            tmp = [self.CLS_IDX]
            tmp += [self.vocab[token] for token in self.tokenizer.tokenize(s)]

            # 长截短补
            if len(tmp) > self.max_sen_len - 1:                 # 31

                tmp = tmp[:(self.max_sen_len - 1)]          # 此时有31个字符向量

                tmp += [self.SEP_IDX]                       # 此时正好够32个

            else:
                tmp += [self.SEP_IDX]                       # 先补上sep，再补pad
                tmp = tmp + [self.PAD_IDX for _ in range(self.max_sen_len - len(tmp))]

            attn_mask = [1 if num != 0 else 0 for num in tmp]       # 注意力机制编码 [1,1,1,0,0,0]

            tensor_ = torch.tensor(tmp, dtype=torch.long)           # 文本信息

            m = torch.tensor(attn_mask, dtype=torch.long)           # 注意力信息

            text.append(tensor_)
            mask.append(m)

            # l = torch.tensor(int(float(l)), dtype=torch.long)              # 加入标签
            l = int(float(l))
            l_numpy = np.array(l)
            l_tensor = torch.from_numpy(l_numpy).long()
            label.append(l_tensor)

            # event_label = torch.tensor(int(e), dtype=torch.long)          # 加入类别信息
            e = int(e)
            e_numpy = np.array(e)
            e_tensor = torch.from_numpy(e_numpy).long()
            eve_label.append(e_tensor)

            #il = torch.tensor(int(il), dtype=torch.long)                  # 加入是否有标注的标签
            il = int(il)
            il_numpy = np.array(il)
            il_tensor = torch.from_numpy(il_numpy).long()
            if_marked_label.append(il_tensor)

            max_len = max(max_len, tensor_.size(0))

        for i in range(len(affection)):
            affection[i] = eval(affection[i])
            affection[i] = np.array(affection[i])
            affection[i] = torch.from_numpy(affection[i])

        # affection = np.array(affection)
        # affection = torch.from_numpy(affection)  # dtype=torch.float64      torch.Size([123, 8, 24])
        print(1)
        data = {
            "text": text,
            "mask": mask,
            "affection": affection,   # len(data["affection"]) = 4689, len(data["affection"][0]) = 8, len(data["affection"][0][0])=24
            "label": label,
            "event_label": eve_label,
            "if_marked_label": if_marked_label,
            "data_index": df_index
        }

        return data

if __name__ == '__main__':

    path = "./bert-model"
    #载入分词器
    tokenizer = BertTokenizer.from_pretrained(path)
    # 载入模型
    model = BertModel.from_pretrained(path)
    vocab_path = "./bert-model/vocab.txt"
    pre_path = "./rumor/"
    all_file_list = [pre_path + "charliehebdo.csv", pre_path + "ferguson.csv", \
                 pre_path + "germanwings-crash.csv", pre_path + "ottawashooting.csv", pre_path + "sydneysiege.csv"]
    # 第一个数据集实验
    # source_file_list = [pre_path + "charliehebdo.csv", pre_path + "ferguson.csv", \
    #              pre_path + "germanwings-crash.csv", pre_path + "ottawashooting.csv"]
    # destination_file_list = [pre_path + "sydneysiege.csv"]

    # 第二个数据集实验
    # source_file_list = [pre_path + "charliehebdo.csv", pre_path + "ferguson.csv", \
    #                     pre_path + "germanwings-crash.csv", pre_path + "sydneysiege.csv"]
    # destination_file_list = [pre_path + "ottawashooting.csv"]
    # 第三个数据集实验
    # source_file_list = [pre_path + "charliehebdo.csv", \
    #                     pre_path + "germanwings-crash.csv", pre_path + "ottawashooting.csv", pre_path + "sydneysiege.csv"]
    # destination_file_list = [pre_path + "ferguson.csv"]
    # 第四个数据集
    source_file_list = [pre_path + "ferguson.csv",\
                        pre_path + "germanwings-crash.csv", pre_path + "ottawashooting.csv", pre_path + "sydneysiege.csv"]
    destination_file_list = [pre_path + "charliehebdo.csv"]

    # 第五个数据集
    # source_file_list = [pre_path + "ferguson.csv",\
    #                     pre_path + "charliehebdo.csv", pre_path + "ottawashooting.csv", pre_path + "sydneysiege.csv"]
    # destination_file_list = [pre_path + "germanwings-crash.csv"]

    # 先添加 事件标签列
    # add_event_label(source_file_list)

    # 添加 哪些 数据是否被标记列
    add_if_marked_label(source_file_list, 1)
    add_if_marked_label(destination_file_list, 0)
    mydic = LoadSingleSentenceClassificationDataset(vocab_path, tokenizer)
    # 得到数据对应的词向量以及长度,相当于构建数据集
    df_ter = 0                          # df_ter 是在构建训练数据时，从目标域中抽取的数据
    df_test = 0
    # 先构建测试集数据
    print("开始构建dataframe数据集")
    # 构建测试集数据
    df_test = mydic.data_process(destination_file_list, "test")
    # 构建候选训练集数据
    df_pool = mydic.data_process(destination_file_list, "pool")
    # 再训练集数据1
    df_train1 = mydic.data_process(source_file_list, "train1")
    # 再训练集数据2
    df_train2 = mydic.data_process(all_file_list, "train2")

    # 统一拿到索引
    index_test = list(df_test.index)
    index_pool = list(df_pool.index)
    index_train1 = list(df_train1.index)
    index_train2 = list(df_train2.index)

    # 再把数据都转变成了Bert的输入
    source_data = mydic.to_bert_input(df_train1, index_train1, "source")
    source_pool_data = mydic.to_bert_input(df_pool, index_pool, "source_pool")
    source_extend_data = mydic.to_bert_input(df_train2, index_train2, "source_extend")
    destination_data = mydic.to_bert_input(df_test, index_test, "destination")

    # 把数据打包进入pkl文件里
    with open('./tmp_data/process_data/source_data.pkl', 'wb') as f:
        pickle.dump(source_data, f)
    with open('./tmp_data/process_data/source_pool_data.pkl', 'wb') as f:
        pickle.dump(source_pool_data, f)
    with open('./tmp_data/process_data/source_extend_data.pkl', 'wb') as f:
        pickle.dump(source_extend_data, f)
    with open('./tmp_data/process_data/destination_data.pkl', 'wb') as f:
        pickle.dump(destination_data, f)



