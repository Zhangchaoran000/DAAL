
from torch.nn.functional import normalize

import pickle
import random
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from os.path import exists
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import BertTokenizer,BertModel,BertConfig
import numpy as np
import pandas as pd
import process_data as ProData
from Samper import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

df_pool_len = pd.read_excel('./tmp_data/process_data/pool_data.xlsx')
num = int(0.05 * len(df_pool_len))
count = len(df_pool_len)

class Rumor_Data(Dataset):
    # dataset类——创建适应任意模型的数据集接口
    def __init__(self, dataset):
        self.text = dataset['text']
        self.mask = dataset['mask']
        self.affection = dataset['affection']
        self.label = dataset['label']
        self.event_label = dataset['event_label']
        self.if_marked_label = dataset['if_marked_label']
        self.data_index = dataset['data_index']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.text[idx], self.mask[idx], self.affection[idx], self.label[idx], self.event_label[idx], self.if_marked_label[idx], self.data_index[idx]

class ReverseLayerF(Function):
    @staticmethod
    def forward(self, x):
        self.lambd = 1
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x):
    return ReverseLayerF.apply(x)


def to_np(x):
    return x.data.cpu().numpy()
class Config(object):
    """配置参数"""
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000
        self.num_epochs = 3
        self.batch_size = 32
        # 每句话处理成的长度(短填长切)
        self.pad_size = 64
        self.learning_rate = 3e-5
        # bert预训练模型的位置
        self.bert_path = './bert_model'
        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # bert隐藏层个数
        self.hidden_size = 768
        self.dropout = 0.3
        self.num_filters = 256
        # 卷积核在序列维度上的尺寸 = n-gram大小 卷积核总数量=filter_size*num_filters
        self.filter_size = (2, 3, 4)

class MyNet(nn.Module):
    def __init__(self, config):
        super(MyNet, self).__init__()
        # 相关参数设置
        self.batch_size = 16
        self.hidden_size = 64
        self.event_num = 5
        model_config = BertConfig.from_pretrained(config.bert_path, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(config.bert_path, config = model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)                      # 即防止过拟合
        self.convs = nn.ModuleList(
            # 输入通道数,输出通道数（卷积核数），卷积核维度
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_size]
        )

        ###  第一个分支  Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(config.hidden_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))


        ### 第二个分支   Domain Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(config.hidden_size, self.hidden_size))    # [batch]
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

        ### 第三个分支
        self.infer_discriminator = nn.Sequential()
        self.infer_discriminator.add_module('e_fc1', nn.Linear(768, self.hidden_size))
        self.infer_discriminator.add_module('e_relu1', nn.ReLU(True))
        self.infer_discriminator.add_module('e_fc2', nn.Linear(self.hidden_size, self.hidden_size))  # [32, 128]
        self.infer_discriminator.add_module('e_relu2', nn.ReLU(True))
        self.infer_discriminator.add_module('e_fc3', nn.Linear(self.hidden_size, 1))  # # [32, 1]
        self.infer_discriminator.add_module('e_softmax', nn.Sigmoid())

        ### 第四个分支  情感鉴别器
        self.gru = nn.GRU(24, 32, batch_first=True, bidirectional=True)   # [batchsize * [8 * 24]]*(24* 32*2)=[batchsize*8*64(32*2)]
        self.affection_discriminator = nn.Sequential()
        self.affection_discriminator.add_module('f_fc1', nn.Linear(128, 64))
        self.infer_discriminator.add_module('f_relu1', nn.ReLU(False))
        self.affection_discriminator.add_module('f_fc2', nn.Linear(64, 1))
        self.affection_discriminator.add_module('f_sigmoid', nn.Sigmoid())

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_size]
        )
        self.fc = nn.Linear(config.num_filters * len(config.filter_size), self.batch_size)
        self.fc2 = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()

    def conv_and_pool(self, x, conv):                                                 # 卷积核
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x

    def forward(self, x1, x2, x3, x4, x5, flag):
        if flag == 0:
            context1 = x1
            mask1 = x2
            context2 = x3
            mask2 = x4
            gru_inputs = x5
            outputs1 = self.bert(context1, attention_mask=mask1)       # shape[batch_size * hidden_size(768)]
            outputs2 = self.bert(context2, attention_mask=mask2)       # shape[batch_size * hidden_size(768)]

            cls1 = self.dropout(outputs1['pooler_output'])
            cls2 = self.dropout(outputs2['pooler_output'])
            # 1分支，二分类器，返回0-1之间
            score = self.class_classifier(cls1)  # shape [batch_size,2]

            # 2分支，领域鉴别器，softmax函数返回概率
            reverse_feature = grad_reverse(cls1)  # 梯度反转
            domain_output = self.domain_classifier(reverse_feature) # shape [batch_size, event_num]

            # 3分支
            lable_output = self.infer_discriminator(cls2) # shape [batch_size, 1]

            # 4分支
            embadding_out = outputs2['last_hidden_state']
            out = embadding_out.unsqueeze(1)
            out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) # [self.batch_size, 768]

            out = self.fc(out)

            gru_inputs = gru_inputs.float()
            gru_out, h = self.gru(gru_inputs, None)
            gru_out = gru_out[:, -1, :]

            mul_out = torch.mm(out, gru_out)
            mul_out = self.fc2(mul_out)
            lable_output2 = self.sig(mul_out)
            return score, domain_output, lable_output, lable_output2

        elif flag == 1:
            context1 = x1
            mask1 = x2
            outputs1 = self.bert(context1, attention_mask=mask1)  # shape[batch_size * hidden_size(768)]
            cls1 = self.dropout(outputs1['pooler_output'])
            score = self.class_classifier(cls1)
            return score
        elif flag == 3:
            context1 = x1
            mask1 = x2
            gru_inputs = x5
            outputs1 = self.bert(context1, attention_mask=mask1)  # shape[batch_size * hidden_size(768)]
            cls1 = self.dropout(outputs1['pooler_output'])
            lable_output1 = self.infer_discriminator(cls1)
            embadding_out = outputs1['last_hidden_state']
            out = embadding_out.unsqueeze(1)
            out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
            out = self.fc(out)

            gru_inputs = gru_inputs.float()
            gru_out, h = self.gru(gru_inputs, None)
            gru_out = gru_out[:, -1, :]
            mul_out = torch.mm(out, gru_out)
            mul_out = self.fc2(mul_out)
            lable_output2 = self.sig(mul_out)
            return lable_output1, lable_output2

        elif flag == 4:
            context1 = x1
            mask1 = x2
            outputs = self.bert(context1, attention_mask=mask1)
            return outputs

def modify_if_marked_label(add_data):
    # 相当于把目标域，选择出来的有标记的数据的mark标记位置为1, add_data即为每次选择出来的数据, 类型是dataframe
    add_data['if_marked_label'] = add_data['if_marked_label'].replace({0: 1})
    pass

def func1(amount,num):
    list1 = []
    for i in range(0,num-1):
        a = random.randint(0,amount)
        list1.append(a)
    list1.sort()
    list1.append(amount)

    list2 = []
    for i in range(len(list1)):
        if i == 0:
            b = list1[i]                # 第一段长度为第 1 个节点 - 0
        else:
            b = list1[i] - list1[i-1]   # 其余段为第 n 个节点 - 第 n-1 个节点
        list2.append(b)
    return list2
def load_data(flag, data_index):
    # 如果是初始化训练，第一次的话，就直接把数据加载进来了
    if len(data_index) == 0:
        if flag == "train1":
            data_path = './tmp_data/process_data/source_data.pkl'
            f = open(data_path, 'rb')
            W = pickle.load(f)
        elif flag == "train2":
            data_path = './tmp_data/process_data/source_extend_data.pkl'
            f = open(data_path, 'rb')
            W = pickle.load(f)
        elif flag == "test":
            data_path = './tmp_data/process_data/destination_data.pkl'
            f = open(data_path, 'rb')
            W = pickle.load(f)
        elif flag == "validate":
            data_path = './tmp_data/process_data/destination_data.pkl'
            f = open(data_path, 'rb')
            W = pickle.load(f)
        elif flag == "pool":
            data_path = './tmp_data/process_data/source_pool_data.pkl'
            f = open(data_path, 'rb')
            W = pickle.load(f)
    # 对于之后的任意次：2，3，4，5次
    else:
        print("data_index---info(len, max):", len(data_index), np.max(data_index))
        if flag == "fine_train":
            df = pd.read_excel('./tmp_data/process_data/pool_data.xlsx')  # 即完整的目标域数据  加载出test的字典张量数据
            print("pool_data文件的长度：", len(df))
            df_tmp = df.iloc[data_index]  # 这是在初始数据上，根据append_index(是之前文件的index)加进去和删除的数据
            modify_if_marked_label(df_tmp)

            df_source_extend = pd.read_excel('./tmp_data/process_data/train_data.xlsx')  # 源 + 0.1目标
            df_old_add = df_source_extend[df_source_extend['event_label'] == 3]             # 拿到初始化时候的目标域0.1数据
            df_new_train = pd.concat([df_tmp, df_old_add], axis=0)                          # 新的训练数据=0.1+0.05
            df_new_train = df_new_train.reset_index(drop=True)                              # 重置了索引
            df_new_train.to_excel('./tmp_data/process_data/new_train_data.xlsx', index=False)  # 这里现在是 0.1的目标+之后选择出来的0.05的目标
            print("new_train_data文件的长度：", len(df_new_train))
            index_train = list(df_new_train.index)                                          # 1...n
            new_train_data = mydict.to_bert_input_new(df_new_train, index_train)
            W = new_train_data
        elif flag == "fine_pool":
            df_pool = pd.read_excel('./tmp_data/process_data/pool_data.xlsx')
            df_new_pool = df_pool.drop(df_pool.index[data_index])  # 训练备选数据删去后index后得到真正的pool样本 这个索引应该还是原来的索引
            df_new_pool = df_new_pool.reset_index(drop=True)        # 重置了索引
            df_new_pool.to_excel('./tmp_data/process_data/new_pool_data.xlsx', index=False)    # 这是删去0.05的目标
            index_pool = list(df_new_pool.index)                    # 返回回去的已经是重置之后的索引了
            new_test_data = mydict.to_bert_input_new(df_new_pool, index_pool)
            W = new_test_data
        elif flag == "fine_train_after":
            df1 = pd.read_excel('./tmp_data/process_data/new_pool_data.xlsx')
            df_train_after = df1.iloc[data_index]                     # 选出来
            modify_if_marked_label(df_train_after)
            df_train_before = pd.read_excel('./tmp_data/process_data/new_train_data.xlsx')
            df_train_after = pd.concat([df_train_before, df_train_after], axis=0)
            df_train_after = df_train_after.reset_index(drop=True)
            print("new_train_data文件的长度：", len(df_train_after))
            df_train_after.to_excel('./tmp_data/process_data/new_train_data.xlsx', index=False)
            index_train = list(df_train_after.index)
            new_train_after_data = mydict.to_bert_input_new(df_train_after, index_train)
            W = new_train_after_data
        elif flag == "fine_pool_after":
            df2 = pd.read_excel('./tmp_data/process_data/new_pool_data.xlsx')
            print("new_test_data文件的长度：", len(df2))
            df_pool_after = df2.drop(df2.index[data_index])           # 删除掉
            df_pool_after = df_pool_after.reset_index(drop=True)

            df_pool_after.to_excel('./tmp_data/process_data/new_pool_data.xlsx', index=False)  # 这里现在是 0.1的目标+之后选择出来的0.05的目标
            index_pool = list(df_pool_after.index)
            new_pool_after_data = mydict.to_bert_input_new(df_pool_after, index_pool)
            W = new_pool_after_data

    return W

path = "./bert_model"
tokenizer = BertTokenizer.from_pretrained(path)
vocab_path = "./bert_model/vocab.txt"
mydict = ProData.LoadSingleSentenceClassificationDataset(vocab_path, tokenizer)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
config = Config()


# 加载 dataset
def load_dataset(data_index, flag):
    '''
    文件说明：
    './tmp_data/process_data/train_data.xlsx'
    './tmp_data/process_data/train2_data.xlsx'
    './tmp_data/process_data/test_data.xlsx'      这三个文件都是预处理数据的，即不能删改的

    './tmp_data/process_data/new_train_data.xlsx'
    './tmp_data/process_data/new_test_data.xlsx'  这两个文件都是微调时候，动态产生的文件，可以调整
    '''

    if flag == 0:
        # 返回预训练时期的训练的测试原始数据
        train1 = load_data("train1", data_index)
        train2 = load_data("train2", data_index)
        pool = load_data("pool", data_index)
        test = load_data("test", data_index)
        validate = load_data("validate", data_index)
        # 加载训练数据
        print("loading data----------------------------------预训练阶段")
        train_dataset = Rumor_Data(train1)
        train_dataset2 = Rumor_Data(train2)
        # train_loader2是长的，即全部，train_loader是只说打了标签的
        pool_dataset = Rumor_Data(pool)
        # 加载验证集数据
        validate_dataset = Rumor_Data(validate)
        # 加载测试数据
        test_dataset = Rumor_Data(test)

        return train_dataset, train_dataset2, validate_dataset, test_dataset, pool_dataset
    if flag == 1:
        # 加载微调时期的训练数据和测试数据
        print("loading data----------------------------------第一次微调")
        fine_train = load_data("fine_train", data_index)             # 这里的长度是对的
        fine_pool = load_data("fine_pool", data_index)               # 这里有问题，还是1093
        # 加载微调训练集
        fine_train_dataset = Rumor_Data(fine_train)
        # 加载微调测试集
        fine_pool_dataset = Rumor_Data(fine_pool)
        return fine_train_dataset, fine_pool_dataset
    if flag == 2:
        print("loading data----------------------------------后n次微调")
        fine_train_after = load_data("fine_train_after", data_index)  # 这里的长度是对的
        fine_pool_after = load_data("fine_pool_after", data_index)  # 这里有问题，还是1093
        # 加载微调训练集
        fine_train_dataset = Rumor_Data(fine_train_after)
        # 加载微调测试集
        fine_pool_dataset = Rumor_Data(fine_pool_after)
        return fine_train_dataset, fine_pool_dataset

# 定义预训练的训练方法
def train(models, config, train_dataset, train_dataset2, test_dataset, pool_dataset, flag):

    criterion = nn.BCELoss()
    criterion2 = nn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': models.bert.parameters()},  # 学习率为3e-5
        {'params': models.class_classifier.parameters(), 'lr': 0.001},   # 0.001
        {'params': models.domain_classifier.parameters(), 'lr': 0.005},
        {'params': models.infer_discriminator.parameters(), 'lr': 0.005}, # 0.005
        {'params': models.gru.parameters(), 'lr': 0.005},
        {'params': models.affection_discriminator.parameters(), 'lr': 0.005},
    ], lr=3e-5)
    file_name = './tmp_data/process_data/pre-train-network.pkl'  # 1.4873

    if exists(file_name):
        print("加载已经保存预训练的模型")
        models.load_state_dict(torch.load(file_name))
    # 开始训练过程
    else:
        print("训练模型！！!")
        for epoch in range(2):
            train_epoch(models, train_dataset, train_dataset2, optimizer, criterion, criterion2, epoch, flag)
        torch.save(models.state_dict(), file_name)

    best_dir = 'null'
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=16,
                             shuffle=True,
                             drop_last=True)
    pool_loader = DataLoader(dataset=pool_dataset,
                             batch_size=16,
                             shuffle=True,
                             drop_last=True)
    test(models, config, train_dataset, test_loader, best_dir, flag)
    append_data = select_best_data(models, config, pool_loader, train_dataset, pool_dataset, flag)

    return append_data

def train_epoch(models, train_dataset, train_dataset2, optimizer, criterion, criterion2, epoch, flag = 0):
    print("第epoch：", epoch)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=16,
                              shuffle=True,
                              drop_last = True)
    train_loader2 = DataLoader(dataset=train_dataset2,
                               batch_size=16,
                               shuffle=True,
                               drop_last = True)
    acc_vector = []
    cost_vector = []
    valid_acc_vector = []
    class_cost_vector = []
    domain_cost_vector = []
    mark_cost_vector = []
    epoch_loss, epoch_acc = 0., 0.
    total_len = 0
    models.train()
    iter2 = iter(train_loader)
    flag = np.array(flag)
    flag = torch.from_numpy(flag).long()
    flag = flag.to(device)
    # 分batch进行训练
    # 外面那层是多的，train_loader2
    # train_loader是只打了标签的部分数据
    for i, (train_text2, train_mask2, train_affection2, train_labels2, event_labels2, train_marked_label2, train_data_index2) in enumerate(train_loader2):
        (train_text1, train_mask1, train_affection1, train_labels1, event_labels1, train_marked_label1,train_data_index1) = iter2.__next__()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        x1 = train_text1
        x2 = train_mask1
        x3 = train_text2
        x4 = train_mask2
        x5 = train_affection2  # torch.Size([32, 8, 24])
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)
        x5 = x5.to(device)
        print(1)
        predict, domain_outputs, lable_outputs, lable_outputs2 = models(x1, x2, x3, x4, x5, flag)  # predict torch.Size([32, 2])

        train_labels = train_labels1.long()
        event_labels = event_labels1.long()
        train_marked_labels_soft = train_marked_label2.long()
        train_marked_labels = train_marked_label2.float()
        train_labels = train_labels.unsqueeze(1)
        train_marked_labels_soft = train_marked_labels_soft.unsqueeze(1)
        train_marked_labels = train_marked_labels.unsqueeze(1)

        train_labels = train_labels.to(device)
        event_labels = event_labels.to(device)
        train_marked_labels_soft = train_marked_labels_soft.to(device)
        train_marked_labels = train_marked_labels.to(device)

        train_labels = train_labels.squeeze(dim = 1)
        class_loss = criterion2(predict, train_labels)  # loss2(x,y.long())，cross损失函数里面要求，target的类型应该是long类型，input类型不做要求
        domain_loss = criterion2(domain_outputs, event_labels)
        train_marked_labels_soft = train_marked_labels_soft.squeeze(dim = 1)
        mark_loss = criterion(lable_outputs, train_marked_labels)
        mark_loss2 = criterion(lable_outputs2, train_marked_labels)
        # 用 预测loss来作为标准
        # class_loss2 = criterion3(predict, train_labels.squeeze(dim=1))
        # loss_loss = LossPredLoss(pred_loss, class_loss2)
        loss = class_loss + domain_loss + 2*(mark_loss + mark_loss2)
        # print("loss", loss)
        loss.backward()
        optimizer.step()

        _, argmax = torch.max(predict, 1)
        acc = (train_labels == argmax.squeeze()).float().mean()
        epoch_loss += loss * len(train_labels)
        epoch_acc += acc * len(train_labels)
        total_len += len(train_labels)
        # print(acc)
        acc_vector.append(acc.item())
        class_cost_vector.append(class_loss.item())
        domain_cost_vector.append(domain_loss.item())
        mark_cost_vector.append(mark_loss.item())
        cost_vector.append(loss.item())

        if (i + 1) % len(train_loader) == 0:

            iter2 = iter(train_loader)

    print('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, if_marked loss: %.4f, Train_Acc: %.4f'
            % (epoch + 1, 10, np.mean(cost_vector), np.mean(class_cost_vector),
                np.mean(domain_cost_vector), np.mean(mark_cost_vector),
                np.mean(acc_vector)))

# 微调时候的训练
def train_finetune(models, train_dataset, test_dataset, pool_dataset, flag):
    # 即 在微调训练的时候，就把目前的训练集和测试集再仍到第三个分支(0.1+0.05 : ...), 让它识别到下一次更不像本次新训练数据的测试数据
    append_data = []
    criterion = nn.BCELoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': models.bert.parameters()},  # 学习率为3e-5
        {'params': models.class_classifier.parameters(), 'lr': 0.001},
        {'params': models.domain_classifier.parameters(), 'lr': 0.005},
        {'params': models.infer_discriminator.parameters(), 'lr': 0.001},
        {'params': models.gru.parameters(), 'lr': 0.005},
        {'params': models.affection_discriminator.parameters(), 'lr': 0.005},
    ], lr=3e-5)
    freeze_layers = ['bert']
    for name, param in models.named_parameters():                # 打开Bert层
        for ele in freeze_layers:
            if ele in name:
                param.requires_grad = True
                break
    for i in range(5):              # 之前的模型是5(保存了的)
        # 微调第一个分支
        train_finetune_epoch(models, train_dataset, optimizer, criterion2, flag)
        best_dir = "null"
        # "微调"过程中的每一次测试
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=16,
                                 shuffle=True,
                                 drop_last=True)
        pool_loader = DataLoader(dataset=pool_dataset,
                                 batch_size=16,
                                 shuffle=True,
                                 drop_last=True)
        test(models, config, train_dataset, test_loader, best_dir, flag)
        append_data1 = select_best_data(models, config, pool_loader,  train_dataset, pool_dataset, flag)

    for i in range(2):
        for name, param in models.named_parameters():  # 冻结Bert层
            for ele in freeze_layers:
                if ele in name:
                    param.requires_grad = False
                    break
        optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001) # 0.00001
        ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.98)
        train_finetune3_epoch(models, train_dataset, test_dataset, optimizer1, ExpLR, criterion, flag)
        best_dir = "null"
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=16,
                                 shuffle=True,
                                 drop_last=True)
        pool_loader = DataLoader(dataset=pool_dataset,
                                 batch_size=16,
                                 shuffle=True,
                                 drop_last=True)
        test(models, config, train_dataset, test_loader, best_dir, flag)
        append_data1 = select_best_data(models, config, pool_loader,  train_dataset, pool_dataset, flag)

    return append_data1

# 微调第一个分支的函数
def train_finetune_epoch(models, train_dataset, optimizer, criterion ,flag = 1):
    models.train()
    flag = 1
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=16,
                              shuffle=True,
                              drop_last = True)

    flag = np.array(flag)
    flag = torch.from_numpy(flag).long()
    flag = flag.to(device)
    for i, (train_text, train_mask, train_affection, train_labels, event_labels, train_marked_label, train_data_index) in enumerate(train_loader):
        optimizer.zero_grad()
        train_text = train_text.to(device)
        train_mask = train_mask.to(device)
        train_affection = train_affection.to(device)
        predict = models(train_text, train_mask, train_text, train_mask, train_affection, flag)
        train_labels = train_labels.long()
        train_labels = train_labels.unsqueeze(1)
        train_labels = train_labels.to(device)
        train_labels = train_labels.squeeze(dim=1)
        class_loss = criterion(predict, train_labels)
        class_loss.backward()
        optimizer.step()

# 微调第三个分支的函数
def train_finetune3_epoch(models, train_dataset, test_dataset, optimizer,ExpLR, criterion, flag):
    models.train()
    cifar_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_loader = DataLoader(dataset=cifar_dataset,
                              batch_size=16,
                              shuffle=True,
                              drop_last= True)
    flag = 3
    flag = np.array(flag)
    flag = torch.from_numpy(flag).long()
    flag = flag.to(device)
    for i, (train_text, train_mask, train_affection, train_labels, event_labels, train_marked_label, train_data_index) in enumerate(train_loader):
        optimizer.zero_grad()
        train_text = train_text.to(device)
        train_mask = train_mask.to(device)
        train_affection = train_affection.to(device)          # 是否标记的标签
        label_out, label_out2 = models(train_text, train_mask, train_text, train_mask, train_affection, flag)
        train_marked_label = train_marked_label.float()
        train_marked_label = train_marked_label.unsqueeze(1)
        train_marked_label1 = train_marked_label
        train_marked_label = train_marked_label.to(device)
        train_marked_label1 = train_marked_label1.to(device)
        mark_loss = criterion(label_out, train_marked_label)
        mark_loss2 = criterion(label_out2, train_marked_label1)
        loss = mark_loss2 + mark_loss
        loss.backward()
        optimizer.step()
        ExpLR.step()
    pass

# 定义测试方法
def test(models, config, train_dataset, test_loader, best_dir, flag):
    # 仅仅作为测试功能的函数了
    global count
    epoch_acc = 0.
    epoch_test_accuracy = 0.
    total_len = 0
    all_preds_mark = []
    all_preds_mark2 = []
    all_preds = []                                # 只拿到了最大一列的值
    all_prefict = []
    all_index = []
    if exists(best_dir):
        models = MyNet(config).to(device)
        models.load_state_dict(torch.load(best_dir))
        print('加载已保存模型！')
    if torch.cuda.is_available():
        models.cuda()
    models.eval()
    test_score = []
    test_pred = []
    test_true = []
    flag = 0
    flag = np.array(flag)
    flag = torch.from_numpy(flag).long()
    flag = flag.to(device)
    with torch.no_grad():  ###插在此处
        # 分batch进行测试
        for i, (test_text, test_mask, test_affection, test_labels, event_labels, test_marked_label, test_data_index) in enumerate(test_loader):
            x1 = test_text
            x2 = test_mask
            x3 = test_affection
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            predict_label, predict_domain, predict_marked_label, predict_marked_label2 = models(x1, x2, x1, x2, x3, flag)
            # print("predict_marked_label的形状:", predict_marked_label.shape)     torch.Size([32, 1])
            preds_mark = predict_marked_label.cpu().data
            preds_mark2 = predict_marked_label2.cpu().data
            # 这里拿到的predict_label是经过了 softmax层的
            predict_label = predict_label.cpu().data
            # print("predict_label", predict_label)
            _, test_argmax = torch.max(predict_label, 1)                 # 相当于按照概率拿到了索引，即0，1
            # 得到每个batch预测的标签
            if i == 0:
                test_score = to_np(predict_label.squeeze())
                test_pred = to_np(test_argmax.squeeze())
                test_true = to_np(test_labels.squeeze())
            else:
                test_score = np.concatenate((test_score, to_np(predict_label.squeeze())), axis=0)
                test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
                test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)
            all_preds_mark.extend(preds_mark)            # 在列表末尾一次性追加另一个序列中的多个值
            all_preds_mark2.extend(preds_mark2)
            all_index.extend(test_data_index)

            all_preds.append(_)
            all_prefict.append(predict_label)

            test_labels = test_labels.float()
            test_labels = test_labels.unsqueeze(1)
            test_labels = test_labels.to(device)
            total_len += len(test_labels)
        # 最后把 predict_marked_label 拼起来，然后选择最低的几个（最靠近0的几个）选择出来
        all_preds_mark = torch.stack(all_preds_mark)    # all_preds_mark 把所有数据的0维拼接起来
        all_preds_mark = all_preds_mark.view(-1)
        all_preds_mark2 = torch.stack(all_preds_mark2)
        all_preds_mark2 = all_preds_mark2.view(-1)

        all_preds = torch.cat(all_preds)                # torch.cat, 把张量shape不相等的拼接起来
        all_preds = all_preds.view(-1)
        all_prefict = torch.cat(all_prefict)
        # print(all_preds.tolist())                                  最后都接近了0
        # need to multiply by -1 to be able to use torch.topk        负的好拿一些
        all_preds_mark *= -1
        all_preds_mark2 *= -1
        all_preds *= -1
        # 计算准确率
        test_accuracy = metrics.accuracy_score(test_true, test_pred)
        # F1值 可以解释为精度和查全率的加权平均值，其中F1分数在1时达到最佳值，在0时达到最差值。
        test_f1 = metrics.f1_score(test_true, test_pred)  # average='macro'
        # precison_score：预测为正类且预测正确的数量/预测为正类的数量
        test_precision = metrics.precision_score(test_true, test_pred)
        # 召回率 被预测为正的样本占正样本总量的比例。Recall体现了模型对正样本的识别能力，Recall越高，模型对正样本的识别能力越强。
        test_recall = metrics.recall_score(test_true, test_pred)
        # test_score_convert 就是把模型输出结果的对于第二列，每个对于1的预测概率来输出了
        test_score_convert = [x[1] for x in test_score]
        # ROC曲线，围成面积(记作AUC）越大，说明性能越好
        test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')
        test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)
        test_precision2, test_recall2, test_f12, _ = precision_recall_fscore_support(test_true, test_pred, average = "micro")
        print("Classification Acc: %.4f, AUC-ROC: %.4f"
              % (test_accuracy, test_aucroc))
        print("Classification report:\n%s\n"
              % (metrics.classification_report(test_true, test_pred)))
        print("Classification confusion matrix:\n%s\n"
              % (test_confusion_matrix))
        print("test_f1, test_precision, test_recall", test_f1, test_precision, test_recall)

def select_best_data(models, config, pool_loader,  train_dataset, pool_dataset, flag):
    flag = 0
    flag = np.array(flag)
    flag = torch.from_numpy(flag).long()
    flag = flag.to(device)
    # 相当于对已经训练好的模型，来选出那个是最不像的
    all_preds_mark = []
    all_preds_mark2 = []
    all_preds = []
    all_prefict = []
    all_index = []
    with torch.no_grad():  ###插在此处
        # 分batch进行测试
        for i, (pool_text, pool_mask, pool_affection, pool_labels, event_labels, pool_marked_label,pool_data_index) in enumerate(pool_loader):
            x1 = pool_text
            x2 = pool_mask
            x3 = pool_affection
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            predict_label, predict_domain, predict_marked_label, predict_marked_label2 = models(x1, x2, x1, x2, x3, flag)
            # print("predict_marked_label的形状:", predict_marked_label.shape)     torch.Size([32, 1])
            preds_mark = predict_marked_label.cpu().data
            preds_mark2 = predict_marked_label2.cpu().data

            predict_label = predict_label.cpu().data
            _, test_argmax = torch.max(predict_label, 1)  # 相当于按照概率拿到了索引，即0，1

            all_preds_mark.extend(preds_mark)  # 在列表末尾一次性追加另一个序列中的多个值
            all_preds_mark2.extend(preds_mark2)
            all_index.extend(pool_data_index)

            all_preds.append(_)
            all_prefict.append(predict_label)
        all_preds_mark = torch.stack(all_preds_mark)  # all_preds_mark 把所有数据的0维拼接起来
        all_preds_mark = all_preds_mark.view(-1)
        all_preds_mark2 = torch.stack(all_preds_mark2)
        all_preds_mark2 = all_preds_mark2.view(-1)

        all_preds = torch.cat(all_preds)  # torch.cat, 把张量shape不相等的拼接起来
        all_preds = all_preds.view(-1)
        all_prefict = torch.cat(all_prefict)

        # need to multiply by -1 to be able to use torch.topk        负的好拿一些
        all_preds_mark *= -1
        all_preds_mark2 *= -1
        all_preds *= -1
        # DAAL的办法
        append_data = select_data(all_preds_mark, all_preds_mark2, all_index, num)
        # 不确定熵方法
        # uncertaintysampler = UncertaintySampling(2, 0)
        # append_data = uncertaintysampler.query(all_preds, all_index, num)
        # print("这次pool_data数量是:", count)
        # append_data = random_index(count, num)
        # core-set方法
        # coresetsampler = CoreSetSampling(2, 0)
        # append_data = coresetsampler.greedy_k_center(train_dataset, pool_dataset, num)
        return append_data

def select_data(predict_marked_label, predict_marked_label2, data_index, num1):
    # 这里传入的 data_index 是数据文件 里面的index列
    pool_data1 = predict_marked_label
    pool_data2 = predict_marked_label2

    pool_data1 = normalize(pool_data1, p=1.0, dim=0)
    pool_data2 = normalize(pool_data2, p=1.0, dim=0)

    pool_data = 0.5*pool_data2 + pool_data1

    print("备选数据池内, pool_data: ", len(pool_data))
    _, querry_indices = torch.topk(pool_data, num1)  # 取一个tensor的topk元素
    querry_pool_indices = np.asarray(data_index)[querry_indices]  # 返回其索引
    return querry_pool_indices

def random_index(max,num):
    l = [i for i in range(max)]
    print("max:", max)
    random_pool_indices = random.sample(l, num)
    return random_pool_indices

def main(config, models):

    append_data_index = []
    append_data_tmp = []
    test_dataset = 0
    global count
    flag = 0  # 预训练模型阶段
    train_dataset, train_dataset2, validate_dataset, test_dataset, pool_dataset = load_dataset(append_data_index, flag)
    ## 通过训练拿到了预训练的模型
    append_data_index = train(models, config, train_dataset, train_dataset2, test_dataset, pool_dataset, flag)
    flag = 1  # 微调阶段
    # 在微调的时候，第一次用的是初始化的0.1的数据+0.05新选择的数据，第二次在新选择出来后，训练数据能不能只要第二次新选择的；175:1039, 54:1085
    for i in range(8):              # 0, ... , n-1  0.1 ... 8*0.05=0.4+0.1
        if i == 0:
            # 第一次微调用 0.1+0.05的数据
            print("第%d次微调" %(i+1))
            count = count - num
            train_dataset, pool_dataset = load_dataset(append_data_index, flag)
            append_data_tmp = train_finetune(models, train_dataset, test_dataset, pool_dataset, flag)
            print(append_data_tmp)
        else:
            # 第二次微调只要0.05的数据每次选出的
            print("第%d次微调" % (i+1))
            count = count - num

            flag = 2
            train_dataset, pool_dataset = load_dataset(append_data_tmp, flag)
            append_data_tmp = train_finetune(models, train_dataset, test_dataset, pool_dataset, flag)
            print(append_data_tmp)

if __name__ == '__main__':
    model = MyNet(config).to(device)
    config = Config()
    main(config, model)
    print("over")
