# -*- codeing = utf-8 -*-
# @Time : 2022-08-17 11:07
# @Author : 张超然
# @File ： CoreSet_Samper.py
# @Software: PyCharm
import gc
import torch
import numpy as np
from scipy.spatial import distance_matrix
from torch.nn.functional import normalize
class QueryMethod:
    """
    A general class for query strategies, with a general method for querying examples to be labeled.
    """

    def __init__(self, num_labels=2, gpu=1):

        self.num_labels = num_labels
        self.gpu = gpu

    def query(self, test_predict, test_index, amount):
        """
        get the indices of labeled examples after the given amount have been queried by the query strategy.
        在给定查询策略后，得到有标签数据的索引
        :param X_train: the training set
        :param Y_train: the training labels
        :param labeled_idx: the indices of the labeled examples
        :param amount: the amount of examples to query
        :return: the new labeled indices (including the ones queried)
        """
        return NotImplemented

    # def update_model(self, new_model):
    #     del self.model
    #     gc.collect()
    #     self.model = new_model

class UncertaintySampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the minimal top confidence.
    基础的查询案例：查询到最小置信度的例子
    """

    def __init__(self, num_labels, gpu):
        super().__init__(num_labels, gpu)

    def query(self, test_predict, test_index, amount):

        # 这里改成他们距离0.5的距离
        # test_predict = test_predict.tolist()
        # test_predict = [x - 0.5 for x in test_predict]

        pool_data = test_predict
        _, querry_indices = torch.topk(pool_data, amount)  # 取一个tensor的topk元素
        querry_pool_indices = np.asarray(test_index)[querry_indices]  # 返回其索引
        return querry_pool_indices

class UncertaintyEntropySampling(QueryMethod):
    """
       The basic uncertainty sampling query strategy, querying the examples with the top entropy.
       基础的查询案例：用信息熵来查询
    """
    def __init__(self, num_labels, gpu):
        super().__init__(num_labels, gpu)

    def query(self, test_predict, test_index, amount):
        pool_data = test_predict
        # pool_data = pool_data.numpy()
        unlabeled_predictions = torch.sum(torch.from_numpy(pool_data * np.log(pool_data + 1e-10)), dim=1)
        unlabeled_predictions = normalize(unlabeled_predictions, p=1.0, dim=0)
        # print('unlabeled_predictions', unlabeled_predictions)
        # print('unlabeled_predictions', len(unlabeled_predictions))
        unlabeled_predictions *= -1
        _, querry_indices = torch.topk(unlabeled_predictions, amount)  # 取一个tensor的topk元素
        querry_pool_indices = np.asarray(test_index)[querry_indices]  # 返回其索引
        return querry_pool_indices


# class UncertaintyEntropySampling(QueryMethod):
#     """
#        The basic uncertainty sampling query strategy, querying the examples with the top entropy.
#        基础的查询案例：用信息熵来查询
#     """
#     def __init__(self, num_labels, gpu):
#         super().__init__(num_labels, gpu)
#
#     def query(self, test_predict, test_index, amount):
#         pool_data = test_predict
#         unlabeled_predictions = torch.sum(pool_data * np.log(pool_data + 1e-10), dim=1)
#         print('unlabeled_predictions', unlabeled_predictions)
#         print('unlabeled_predictions', len(unlabeled_predictions))
#         unlabeled_predictions *= -1
#         _, querry_indices = torch.topk(unlabeled_predictions, amount)  # 取一个tensor的topk元素
#         querry_pool_indices = np.asarray(test_index)[querry_indices]  # 返回其索引
#         return querry_pool_indices

class CoreSetSampling(QueryMethod):
    """
    An implementation of the greedy core set query strategy.
    """

    def __init__(self, num_labels, gpu):
        super().__init__(num_labels, gpu)

    def greedy_k_center(self, train_dataset, test_dataset, amount):
        labeled = []
        unlabeled = []
        for i, (train_text, train_mask, train_affection, train_labels, train_labels, train_marked_label, train_data_index) in enumerate(train_dataset):
            labeled.append(train_text.numpy())
        for i, (test_text, test_mask, test_affection, test_labels, test_labels, test_marked_label, test_data_index) in enumerate(test_dataset):
            unlabeled.append(test_text.numpy())
        greedy_indices = []

        labeled = np.array(labeled)
        unlabeled = np.array(unlabeled)
        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        # labeled[0].reshape((1, labeled.shape[1])) 这句话是把其变成2维numpy，便于其计算距离
        # len(np.min(distance_matrix(label.ed[0].reshape((1, labeled.shape[1])), unlabeled), axis=0)) = 1093
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        # [[26800.85291553 29789.73967661 33616.04480007 ... 36828.27025534
        #   52557.46453169 29255.35424499]]            1093
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist1 = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist1, axis=0)                            # 上一个结果叠加后，axis=0是返回每一列的最小值
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)           # 取出中元素最大值所对应的索引
        greedy_indices.append(farthest)          # 相当于 第farthest个数据是距离上面构建的j个距离中心的最远的一个元素
        for i in range(amount-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)
        print(np.array(greedy_indices))
        return greedy_indices

    # def query(self, X_train, Y_train, labeled_idx, amount):
    #
    #     unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
    #
    #     # use the learned representation for the k-greedy-center algorithm:
    #     representation_model = Model(inputs=self.model.input, outputs=self.model.get_layer('softmax').input)
    #     representation = representation_model.predict(X_train, verbose=0)
    #     new_indices = self.greedy_k_center(representation[labeled_idx, :], representation[unlabeled_idx, :], amount)
    #     return np.hstack((labeled_idx, unlabeled_idx[new_indices]))
