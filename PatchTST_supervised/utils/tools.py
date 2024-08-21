import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import pycatch22
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, is_cluster=0):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            # if is_cluster:
            #     for model_index in range(len(model)):  # 此时的model是一个list，里面有K个model，K为cluster个数
            #         self.save_checkpoint(val_loss, model, path, is_cluster, model_index)
            # else:
            #     self.save_checkpoint(val_loss, model, path, is_cluster)
            self.save_checkpoint(val_loss, model, path, is_cluster)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # if is_cluster:
            #     for model_index in range(len(model)):
            #         self.save_checkpoint(val_loss, model, path, is_cluster, model_index)
            # else:
            #     self.save_checkpoint(val_loss, model, path, is_cluster)
            self.save_checkpoint(val_loss, model, path, is_cluster)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, is_cluster=0, model_index=0):
        if is_cluster:
            # if self.verbose and model_index==0:
            #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  '
            #           f'Saving model for cluster model...')
            # torch.save(model[model_index].state_dict(), path + '/' + f'checkpoint_cluster_{model_index}.pth')
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  '
                      f'Saving model for cluster model...')
                torch.save(model.state_dict(), path + '/' + f'checkpoint_cluster.pth')
        else:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  '
                      f'Saving model ...')
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


def get_catch22_data(data):
    """
    :param data: 原始的多元时间序列，shape1是通道
    :return: 提取的22个特征组成的新的多元序列，shape1是通道
    """
    tmp = pycatch22.catch22_all(data.iloc[:, 0].values)
    data2 = pd.DataFrame(np.array(tmp['values']).reshape(1, -1), columns=tmp['names'])
    for i in range(1, data.shape[1]):
        tmp = pycatch22.catch22_all(data.iloc[:, i].values)
        new_line = pd.DataFrame(np.array(tmp['values']).reshape(1, -1), columns=tmp['names'])
        data2 = pd.concat([data2, new_line])
    data2 = data2.values.T
    return data2


def find_best_k(data, flag, random_state=None):
    k_score, k = [], []
    max_k = 10 if data.shape[1] > 10 else data.shape[1]
    for i in np.arange(2, max_k):
        # for i in np.arange(2, np.min(8, data2.shape[1])):
        # 对数据进行聚类
        kmeans2 = KMeans(n_clusters=i, n_init=20, random_state=random_state)
        # kmeans2 = AgglomerativeClustering(n_clusters=i, random_state=cluster_random_state)
        kmeans2.fit(data.T)  # PEMS
        # kmeans.fit(df_raw_std.T)  # ECL, traffic, weather
        # 获取聚类中心
        cluster_centroids = kmeans2.cluster_centers_
        # 获取每个样本的聚类标签
        labels = kmeans2.labels_
        score = silhouette_score(data.T, labels)  # silhouette_score, calinski_harabasz_score, davies_bouldin_score
        k_score.append(score)
        k.append(i)
        if flag == 'train':
            print(f'n={i}, silhouette_score={np.round(score, 4)}')
    dict_shape = dict(zip(k, k_score))
    best_k = sorted(dict_shape.items(), key=lambda x: x[1], reverse=True)[0][0]
    if flag == 'train':
        print(f'best K={best_k}!!!')

    return best_k

def recluster(data, labels, cluster_centroids, label_dict, sra_dict, flag='train', threshold=0.8):
    """

    :param data: 重新聚类的原始数据
    :param sra_dict: key：类index，value：类内相关系数均值
    :param label_dict: key：类index，value：类内的序列在data中的index
    :return: 把类内相关系数低于阈值的重新聚到1个类内，否则不变
    """
    # 定义一个新的大类列表来存储需要重新归类的类别
    new_class = []
    new_label_dict = {}
    new_sra_dict = {}
    # 遍历 sra_dict 并处理类内均值小于 0.6 的类别
    for i in sra_dict:
        if sra_dict[i] < threshold:
            # 将当前类别添加到新大类中
            new_class.extend(label_dict[i])
        else:
            # 保留类内均值大于等于 0.6 的类别
            new_label_dict[i] = label_dict[i]
            new_sra_dict[i] = sra_dict[i]
    # 如果 new_class 非空，则将其添加为新的大类
    if new_class:
        new_label_key = max(label_dict.keys()) + 1  # 新的大类键可以是当前字典键的最大值加1
        new_label_dict[new_label_key] = new_class
        # 计算新类内的相关系数
        if len(new_label_dict[new_label_key])>1:
            corr = np.corrcoef(data[:, new_label_dict[new_label_key]].T)
            mask = ~np.eye(corr.shape[0], dtype=bool)
            non_diagonal_elements = corr[mask]
            new_sra_dict[new_label_key] = np.mean(non_diagonal_elements)
        else:
            new_sra_dict[new_label_key] = 0
        # 重新设置新的类别字典的 key 为 0, 1, 2, ...
        sorted_new_label_dict = {}
        sorted_new_sra_dict = {}
        for new_key, old_key in enumerate(new_label_dict):
            sorted_new_label_dict[new_key] = new_label_dict[old_key]
            sorted_new_sra_dict[new_key] = new_sra_dict[old_key]
        if flag=='train':
            # 打印新的类别分配情况
            print(f">>>>>>>>>>>>After ReClustering: {len(sorted_new_label_dict.keys())} clusters in total.")
            for key, value in sorted_new_label_dict.items():
                print(f"  Category {key}: {len(value)} sequences, mean(corr)={np.round(sorted_new_sra_dict[key], 4)}, "
                      f"channels in this cluster: {value}")
        # 计算数组的最大索引值，以确定数组的大小
        max_index = max(max(v) for v in label_dict.values())
        # 创建一个数组，并初始化为 -1（或其他值，以便区分未填充的位置）
        new_labels = np.full(max_index + 1, -1)
        # 填充数组
        for key, indices in sorted_new_label_dict.items():
            for index in indices:
                new_labels[index] = key
        new_cluster_centroids = np.zeros((len(sorted_new_label_dict.keys()), data.shape[0]))
        for i in sorted_new_label_dict.keys():
            new_cluster_centroids[i, :] = data[:, new_labels == i].mean(axis=1)
        return new_labels, new_cluster_centroids, sorted_new_label_dict, sorted_new_sra_dict
    else:
        return labels, cluster_centroids, label_dict, sra_dict


def get_label_sra_dict(labels, data, flag):
    # 建立一个字典，用于保存聚类以后每一类的变量index
    label_dict = {}
    for label in np.unique(labels):
        if label not in label_dict:
            label_dict[label] = list(np.where(labels == label)[0])
        # print(df_raw.iloc[:, np.where(labels == label)[0]])
    sra_dict = {}
    for i in np.unique(labels):
        if len(label_dict[i]) >= 2:
            # 生成一个布尔掩码矩阵，掩盖对角元素
            corr = np.corrcoef(data[:, label_dict[i]].T)
            mask = ~np.eye(corr.shape[0], dtype=bool)
            # 提取非对角元素
            non_diagonal_elements = corr[mask]
            sra_dict[i] = np.mean(np.abs(non_diagonal_elements))
        else:
            sra_dict[i] = 0

    return label_dict, sra_dict


def correlation_group(data, threshold):
    # 定义阈值
    correlation_matrix = np.corrcoef(data.T)
    # 初始化类标识
    clusters = []
    visited = set()
    single_clusters = []

    # 遍历每个向量并执行DFS
    for i in range(len(correlation_matrix)):
        if i not in visited:
            new_cluster = [i]
            visited.add(i)
            for j in range(len(correlation_matrix)):
                if j not in visited:
                    if all(np.abs(correlation_matrix[j, x]) >= threshold for x in new_cluster):
                    # if all(correlation_matrix[j, x] >= threshold for x in new_cluster):
                        new_cluster.append(j)
                        visited.add(j)

            if len(new_cluster) == 1:
                single_clusters.append(new_cluster[0])  # 将单独一类的放入单独类列表
            else:
                clusters.append(new_cluster)

    # 将所有满足条件的大于 threshold 的类合并为一个大类
    if clusters:
        merged_cluster = [item for sublist in clusters for item in sublist]
        clusters = [merged_cluster]

    # 将所有单独一类的向量合并到一个大类中
    if single_clusters:
        clusters.append(single_clusters)

    # 输出每个聚类及其最低阈值
    new_index = []
    label_dict,sra_dict = {}, {}
    for idx, cluster in enumerate(clusters):
        min_threshold = 1.0
        new_index.extend(cluster)
        for x in cluster:
            for y in cluster:
                if x != y:
                    min_threshold = min(min_threshold, np.abs(correlation_matrix[x, y]))
                    # min_threshold = min(min_threshold, correlation_matrix[x, y])
                    label_dict[idx] = cluster
                    sra_dict[idx] = min_threshold
        print(f"Cluster {idx} has {len(cluster)} channels: {cluster}, Minimum Threshold: {min_threshold:.4f}")
        sra_dict[0] = 1
        sra_dict[1] = 0
    # print(label_dict, sra_dict)
    return new_index, label_dict, sra_dict


def correlation_group2(data, threshold):
    # 定义阈值
    correlation_matrix = np.corrcoef(data.T)
    # 初始化类标识
    clusters = []
    visited = set()
    single_clusters = []

    # 遍历每个向量并执行DFS
    for i in range(len(correlation_matrix)):
        if i not in visited:
            new_cluster = [i]
            visited.add(i)
            for j in range(len(correlation_matrix)):
                if j not in visited:
                    if all(np.abs(correlation_matrix[j, x]) > threshold for x in new_cluster):
                    # if all(correlation_matrix[j, x] >= threshold for x in new_cluster):
                        new_cluster.append(j)
                        visited.add(j)

            if len(new_cluster) == 1:
                single_clusters.append(new_cluster[0])  # 将单独一类的放入单独类列表
            else:
                clusters.append(new_cluster)

    # # 将所有满足条件的大于 0.8 的类合并为一个大类
    # if clusters:
    #     merged_cluster = [item for sublist in clusters for item in sublist]
    #     clusters = [merged_cluster]

    # 将所有单独一类的向量合并到一个大类中
    if single_clusters:
        clusters.append(single_clusters)

    # 输出每个聚类及其最低阈值
    new_index = []
    label_dict,sra_dict = {}, {}
    for idx, cluster in enumerate(clusters):
        min_threshold = 1.0
        new_index.extend(cluster)
        for x in cluster:
            for y in cluster:
                if x != y:
                    min_threshold = min(1, np.abs(correlation_matrix[x, y]))
                    # min_threshold = min(min_threshold, correlation_matrix[x, y])
                    label_dict[idx] = cluster
                    sra_dict[idx] = min_threshold
        print(f"Cluster {idx} has {len(cluster)} channels: {cluster}, Maximum Threshold: {min_threshold:.4f}")
        # sra_dict[0] = 1
        # sra_dict[1] = 0
    print(label_dict, sra_dict)
    return new_index, label_dict, sra_dict


def correlation_group_2class(data, threshold):
    # 定义阈值
    correlation_matrix = np.corrcoef(data.T)
    # 初始化类标识
    clusters = []
    visited = set()
    single_clusters = []

    # 遍历每个向量并执行DFS
    for i in range(len(correlation_matrix)):
        if i not in visited:
            new_cluster = [i]
            visited.add(i)
            for j in range(len(correlation_matrix)):
                if j not in visited:
                    if all(np.abs(correlation_matrix[j, x]) >= threshold for x in new_cluster):
                    # if all(correlation_matrix[j, x] >= threshold for x in new_cluster):
                        new_cluster.append(j)
                        visited.add(j)

            if len(new_cluster) == 1:
                single_clusters.append(new_cluster[0])  # 将单独一类的放入单独类列表
            else:
                clusters.append(new_cluster)

    # 将所有满足条件的大于 0.8 的类合并为一个大类
    if clusters:
        merged_cluster = [item for sublist in clusters for item in sublist]
        clusters = [merged_cluster]

    # 将所有单独一类的向量合并到一个大类中
    if single_clusters:
        clusters.append(single_clusters)

    # 输出每个聚类及其最低阈值
    new_index = []
    label_dict,sra_dict = {}, {}
    for idx, cluster in enumerate(clusters):
        min_threshold = 1.0
        new_index.extend(cluster)
        for x in cluster:
            for y in cluster:
                if x != y:
                    min_threshold = min(min_threshold, np.abs(correlation_matrix[x, y]))
                    # min_threshold = min(min_threshold, correlation_matrix[x, y])
                    label_dict[idx] = cluster
                    sra_dict[idx] = min_threshold
        print(f"Cluster {idx} has {len(cluster)} channels: {cluster}, Minimum Threshold: {min_threshold:.4f}")
        sra_dict[0] = 1
        sra_dict[1] = 0
    # print(label_dict, sra_dict)
    return new_index, label_dict, sra_dict


def kmeans_by_similarity():

    # 假设你有一个 n x m 的数据矩阵，其中 n 是样本数，m 是特征数
    data = np.random.rand(20, 5)  # 示例数据

    # 计算皮尔逊相关系数矩阵
    correlation_matrix = np.corrcoef(data)

    # 将相关系数矩阵转换为距离矩阵
    # 使用 1 - 相关系数 作为“距离”
    distance_matrix = 1 - correlation_matrix

    # K-means 聚类的特征：基于距离矩阵
    # 使用 squareform 将距离矩阵转换为 K-means 需要的特征矩阵
    kmeans = KMeans(n_clusters=3, random_state=0).fit(squareform(distance_matrix))

    # 输出聚类标签
    print("Cluster labels:", kmeans.labels_)


