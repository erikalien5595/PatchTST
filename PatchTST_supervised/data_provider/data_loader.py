import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy import stats
# # 获取当前文件所在目录的父目录的父目录
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#
# print(f'path={parent_dir}')
# import sys
#
# # 将 'supervised' 目录添加到 sys.path
# sys.path.append(parent_dir)
from utils.timefeatures import time_features
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, n_clusters=3, cluster_random_state=42, is_cluster=0,
                 features='S', data_path='ETTh1.csv', use_catch22=False,
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.is_cluster = is_cluster
        if self.is_cluster:
            self.n_clusters = n_clusters
            self.cluster_random_state = cluster_random_state
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        if self.is_cluster:
            # 对数据进行聚类
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.cluster_random_state)  # 假设我们要分成3个簇
            train_data_std = data[border1s[0]:border2s[0]]
            kmeans.fit(data[border1s[0]:border2s[0]].T)
            # 获取每个样本的聚类标签
            labels = kmeans.labels_
            # 计算每个类别的数量
            label_counts = np.bincount(np.int64(labels))
            # 打印每个类别的数量
            if self.flag=='train':
                print(f'Clustering Result:')
                for label, count in enumerate(label_counts):
                    print(f'    Category {label}: {count} sequences')
            # 建立一个字典，用于保存聚类以后每一类的变量index
            self.label_dict = {}
            for label in np.unique(labels):
                if label not in self.label_dict:
                    self.label_dict[label] = list(np.where(labels == label)[0])
            # if self.data_path=='ETTh2.csv':
            #     labels = [0, 1]
            #     self.label_dict = {0: [2, 4, 5, 6], 1: [0, 1, 3]}
            # if self.data_path=='ETTh1.csv':
            #     labels = [0, 1, 2]
            #     self.label_dict = {0: [0, 2], 1: [6, 5, 4], 2: [1, 3]}
            self.sra_dict = {}
            for i in np.unique(labels):
                if len(self.label_dict[i]) > 2:
                    # 生成一个布尔掩码矩阵，掩盖对角元素
                    corr = stats.spearmanr(train_data_std[:, self.label_dict[i]])[0]
                    mask = ~np.eye(corr.shape[0], dtype=bool)
                    # 提取非对角元素
                    non_diagonal_elements = corr[mask]
                    self.sra_dict[i] = np.mean(np.abs(non_diagonal_elements))
                    # print(f"第{i}类：", np.mean(np.abs(non_diagonal_elements)), np.min(np.abs(non_diagonal_elements)),
                    #       np.max(np.abs(non_diagonal_elements)), train_data.columns[self.label_dict[i]])
                elif len(self.label_dict[i]) == 2:
                    corr = stats.spearmanr(train_data_std[:, self.label_dict[i]])[0]
                    self.sra_dict[i] = corr
                    # print(f"第{i}类：", corr, train_data.columns[self.label_dict[i]])
                else:
                    self.sra_dict[i] = 0
                    # print(f"第{i}类只有1个序列: {train_data.columns[self.label_dict[i]]}")
                if self.flag == 'train':
                    print(f'第{i}类相关系数绝对值的均值：{np.round(self.sra_dict[i], 4)}')

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, n_clusters=3, cluster_random_state=42, is_cluster=0,
                 features='S', data_path='ETTm1.csv', use_catch22=False,
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.is_cluster = is_cluster
        if self.is_cluster:
            self.n_clusters = n_clusters
            self.cluster_random_state = cluster_random_state
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if self.is_cluster:
            train_data_std = data[border1s[0]:border2s[0]]
            # 对数据进行聚类
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.cluster_random_state)  # 假设我们要分成3个簇
            kmeans.fit(train_data_std.T)
            # 获取每个样本的聚类标签
            labels = kmeans.labels_
            # 计算每个类别的数量
            label_counts = np.bincount(np.int64(labels))
            # 打印每个类别的数量
            if self.flag=='train':
                print(f'Clustering Result:')
                for label, count in enumerate(label_counts):
                    print(f'    Category {label}: {count} sequences')
            # 建立一个字典，用于保存聚类以后每一类的变量index
            self.label_dict = {}
            for label in np.unique(labels):
                if label not in self.label_dict:
                    self.label_dict[label] = list(np.where(labels == label)[0])
            self.sra_dict = {}
            for i in np.unique(labels):
                if len(self.label_dict[i]) > 2:
                    # 生成一个布尔掩码矩阵，掩盖对角元素
                    corr = stats.spearmanr(train_data_std[:, self.label_dict[i]])[0]
                    mask = ~np.eye(corr.shape[0], dtype=bool)
                    # 提取非对角元素
                    non_diagonal_elements = corr[mask]
                    self.sra_dict[i] = np.mean(np.abs(non_diagonal_elements))
                    # print(f"第{i}类：", np.mean(np.abs(non_diagonal_elements)), np.min(np.abs(non_diagonal_elements)),
                    #       np.max(np.abs(non_diagonal_elements)), train_data.columns[self.label_dict[i]])
                elif len(self.label_dict[i]) == 2:
                    corr = stats.spearmanr(train_data_std[:, self.label_dict[i]])[0]
                    self.sra_dict[i] = corr
                    # print(f"第{i}类：", corr, train_data.columns[self.label_dict[i]])
                else:
                    self.sra_dict[i] = 0
                    # print(f"第{i}类只有1个序列: {train_data.columns[self.label_dict[i]]}")
                if self.flag == 'train':
                    print(f'第{i}类相关系数绝对值的均值：{np.round(self.sra_dict[i], 4)}')

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, n_clusters=3, cluster_random_state=42, is_cluster=0,
                 features='S', data_path='ETTh1.csv', use_catch22=False,
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.is_cluster = is_cluster
        if self.is_cluster:
            self.n_clusters = n_clusters
            self.cluster_random_state = cluster_random_state
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
        self.use_catch22 = use_catch22

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        # ###### 实验通道独立 ######
        # if self.data_path=='weather.csv':
        #     # df_data = df_data.iloc[:, [1, 2, 5, 7, 19]]  # cluster 0
        #     df_data = df_data.iloc[:, [0, 4, 10, 20]]  # cluster 1
        #     if self.flag=='train':
        #         print(df_data.columns)
        # ########################

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        if self.is_cluster:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)  # 假设我们要分成3个簇
            if self.data_path == 'weather.csv':
                print('weather数据处理异常值：修正异常值后重新聚类，但不改变原始数据的标准化')
                train_data.loc[train_data['OT'] == -9999, 'OT'] = 417 #替换成均值
                train_data.loc[train_data['OT'] <350, 'OT'] = 417  # 替换成均值
                train_data.loc[train_data['max. PAR (�mol/m�/s)'] == -9999, 'max. PAR (�mol/m�/s)'] = 0
                train_data.loc[train_data['wv (m/s)'] == -9999, 'wv (m/s)'] = 0
                if self.use_catch22==0:
                    scaler2 = StandardScaler()
                    data2 = scaler2.fit_transform(train_data.values)
            else:
                data2 = data[border1s[0]:border2s[0]]
            ###### catch22 to extract features for clustering ######
            if self.use_catch22==1:
                import pycatch22
                tmp = pycatch22.catch22_all(train_data.iloc[:, 0].values)
                data2 = pd.DataFrame(np.array(tmp['values']).reshape(1, -1), columns=tmp['names'])
                for i in range(1, train_data.shape[1]):
                    tmp = pycatch22.catch22_all(train_data.iloc[:, i].values)
                    new_line = pd.DataFrame(np.array(tmp['values']).reshape(1, -1), columns=tmp['names'])
                    data2 = pd.concat([data2, new_line])
                data2 = data2.values.T
                scaler3 = StandardScaler()
                data2 = scaler3.fit_transform(data2)  # 把提取的22个特征进行标准化
            ########################################################

            # 对数据进行聚类
            kmeans.fit(data2.T)
            # 获取聚类中心
            cluster_centroids = kmeans.cluster_centers_
            # 获取每个样本的聚类标签
            labels = kmeans.labels_
            # 计算每个类别的数量
            label_counts = np.bincount(np.int64(labels))
            # 打印每个类别的数量
            if self.flag=='train':
                print(f'Clustering Result:')
                for label, count in enumerate(label_counts):
                    print(f'    Category {label}: {count} sequences')
            # 建立一个字典，用于保存聚类以后每一类的变量index
            self.label_dict = {}
            for label in np.unique(labels):
                if label not in self.label_dict:
                    self.label_dict[label] = list(np.where(labels == label)[0])
                # print(df_raw.iloc[:, np.where(labels == label)[0]])
            self.sra_dict = {}
            for i in np.unique(labels):
                if len(self.label_dict[i]) > 2:
                    # 生成一个布尔掩码矩阵，掩盖对角元素
                    corr = stats.spearmanr(data2[:, self.label_dict[i]])[0]
                    mask = ~np.eye(corr.shape[0], dtype=bool)
                    # 提取非对角元素
                    non_diagonal_elements = corr[mask]
                    self.sra_dict[i] = np.mean(np.abs(non_diagonal_elements))
                    # print(f"第{i}类：", np.mean(np.abs(non_diagonal_elements)), np.min(np.abs(non_diagonal_elements)),
                    #       np.max(np.abs(non_diagonal_elements)), train_data.columns[self.label_dict[i]])
                elif len(self.label_dict[i]) == 2:
                    corr = stats.spearmanr(data2[:, self.label_dict[i]])[0]
                    self.sra_dict[i] = corr
                    # print(f"第{i}类：", corr, train_data.columns[self.label_dict[i]])
                else:
                    self.sra_dict[i] = 0
                    # print(f"第{i}类只有1个序列: {train_data.columns[self.label_dict[i]]}")
                if self.flag == 'train':
                    print(f'第{i}类相关系数绝对值的均值：{np.round(self.sra_dict[i], 4)}')

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None, n_clusters=3, cluster_random_state=42, is_cluster=0,
                 features='S', data_path='ETTh1.csv', use_catch22=False,
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.is_cluster = is_cluster
        if self.is_cluster:
            self.n_clusters = n_clusters
            self.cluster_random_state = cluster_random_state
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
            train_data_std = self.scaler.transform(train_data)

        if self.is_cluster:
            # 对数据进行聚类
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.cluster_random_state)  # 假设我们要分成3个簇
            kmeans.fit(train_data_std.T)
            # 获取每个样本的聚类标签
            labels = kmeans.labels_
            # 计算每个类别的数量
            label_counts = np.bincount(np.int64(labels))
            # 打印每个类别的数量
            if self.flag=='train':
                print(f'Clustering Result:')
                for label, count in enumerate(label_counts):
                    print(f'    Category {label}: {count} sequences')
            # 建立一个字典，用于保存聚类以后每一类的变量index
            self.label_dict = {}
            for label in np.unique(labels):
                if label not in self.label_dict:
                    self.label_dict[label] = list(np.where(labels == label)[0])
                # print(data[:, np.where(labels == label)[0]])
            self.sra_dict = {}
            for i in np.unique(labels):
                if len(self.label_dict[i]) > 2:
                    # 生成一个布尔掩码矩阵，掩盖对角元素
                    corr = stats.spearmanr(train_data_std[:, self.label_dict[i]])[0]
                    mask = ~np.eye(corr.shape[0], dtype=bool)
                    # 提取非对角元素
                    non_diagonal_elements = corr[mask]
                    self.sra_dict[i] = np.mean(np.abs(non_diagonal_elements))
                    # print(f"第{i}类：", np.mean(np.abs(non_diagonal_elements)), np.min(np.abs(non_diagonal_elements)),
                    #       np.max(np.abs(non_diagonal_elements)), train_data.columns[self.label_dict[i]])
                elif len(self.label_dict[i]) == 2:
                    corr = stats.spearmanr(train_data_std[:, self.label_dict[i]])[0]
                    self.sra_dict[i] = corr
                    # print(f"第{i}类：", corr, train_data.columns[self.label_dict[i]])
                else:
                    self.sra_dict[i] = 0
                    # print(f"第{i}类只有1个序列: {train_data.columns[self.label_dict[i]]}")
                if self.flag == 'train':
                    print(f'第{i}类相关系数绝对值的均值：{np.round(self.sra_dict[i], 4)}')

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None, n_clusters=3, cluster_random_state=42, is_cluster=0,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.is_cluster = is_cluster
        if self.is_cluster:
            self.n_clusters = n_clusters
            self.cluster_random_state = cluster_random_state
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)
        print(df_raw.shape)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        if self.is_cluster:
            # 对数据进行聚类
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.cluster_random_state)  # 假设我们要分成3个簇
            kmeans.fit(data.T)
            # 获取每个样本的聚类标签
            labels = kmeans.labels_
            # 计算每个类别的数量
            label_counts = np.bincount(np.int64(labels))
            # 打印每个类别的数量
            if self.flag=='train':
                print(f'Clustering Result:')
                for label, count in enumerate(label_counts):
                    print(f'    Category {label}: {count} sequences')
            # 建立一个字典，用于保存聚类以后每一类的变量index
            self.label_dict = {}
            for label in np.unique(labels):
                if label not in self.label_dict:
                    self.label_dict[label] = list(np.where(labels == label)[0])
                print(data[:, np.where(labels == label)[0]])

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

if __name__=='__main__':
    # data_set = Dataset_Custom(root_path='../', data_path='dataset/traffic.csv', flag='train',
    #                             features='M', is_cluster=True, size=[96, 48, 24], n_clusters=5, timeenc=0)
    # data_set = Dataset_PEMS(root_path='../dataset/PEMS/', data_path='PEMS03.npz', flag='train',
    #                             features='M', is_cluster=True, size=[96, 48, 24], n_clusters=5, timeenc=0)
    data_set = Dataset_Solar(root_path='../dataset/', data_path='solar_AL.txt', flag='train',
                                features='M', is_cluster=True, size=[96, 48, 24], n_clusters=5, timeenc=0)
