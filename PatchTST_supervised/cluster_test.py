import numpy as np
import pandas as pd
from  data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
import torch
if __name__=='__main__':
    # dat = Dataset_ETT_hour('./dataset/ETT-small/', features="M")
    # dat = Dataset_Custom('./dataset/illness/', features="M", data_path='national_illness.csv')
    # print(dat.data_x.shape, dat.data_y.shape)
    #
    # # 创建一个热力图
    # fig, ax = plt.subplots()
    # heatmap = ax.imshow(np.corrcoef(dat.data_x.T), cmap='coolwarm')
    # # 为热力图增加颜色条标记
    # fig.colorbar(heatmap)
    # # 显示图形
    # plt.show()

    # df_raw = pd.read_csv('./dataset/ETT-small/ETTm2.csv')
    # df_raw = pd.read_csv('./dataset/illness/national_illness.csv')
    df_raw = pd.read_csv('./dataset/weather/weather.csv')
    # df_raw = pd.read_csv('./dataset/electricity/electricity.csv')
    # df_raw = pd.read_csv('./dataset/traffic/traffic.csv').iloc[:1000]

    from sklearn.preprocessing import StandardScaler
    import os
    scaler = StandardScaler()
    data_file = './dataset/PEMS/PEMS03.npz'
    data = np.load(data_file, allow_pickle=True)
    data = data['data'][:, :, 0]
    print(data.shape)

    train_ratio = 0.6
    valid_ratio = 0.2
    train_data = data[:int(train_ratio * len(data))]
    valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
    test_data = data[int((train_ratio + valid_ratio) * len(data)):]
    total_data = [train_data, valid_data, test_data]

    scaler.fit(train_data)
    train_data_std = scaler.transform(train_data)

    cols = list(df_raw.columns)
    cols.remove('OT')
    cols.remove('date')
    df_raw = df_raw[cols + ['OT']]

    # print(self.scaler.mean_)
    # exit()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df_raw.values)
    # df_raw_std = scaler.transform(df_raw.values)
    df_raw_std = df_raw.values # 不做标准化
    # context_window = df_raw.shape[0]
    # patch_len = 96
    # stride = 96
    # patch_num = int((context_window - patch_len) / stride + 1)
    # padding_patch_layer = nn.ReplicationPad1d((0, stride))
    # patch_num += 1
    # z = padding_patch_layer(torch.Tensor(df_raw_std.T))
    # z = z.unfold(dimension=-1, size=patch_len, step=stride)  # z: [bs x nvars x patch_num x patch_len]
    # print(z.shape)

    print(df_raw_std.shape)
    # # 创建一个热力图
    # plt.figure(figsize=[16, 9])
    # sns.set(font_scale=1.5)
    # # sns.heatmap(df_raw.corr(), annot=False, cmap='coolwarm')
    # sns.heatmap(np.corrcoef(train_data_std.T), annot=False, cmap='coolwarm')
    # plt.title('Heatmap of Origin Data PEMS08')
    # plt.show()

    from kshape.core import KShapeClusteringCPU
    from kshape.core_gpu import KShapeClusteringGPU

    univariate_ts_datasets = np.expand_dims(df_raw_std.T, axis=2)
    # univariate_ts_datasets = np.expand_dims(df_raw_std.T, axis=2)  # patch相当于把一元变成多元
    num_clusters = 3
    # # CPU Model
    # ksc = KShapeClusteringCPU(num_clusters, centroid_init='zero', max_iter=100, n_jobs=-1)
    # ksc.fit(univariate_ts_datasets)
    #
    # labels = ksc.labels_ # or ksc.predict(univariate_ts_datasets)
    # cluster_centroids = ksc.centroids_

    # # GPU Model
    # ksg = KShapeClusteringGPU(num_clusters, centroid_init='zero', max_iter=100)
    # ksg.fit(univariate_ts_datasets)
    #
    # labels = ksg.labels_
    # cluster_centroids = ksg.centroids_.detach().cpu().squeeze()

    from sklearn.cluster import KMeans
    # 初始化KMeans对象
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)  # 假设我们要分成3个簇
    # 对数据进行聚类
    kmeans.fit(train_data_std.T)
    # 获取聚类中心
    cluster_centroids = kmeans.cluster_centers_
    print(cluster_centroids.shape, cluster_centroids)
    # 获取每个样本的聚类标签
    labels = kmeans.labels_

    print(labels.shape, labels, np.unique(labels))
    print(cluster_centroids.shape, cluster_centroids)
    # 计算每个类别的数量
    label_counts = np.bincount(np.int64(labels))

    # 打印每个类别的数量
    for label, count in enumerate(label_counts):
        print(f'Category {label}: {count} sequences')

    # 建立一个字典，用于保存聚类以后每一类的变量index
    label_dict = {}
    for label in np.unique(labels):
        if 'label' not in label_dict:
            label_dict[label] = list(np.where(labels == label)[0])
        # print(df_raw.iloc[:, np.where(labels == label)[0]])
    print(label_dict)


    # 颜色定义
    colors = ['red', 'blue', 'green']
    light_colors = ['lightcoral', 'lightblue', 'lightgreen']
    # 绘制原始序列和中心序列
    for i, center in enumerate(cluster_centroids):
        plt.figure(figsize=(12, 9))
        # cluster_data = df_raw_std[:, labels == i]
        cluster_data = train_data_std[:, labels == i]
        plt.plot(cluster_data[-1000:], color=light_colors[i%3], alpha=0.5)
        plt.plot(center[-1000:], '--', color=colors[i%3], linewidth=2, label=f'Center {i + 1}')
        plt.xlabel('Sequence Index')
        plt.ylabel('Value')
        plt.title(f'Clustered Sequences and Their Centers for cluster {i}')
        plt.show()

    # 按聚类结果排序
    sorted_indices = np.argsort(labels)
    # sorted_data = df_raw_std[:, sorted_indices]
    sorted_data = train_data_std[:, sorted_indices]

    # 使用seaborn绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.corrcoef(sorted_data.T), cmap='coolwarm')
    plt.title('Heatmap of Clustered Data for PEMS03')
    plt.xlabel('Sequence Index')
    plt.ylabel('Data Points')
    plt.show()


