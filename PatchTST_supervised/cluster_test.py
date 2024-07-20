import os
import numpy as np
import pandas as pd
from torch import nn
import torch
from  data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)

if __name__=='__main__':
    scaler = StandardScaler()
    flag = 'train'
    cluster_method = 'kmeans'
    num_clusters = 5
    seq_len = 96
    data_name = 'weather'
    type_map = {'train': 0, 'val': 1, 'test': 2}
    set_type = type_map[flag]
    ######################### PEMS #########################
    if data_name.startswith('PEMS'):
        data_file = f'./dataset/PEMS/{data_name}.npz'
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]
        df_raw = pd.DataFrame(data)
    ######################### Solar-Energy #########################
    elif data_name == 'Solar':
        df_raw = []
        with open('./dataset/solar_AL.txt', "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

    ######################### ECL, traffic, weather, etc #########################
    else:
        if data_name == 'ETTh1':
            df_raw = pd.read_csv('./dataset/ETT-small/ETTh1.csv')
        elif data_name == 'ETTh2':
                df_raw = pd.read_csv('./dataset/ETT-small/ETTh2.csv')
        elif data_name == 'ETTm1':
            df_raw = pd.read_csv('./dataset/ETT-small/ETTm1.csv')
        elif data_name == 'ETTm2':
            df_raw = pd.read_csv('./dataset/ETT-small/ETTm2.csv')
        elif data_name == 'ECL':
            df_raw = pd.read_csv('./dataset/electricity.csv')
        elif data_name == 'traffic':
            df_raw = pd.read_csv('./dataset/traffic.csv').iloc[:2000]
        elif data_name == 'weather':
            df_raw = pd.read_csv('./dataset/weather.csv')
            # 异常值处理
            df_raw.loc[df_raw['OT'] == -9999, 'OT'] = 417 #替换成均值
            df_raw.loc[df_raw['max. PAR (�mol/m�/s)'] == -9999, 'max. PAR (�mol/m�/s)'] = 0
            df_raw.loc[df_raw['wv (m/s)'] == -9999, 'wv (m/s)'] = 0
        elif data_name == 'exchange':
            df_raw = pd.read_csv('./dataset/exchange_rate/exchange_rate.csv')
        else:
            df_raw = pd.read_csv('./dataset/illness/national_illness.csv')
        cols = list(df_raw.columns)
        cols.remove('OT')
        cols.remove('date')
        df_raw = df_raw[cols + ['OT']]

    print(df_raw.shape)
    # print(df_raw.describe())
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, len(df_raw)]
    border1 = border1s[set_type]
    border2 = border2s[set_type]
    df_data = df_raw.values
    train_data = df_data[border1s[0]:border2s[0]]

    scaler.fit(train_data)
    train_data_std = scaler.transform(train_data)

    # 创建一个热力图
    plt.figure(figsize=[16, 9])
    sns.set(font_scale=1.5)
    # sns.heatmap(df_raw.corr(), annot=False, cmap='coolwarm')
    sns.heatmap(np.corrcoef(train_data_std.T), annot=False, cmap='coolwarm')
    plt.title(f'Heatmap of Origin Data {data_name}')
    plt.show()

    from kshape.core import KShapeClusteringCPU
    from kshape.core_gpu import KShapeClusteringGPU

    if cluster_method == 'kmeans':
        # 初始化KMeans对象
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)  # 假设我们要分成3个簇
        # 对数据进行聚类
        kmeans.fit(train_data_std.T)  # PEMS
        # kmeans.fit(df_raw_std.T)  # ECL, traffic, weather
        # 获取聚类中心
        cluster_centroids = kmeans.cluster_centers_
        # 获取每个样本的聚类标签
        labels = kmeans.labels_
    elif cluster_method == 'kshape':
        univariate_ts_datasets = np.expand_dims(train_data_std.T, axis=2)
        ##### CPU Model #####
        ksc = KShapeClusteringCPU(num_clusters, centroid_init='zero', max_iter=100, n_jobs=-1)
        ksc.fit(univariate_ts_datasets)
        labels = ksc.labels_ # or ksc.predict(univariate_ts_datasets)
        cluster_centroids = ksc.centroids_

        # ##### GPU Model #####
        ksg = KShapeClusteringGPU(num_clusters, centroid_init='zero', max_iter=100)
        # ksg.fit(univariate_ts_datasets)
        # labels = ksg.labels_
        # cluster_centroids = ksg.centroids_.detach().cpu().squeeze()
    else:
        pass
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
    colors = ['red', 'blue', 'green', 'black', 'orange']
    light_colors = ['lightcoral', 'lightblue', 'lightgreen', 'grey', 'pink']
    # # 绘制原始序列和中心序列
    # for i, center in enumerate(cluster_centroids):
    #     plt.figure(figsize=(12, 9))
    #     # cluster_data = df_raw_std[:, labels == i]
    #     cluster_data = train_data_std[:, labels == i]
    #     plt.plot(cluster_data[0:], color=light_colors[i%3], alpha=0.5)
    #     plt.plot(center[0:], '--', color=colors[i%3], linewidth=2, label=f'Center {i + 1}')
    #     plt.xlabel('Sequence Index')
    #     plt.ylabel('Value')
    #     plt.title(f'Clustered Sequences and Their Centers for cluster {i}')
    #     plt.show()
    # 创建多个子图
    fig, axes = plt.subplots(num_clusters, 1, figsize=(12, 9 * num_clusters))

    # 如果只有一个子图，axes不会是列表，需要转为列表
    if num_clusters == 1:
        axes = [axes]

    # 绘制每个聚类的原始序列和中心序列
    for i, (center, ax) in enumerate(zip(cluster_centroids, axes)):
        cluster_data = train_data_std[:, labels == i]
        ax.plot(cluster_data, color=light_colors[i % len(light_colors)], alpha=0.5)
        ax.plot(center, '--', color=colors[i % len(colors)], linewidth=2, label=f'Center {i + 1}')
        ax.set_xlabel('Sequence Index')
        ax.set_ylabel('Value')
        ax.set_title(f'Clustered Sequences and Their Centers for cluster {i}')
        ax.legend()

    # 调整布局
    plt.tight_layout()
    plt.show()

    # 按聚类结果排序
    sorted_indices = np.argsort(labels)
    # sorted_data = df_raw_std[:, sorted_indices]
    sorted_data = train_data_std[:, sorted_indices]
    for i in label_dict:
        corr = np.corrcoef(train_data_std[:, label_dict[i]].T)
        if len(label_dict[i])>1:
            # 生成一个布尔掩码矩阵，掩盖对角元素
            mask = ~np.eye(corr.shape[0], dtype=bool)
            # 提取非对角元素
            non_diagonal_elements = corr[mask]
            print(f"第{i}类：", np.mean(np.abs(non_diagonal_elements)))
        else:
            print(f"第{i}类只有1个序列")
    # 使用seaborn绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.corrcoef(sorted_data.T), cmap='coolwarm', annot=True)
    plt.title(f'Heatmap of Clustered Data for {data_name}')
    plt.xlabel('Sequence Index')
    plt.ylabel('Data Points')
    plt.show()


