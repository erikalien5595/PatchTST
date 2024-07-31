import os
import numpy as np
import pandas as pd
from torch import nn
import torch
from  data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)

if __name__=='__main__':
    scaler = StandardScaler()
    flag = 'train'
    cluster_method = 'kmeans'
    num_clusters = 2
    seq_len = 96
    use_catch22 = 0
    data_name = 'Solar'
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
            df_raw = pd.read_csv('./dataset/traffic.csv')#.iloc[:2000]
        elif data_name == 'weather':
            df_raw = pd.read_csv('./dataset/weather.csv')
            # # 异常值处理
            # df_raw.loc[df_raw['OT'] == -9999, 'OT'] = 417  # 替换成均值
            # df_raw.loc[df_raw['OT'] <350, 'OT'] = 417  # 替换成均值
            # df_raw.loc[df_raw['max. PAR (�mol/m�/s)'] == -9999, 'max. PAR (�mol/m�/s)'] = 0
            # df_raw.loc[df_raw['wv (m/s)'] == -9999, 'wv (m/s)'] = 0
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
    if df_raw.shape[1]<=30:
        pdf_pages = PdfPages(f'./clusterResults/origin_series_{data_name}.pdf')
        for i in df_raw.columns:
            fig = plt.figure(figsize=[9, 6])
            plt.plot(df_raw[i], label='Train Error')
            plt.xlabel('time')
            plt.ylabel('value')
            plt.title(f'{i}')
            plt.legend()
            plt.grid(True)
            pdf_pages.savefig(fig, bbox_inches='tight')
            plt.close()
        pdf_pages.close()

    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, len(df_raw)]
    if data_name.startswith('ETTh'):
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    if data_name.startswith('ETTm'):
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    border1 = border1s[set_type]
    border2 = border2s[set_type]
    df_data = df_raw.copy()
    train_data = df_data[border1s[0]:border2s[0]]

    scaler.fit(train_data.values)
    train_data_std = scaler.transform(train_data.values)
    if df_raw.shape[1]<=30:
        pdf_pages = PdfPages(f'./clusterResults/standardized_series_{data_name}.pdf')
        for i in range(df_raw.shape[1]):
            fig = plt.figure(figsize=[9, 6])
            plt.plot(train_data_std[:, i], label='Train Error')
            plt.xlabel('time')
            plt.ylabel('value')
            plt.title(f'{i}')
            plt.legend()
            plt.grid(True)
            pdf_pages.savefig(fig, bbox_inches='tight')
            plt.close()
        pdf_pages.close()

    # # 创建一个热力图
    # plt.figure(figsize=[16, 9])
    # sns.set(font_scale=1.5)
    # sns.heatmap(np.corrcoef(train_data_std.T), annot=False, cmap='coolwarm')
    # plt.title(f'Heatmap of Origin Data {data_name}')
    # plt.show()

    from kshape.core import KShapeClusteringCPU
    from kshape.core_gpu import KShapeClusteringGPU

    if cluster_method == 'kmeans':
        # 初始化KMeans对象
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)  # 假设我们要分成3个簇
        if data_name == 'weather':
            train_data.loc[train_data['OT'] == -9999, 'OT'] = 417  # 替换成均值
            train_data.loc[train_data['OT'] < 350, 'OT'] = 417  # 替换成均值
            train_data.loc[train_data['max. PAR (�mol/m�/s)'] == -9999, 'max. PAR (�mol/m�/s)'] = 0
            train_data.loc[train_data['wv (m/s)'] == -9999, 'wv (m/s)'] = 0
            scaler2 = StandardScaler()
            if use_catch22 == 0:
                data2 = scaler2.fit_transform(train_data.values)
            # data2 = scaler.transform(train_data.values)
        else:
            data2 = train_data_std
        ###### catch22 to extract features for clustering ######
        if use_catch22==1:
            import pycatch22
            tmp = pycatch22.catch22_all(train_data.iloc[:, 0].values)
            data2 = pd.DataFrame(np.array(tmp['values']).reshape(1, -1), columns=tmp['names'])
            for i in range(1, train_data.shape[1]):
                tmp = pycatch22.catch22_all(train_data.iloc[:, i].values)
                new_line = pd.DataFrame(np.array(tmp['values']).reshape(1, -1), columns=tmp['names'])
                data2 = pd.concat([data2, new_line])
            data2 = data2.values.T
            scaler3 = StandardScaler()
            data2 = scaler3.fit_transform(data2)
        ########################################################
        print(data2.shape, data2.mean())
        k_score, k_shape = [], []
        for i in np.arange(2, 11):
        # for i in np.arange(2, np.min(8, data2.shape[1])):
            # 对数据进行聚类
            kmeans2 = KMeans(n_clusters=i, random_state=42)
            kmeans2.fit(data2.T)  # PEMS
            # kmeans.fit(df_raw_std.T)  # ECL, traffic, weather
            # 获取聚类中心
            cluster_centroids = kmeans2.cluster_centers_
            # 获取每个样本的聚类标签
            labels = kmeans2.labels_
            score = silhouette_score(data2.T, labels)
            k_score.append(score)
            k_shape.append(i)
            print(f'n={i}, silhouette_score={np.round(score, 4)}')
        dict_shape = dict(zip(k_shape, k_score))
        best_shape = sorted(dict_shape.items(), key=lambda x: x[1], reverse=True)[0][0]
        print(best_shape)

        # 对数据进行聚类
        kmeans.fit(data2.T)  # PEMS
        # kmeans.fit(df_raw_std.T)  # ECL, traffic, weather
        # 获取聚类中心
        cluster_centroids = kmeans.cluster_centers_
        # 获取每个样本的聚类标签
        labels = kmeans.labels_
        score = silhouette_score(data2.T, labels)
        print(f'silhouette_score={score}')
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
    # print(cluster_centroids.shape, cluster_centroids)
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

    # 按聚类结果排序
    sorted_indices = np.argsort(labels)
    # sorted_data = df_raw_std[:, sorted_indices]
    sorted_data = data2[:, sorted_indices]
    for i in label_dict:
        # corr = np.corrcoef(train_data_std[:, label_dict[i]].T)
        if len(label_dict[i])>2:
            # 生成一个布尔掩码矩阵，掩盖对角元素
            corr = stats.spearmanr(data2[:, label_dict[i]])[0]
            mask = ~np.eye(corr.shape[0], dtype=bool)
            # 提取非对角元素
            non_diagonal_elements = corr[mask]
            print(f"第{i}类：", np.round(np.mean(np.abs(non_diagonal_elements)), 5),
                  np.round(np.min(np.abs(non_diagonal_elements)), 5),
                  np.round(np.max(np.abs(non_diagonal_elements)), 5),
                  train_data.columns[label_dict[i]], np.var(np.abs(non_diagonal_elements)), np.mean(np.abs(non_diagonal_elements))+1/(1+np.std(np.abs(non_diagonal_elements))))
        elif len(label_dict[i])==2:
            corr = stats.spearmanr(data2[:, label_dict[i]])[0]
            print(f"第{i}类：", np.round(corr, 5), train_data.columns[label_dict[i]])
        else:
            print(f"第{i}类只有1个序列: {train_data.columns[label_dict[i]]}")
    # 使用seaborn绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.corrcoef(sorted_data.T), cmap='coolwarm', annot=False)
    plt.title(f'Heatmap of Clustered Data for {data_name}')
    plt.xlabel('Sequence Index')
    plt.ylabel('Data Points')
    plt.savefig(f'./clusterResults/heatmap_{data_name}_cluster{num_clusters}.png', bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(10, 8))
    sns.heatmap(stats.spearmanr(sorted_data)[0], cmap='coolwarm', annot=False)
    plt.title(f'Heatmap of Clustered Data for {data_name}(Spearman Correlation)')
    plt.xlabel('Sequence Index')
    plt.ylabel('Data Points')
    plt.savefig(f'./clusterResults/heatmap_{data_name}_cluster{num_clusters}_Spearman.png', bbox_inches='tight')
    plt.close()
    # sns.heatmap(stats.spearmanr(sorted_data)[0]-np.corrcoef(sorted_data.T), cmap='coolwarm', annot=False)
    # plt.title(f'Heatmap of Clustered Data for {data_name}(Correlation Difference)')
    # plt.xlabel('Sequence Index')
    # plt.ylabel('Data Points')
    # plt.savefig(f'./clusterResults/heatmap_{data_name}_cluster{num_clusters}_diff.png', bbox_inches='tight')
    # plt.close()

    # 颜色定义
    colors = ['red', 'blue', 'green', 'black', 'orange']
    light_colors = ['lightcoral', 'lightblue', 'lightgreen', 'grey', 'pink']
    # 绘制原始序列和中心序列
    pdf_pages = PdfPages(f'./clusterResults/{data_name}_cluster{num_clusters}.pdf')
    for i, center in enumerate(cluster_centroids):
        fig = plt.figure(figsize=[12, 9])
        # cluster_data = df_raw_std[:, labels == i]
        if use_catch22==1:
            cluster_data = data2[:, labels == i]
        else:
            cluster_data = train_data_std[:, labels == i]
        plt.plot(cluster_data[0:], color=light_colors[i%3], alpha=0.5)
        plt.plot(center[0:], '--', color=colors[i%3], linewidth=2, label=f'Center {i + 1}')
        plt.xlabel('Sequence Index')
        plt.ylabel('Value')
        plt.title(f'Clustered Sequences and Their Centers for cluster {i}')
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close()
    pdf_pages.close()

    # # 创建多个子图
    # fig, axes = plt.subplots(num_clusters, 1, figsize=(12, 9 * num_clusters))
    # # 如果只有一个子图，axes不会是列表，需要转为列表
    # if num_clusters == 1:
    #     axes = [axes]
    # # 绘制每个聚类的原始序列和中心序列
    # for i, (center, ax) in enumerate(zip(cluster_centroids, axes)):
    #     cluster_data = train_data_std[:, labels == i]
    #     ax.plot(cluster_data, color=light_colors[i % len(light_colors)], alpha=0.5)
    #     ax.plot(center, '--', color=colors[i % len(colors)], linewidth=2, label=f'Center {i + 1}')
    #     ax.set_xlabel('Sequence Index')
    #     ax.set_ylabel('Value')
    #     ax.set_title(f'Clustered Sequences and Their Centers for cluster {i}')
    #     ax.legend()

    # 调整布局
    plt.tight_layout()
    plt.show()


