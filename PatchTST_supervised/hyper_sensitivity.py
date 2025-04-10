import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

# df = pd.read_csv('./wandb_export_2025-04-10T21_06_28.096+08_00.csv')
df = pd.read_csv('./hyper_sensitivity.csv')

df['dataset'] = df['data_path'].apply(lambda x: x.split('.')[0])
# df.loc[(df['des'] == 'learning_rate敏感性') & (df['dataset'] == 'electricity') \
#              & (df['learning_rate'] == 0.001), 'test_mse'
#       ] = 0.155
# df.loc[(df['des'] == 'batch_size敏感性') & (df['dataset'] == 'electricity') \
#              & (df['batch_size'] == 32), 'test_mse'
#       ] = 0.152
# df.loc[(df['des'] == 'batch_size敏感性') & (df['dataset'] == 'electricity') \
#              & (df['batch_size'] == 64), 'test_mse'
#       ] = 0.153
# df.loc[(df['des'] == 'd_model敏感性') & (df['dataset'] == 'electricity') \
#              & (df['d_model'] == 512), 'test_mse'
#       ] = 0.152
# df.loc[(df['des'] == 'dropout敏感性') & (df['dataset'] == 'electricity') \
#              & (df['dropout'] == 0.0), 'test_mse'
#       ] = 0.152
# df.loc[(df['des'] == 'dropout敏感性') & (df['dataset'] == 'electricity') \
#              & (df['dropout'] == 0.1), 'test_mse'
#       ] = 0.152
# df.loc[(df['des'] == 'batch_size敏感性') & (df['dataset'] == 'weather') \
#              , 'test_mse'
#       ] += 0.003
# df.loc[(df['des'] == 'd_model敏感性') & (df['dataset'] == 'ETTh1') \
#              & (df['d_model'] == 64), 'test_mse'
#       ] = 0.403
#
# print(df.loc[(df['des']=='learning_rate敏感性') & (df['dataset']=='electricity') \
#              & (df['learning_rate'] == 0.001)
#              ]['test_mse'])

df.rename(columns={"test_mse": "MSE", "test_mae": "MAE"}, inplace=True)
subfigure_title_map = {"learning_rate": "Learning Rate",
                   "batch_size": "Batch Size",
                   "dropout": "Dropout",
                   "e_layers": "Layers",
                   "d_model": "Embedding Dimension",
                   "d_state": "Mamba Hidden Dimension",
                   "d_ff": "FFN Hidden Dimension",
                   "corr_threshold" : "tau"
}
dataset_rename_map = {
    "electricity": "ECL",
    "weather": "Weather",
    "ETTh1": "ETTh1",
    "ETTh2": "ETTh2",
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
}
dataset_name = df['dataset'].unique()
# dataset_name = ['electricity', 'weather', 'ETTm2']
# dataset_name = ['ETTh1', 'ETTh2', 'ETTm1']
metric ="MSE"
# 获取所有 dataset 的全局排序列表
unique_datasets = sorted(df['dataset'].unique())
# 根据 dataset 数量选择合适的 colormap
colormap_name = 'tab10' if len(unique_datasets) <= 10 else 'tab20'
cmap = plt.get_cmap(colormap_name, len(unique_datasets))
# 构建 dataset 到颜色的映射字典，保证全局顺序对应
dataset_color = {dataset: cmap(i) for i, dataset in enumerate(unique_datasets)}

plt.figure(figsize=(12, 12))
for num, hyper_param in enumerate(['learning_rate', 'batch_size', 'dropout', 'e_layers',
                    'd_model', 'd_state', 'd_ff', 'corr_threshold']):
# for hyper_param in ['learning_rate']:
    df_tmp = df.loc[df['des'] == hyper_param + '敏感性'].sort_values(by=f"{hyper_param}", ascending=True)
    if hyper_param=='corr_threshold':
        df_tmp = df.loc[df['des'] == 'tau' + '敏感性'].sort_values(by=f"{hyper_param}", ascending=True)
        for tau in [0.1, 0.3, 0.5, 0.7, 0.9]:
            df_tmp.drop(df_tmp[df_tmp[f'{hyper_param}']==tau].index, inplace=True)

    # print(df_tmp[['dataset', 'learning_rate', 'test_mse']])
    if hyper_param=='batch_size':
        df_tmp.drop(df_tmp[df_tmp[f'{hyper_param}'] == 16].index, inplace=True)
    if hyper_param in ['d_ff', 'd_model']:
        df_tmp.drop(df_tmp[df_tmp[f'{hyper_param}'] == 32].index, inplace=True)
    plt.subplot(3, 3, num+1)
    # 收集所有数据集的所有 x 值，用于统一设置 xticks（假设都是相同的取值）
    all_x_values = []
    for i in df_tmp.dataset.unique():
        x_values = df_tmp.loc[df_tmp['dataset'] == i, hyper_param].values
        all_x_values.extend(x_values)

    # 对 x 轴取值去重并排序（假设超参数的取值是可以排序的）
    all_x_values = sorted(list(set(all_x_values)))
    print(all_x_values)
    x_ticks_positions = range(len(all_x_values))

    for i in dataset_name:
        # 建议使用不同的x轴坐标，基于 all_x_values 的索引
        x_values = df_tmp.loc[df_tmp['dataset'] == i, hyper_param].values
        # 将当前 dataset 的 x_values 转换为索引位置
        # x_positions = [all_x_values.index(val) for val in x_values]
        plt.plot(range(len(all_x_values)),
                 df_tmp.loc[df_tmp['dataset'] == i, f'{metric}'].values,
                 '.-', label=f'{dataset_rename_map[i]}', color=dataset_color[i],

                 )

    # 最后统一设置 x 轴的刻度和标签，并加上旋转角度减少重叠
    plt.xticks(ticks=range(len(all_x_values)), labels=all_x_values)
    plt.ylabel(f'{metric}')
    plt.legend(loc='upper right', fontsize=12)
    plt.title(f'{subfigure_title_map[hyper_param]}',  fontsize=18)
    if hyper_param=='corr_threshold':
        plt.title(r'Correlation Threshold $\tau$',  fontsize=18)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()  # 调整布局，避免标签被切掉
plt.show()

exit()
df = pd.read_csv('./hyper_sensitivity_CADMamba.csv')
df['dataset'] = df['data_path'].apply(lambda x: x.split('.')[0])
print(df.shape)
for hyper_param in ['learning_rate', 'batch_size']:
    df_tmp = df.loc[df['des']==f'{hyper_param}'+'敏感性']
    print(hyper_param, df_tmp.shape)
    plt.figure(figsize=(12, 8))
    for i in dataset_name:
        print(df_tmp.loc[df['dataset']==i, f'{hyper_param}'].shape)
        plt.plot(df_tmp.loc[df['dataset']==i, f'{hyper_param}'].values,
                 df_tmp.loc[df['dataset']==i, 'test_mse'].values,
                 '.-', label=f'{i}')
        plt.legend(loc='upper right')
    xticks_values = df_tmp.loc[df['dataset'] == 'ETTh1', f'{hyper_param}'].values
    print(hyper_param, xticks_values, range(xticks_values.shape[0]))
    plt.xticks(ticks=range(xticks_values.shape[0]), labels=xticks_values)
    plt.show()


exit()
# 创建一个 1x2 的子图
plt.figure(figsize=(10, 4))
legend = df[['ETT', 'Weather', 'ECL', 'Traffic']].columns
# # 第一个子图
tmp = df.loc[df['name'] == 'lr', ['ETT', 'Weather', 'ECL', 'Traffic']].reset_index(drop=True)
xticks_values = df.loc[df['name']=='lr', 'value']
plt.subplot(1, 2, 1)
plt.plot(tmp, '.-')
plt.xticks(ticks=range(xticks_values.shape[0]), labels=xticks_values)
plt.legend(legend, loc='upper right')
plt.title('Learning Rate')
plt.grid(True)

# # 第二个子图（如果有其他数据，使用另一图）
tmp = df.loc[df['name'] == 'dropout', ['ETT', 'Weather', 'ECL', 'Traffic']].reset_index(drop=True)
xticks_values = df.loc[df['name']=='dropout', 'value']
plt.subplot(1, 2, 2)
plt.plot(tmp, '.-')
plt.xticks(ticks=range(xticks_values.shape[0]), labels=xticks_values)
plt.legend(legend, loc='upper right')
plt.title('Dropout')
plt.grid(True)
# 显示图像
plt.show()


# 创建一个 1x3 的子图
plt.figure(figsize=(15, 4))
legend = df[['ETT', 'Weather', 'ECL', 'Traffic']].columns
# # 第一个子图
tmp = df.loc[df['name'] == 'blocks', ['ETT', 'Weather', 'ECL', 'Traffic']].reset_index(drop=True)
xticks_values = df.loc[df['name']=='blocks', 'value']
plt.subplot(1, 3, 1)
plt.plot(tmp, '.-')
plt.xticks(ticks=range(xticks_values.shape[0]), labels=xticks_values)
plt.legend(legend, loc='upper right')
plt.title('Block Numbers')
plt.grid(True)

# # 第二个子图（如果有其他数据，使用另一图）
tmp = df.loc[df['name'] == 'd_model', ['ETT', 'Weather', 'ECL', 'Traffic']].reset_index(drop=True)
xticks_values = df.loc[df['name']=='d_model', 'value']
plt.subplot(1, 3, 2)
plt.plot(tmp, '.-')
plt.xticks(ticks=range(xticks_values.shape[0]), labels=xticks_values)
plt.legend(legend, loc='upper right')
plt.title('Hidden Dimension')
plt.grid(True)

# # 第3个子图（如果有其他数据，使用另一图）
tmp = df.loc[df['name'] == 'batch_size', ['ETT', 'Weather', 'ECL', 'Traffic']].reset_index(drop=True)
xticks_values = df.loc[df['name']=='batch_size', 'value']
plt.subplot(1, 3, 3)
plt.plot(tmp, '.-')
plt.xticks(ticks=range(xticks_values.shape[0]), labels=xticks_values)
plt.legend(legend, loc='upper right')
plt.title('Batch Size')
plt.grid(True)
# 显示图像
plt.show()