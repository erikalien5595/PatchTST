import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

df = pd.read_csv('./wandb_export_2025-04-09T21_57_16.281+08_00.csv')
df['dataset'] = df['data_path'].apply(lambda x: x.split('.')[0])

for hyper_param in ['learning_rate', 'batch_size', 'dropout', 'e_layers',
                    'd_model', 'd_state', 'd_ff']:
    df_tmp = df.loc[df['des'] == hyper_param + '敏感性']
    if hyper_param=='batch_size':
        df_tmp.drop(df_tmp[df_tmp[f'{hyper_param}'] == 16].index, inplace=True)
    if hyper_param in ['d_ff', 'd_model']:
        df_tmp.drop(df_tmp[df_tmp[f'{hyper_param}'] == 32].index, inplace=True)
    plt.figure(figsize=(12, 8))

    # 收集所有数据集的所有 x 值，用于统一设置 xticks（假设都是相同的取值）
    all_x_values = []
    for i in df_tmp.dataset.unique():
        x_values = df_tmp.loc[df_tmp['dataset'] == i, hyper_param].values
        all_x_values.extend(x_values)

    # 对 x 轴取值去重并排序（假设超参数的取值是可以排序的）
    all_x_values = sorted(list(set(all_x_values)))
    x_ticks_positions = range(len(all_x_values))

    for i in df_tmp.dataset.unique():
        # 建议使用不同的x轴坐标，基于 all_x_values 的索引
        x_values = df_tmp.loc[df_tmp['dataset'] == i, hyper_param].values
        # 将当前 dataset 的 x_values 转换为索引位置
        x_positions = [all_x_values.index(val) for val in x_values]
        plt.plot(x_positions,
                 df_tmp.loc[df_tmp['dataset'] == i, 'test_mse'].values,
                 '.-', label=f'{i}')

    # 最后统一设置 x 轴的刻度和标签，并加上旋转角度减少重叠
    plt.xticks(ticks=x_ticks_positions, labels=all_x_values, rotation=45)
    plt.xlabel(hyper_param)
    plt.ylabel('test_mse')
    plt.legend(loc='upper right')
    plt.title(f'{hyper_param}')
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
    for i in df_tmp.dataset.unique():
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