import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

# 假设你有一个 n x m 的数据矩阵，其中 n 是样本数，m 是特征数
data = np.random.rand(3, 5)  # 示例数据

# 计算皮尔逊相关系数矩阵
correlation_matrix = np.corrcoef(data)

# 将相关系数矩阵转换为距离矩阵
# 使用 1 - 相关系数 作为“距离”，相关系数越高，距离越低
distance_matrix = 1 - np.abs(correlation_matrix)
print('corr=', correlation_matrix)
print('dist=', distance_matrix)
# 使用自定义距离矩阵进行 K-means 聚类
# 由于 K-means 不能直接使用距离矩阵，因此需要一种替代方法
# 可以通过多维尺度分析（MDS）将距离矩阵转换为嵌入空间

from sklearn.manifold import MDS

# 将距离矩阵转化为二维坐标，用于K-means聚类
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
embedding = mds.fit_transform(distance_matrix)
print(embedding)

# 在转换后的坐标系上应用 K-means
kmeans = KMeans(n_clusters=3, random_state=0).fit(embedding)

# 输出聚类标签
print("Cluster labels:", kmeans.labels_)




# import numpy as np
#
# ## A noisy sine wave as query
# idx = np.linspace(0,6.28,num=100)
# query = np.sin(idx) + np.random.uniform(size=100)/10.0
#
# ## A cosine is for template; sin and cos are offset by 25 samples
# template = np.cos(idx)
#
# ## Find the best match with the canonical recursion formula
# from dtw import dtw
#
#
#
# alignment = dtw(query, template, distance)
#
# ## Display the warping curve, i.e. the alignment curve
# alignment.plot(type="threeway")