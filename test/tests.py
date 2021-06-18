import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from tqdm import tqdm

from package import preprocessor, numerical, metrics
from package.utils import *

# TODO:
#  1. Test package


# paths to the data file, put your own
project_path = "C:/Users/Dell XPS 13/Documents/Research in Bradford, fall 2020/Project/"
data = pd.read_csv(project_path+"Vehicle data/TripStatistics_data_updated_timezones.csv")
# test_vins = list(pd.read_excel(project_path+"Vehicle data/Predictive_model_performance.xlsx", sheet_name='Sheet1')['VIN'])

# print(dataframe.info())
# print(preprocessor.Transform(dataframe, new_data=True).log_transformation('agg_traveling_duration_min'))
# print(preprocessor.Adder(dataframe, new_data=True).quantile('agg_traveling_duration_min', num_of_quantiles=6))

preprocessor.TimeModifier(data).change_columns('StartTime', 'StopTime')
preprocessor.Adder(data).duration('StartTime', 'StopTime')

# choose necessary subset
# data = data.loc[dataframe['VIN'].isin(test_vins)].reset_index(drop=True)

# добавляем колонки
preprocessor.Adder(data).time_zone('StartTime', num_zones=4)
preprocessor.Adder(data).zone('duration', num_zones=[4])

# print(utils.column_dict(data, 'StartTime_zone'))

# vin = data.loc[data['VIN'] == 'VIN_1'].reset_index(drop=True)

# pdfs = numerical.PDF(data,
#                      'StartTime_zone', 'duration_zone',
#                      parameter='VIN').summation()
#
nodes, matrix = numerical.Stat(data, 'StartTime_zone', 'duration_zone', parameter='VIN', approach='summation',
                               return_keys=True).js_divergence()

# print(matrix)

r = 2
iterations = 10

new = normalize(add_diagonal(matrix.copy(), 1.0))

inside, outside = [], []
for i in (range(iterations)):
    new = normalize(power(new @ new, r))
    clusters = merge_sets(extremums(new))
    result = [list(cluster) for cluster in merge_sets(clusters)]
    d = dict.fromkeys([i for i in range(len(result))])
    for key in d.keys():
        d[key] = result[key]
    ins = metrics.Metric(d, matrix).fit(metric='inside', approach='avg')
    out = metrics.Metric(d, matrix).fit(metric='outside', approach='avg')
    inside.append(ins)
    outside.append(out)

    print('iteration: ', i + 1)
    print('number of clusters: ', len(d.keys()))
    print('inside: ', round(ins, 4))
    print('outside: ', round(out, 4))

    print('\n')

# test_graph = graph.Graph(nodes, matrix)

# print(test_graph.dataframe_format())

# test_data = pd.DataFrame.from_dict(pdfs, orient="index")
# X = np.array(test_data)
#
# clustering = KMeans(n_clusters=5, random_state=0).fit(X)
# labels = list(clustering.labels_)
#
# d = []
# for i in range(matrix.shape[0]):
#     key = labels[i]
#     d.append([key, i])
# new = [[] for i in range(5)]
# for elem in d:
#     new[elem[0]].append(elem[1])
# clusters = dict.fromkeys(sorted(list(set(labels))))
# for key in clusters.keys():
#     clusters[key] = new[key]
#
# result = metrics.Metric(clusters, matrix).internal("avg")
#
# print(result)

# # K-Means
# n_clusters = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#
# # DBSCAN
# eps = [0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04]
#
# for num in n_clusters:
#     k_means = KMeans(n_clusters=num, random_state=0).fit(X)
#     print(k_means.labels_)
#     test_data['KMeans_' + str(num) + '_clusters'] = k_means.labels_
#
# for num in eps:
#     dbscan = DBSCAN(eps=num, min_samples=1).fit(X)
#     print(dbscan.labels_)
#     test_data['DBSCAN_' + str(num) + '_eps'] = dbscan.labels_

# test_data.to_excel(project_path+"Vehicle data/test_clustering.xlsx", sheet_name="clusters")