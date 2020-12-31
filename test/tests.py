import pandas as pd
from package import preprocessor, numerical, graph

# paths to the data file, put your own
project_path = "C:/Users/Dell XPS 13/Documents/Research in Bradford, fall 2020/Project/"
dataframe = pd.read_csv(project_path+"Vehicle data/TripStatistics_data_updated_timezones.csv")
test_vins = list(pd.read_excel(project_path+"Vehicle data/Predictive_model_performance.xlsx", sheet_name='Sheet1')['VIN'])

# print(dataframe.info())

# print(preprocessor.Transform(dataframe, new_data=True).log_transformation('agg_traveling_duration_min'))
# print(preprocessor.Adder(dataframe, new_data=True).quantile('agg_traveling_duration_min', num_of_quantiles=6))

preprocessor.TimeModifier(dataframe).change_columns('StartTime', 'StopTime')
preprocessor.Adder(dataframe).duration('StartTime', 'StopTime')

data = dataframe.loc[dataframe['VIN'].isin(test_vins)].reset_index(drop=True)  # choose necessary subset
preprocessor.Adder(data).time_zone('StartTime', num_of_zones=4)
preprocessor.Adder(data).duration_zone(num_of_zones=4)

# vin = data.loc[data['VIN'] == 'VIN_1'].reset_index(drop=True)

nodes, matrix = numerical.Stat(data, 'StartTime_zone', 'duration_zone', parameter='VIN', approach='diagonal',
                               return_keys=True).js_divergence()

test_graph = graph.Graph(nodes, matrix)
# l = list(test_graph.dataframe_format()['VIN_1'])[1:]

print(test_graph.dataframe_format())
