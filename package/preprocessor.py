import pandas as pd
import datetime as dt
from math import log, exp

# TODO: nothing


def binary_search(li, x):
    i = 0
    j = len(li) - 1
    m = int(j / 2)
    while (li[m] > x or x > li[m + 1]) and i < j:
        if x > li[m + 1]:
            i = m + 1
        elif x < li[m]:
            j = m - 1
        else:
            raise ValueError('Check data or algorithm')
        m = int((i + j) / 2)
    if x <= li[m]:
        return m
    elif x <= li[m + 1]:
        return m+1
    else:
        raise ValueError('Check data or algorithm')


class Descriptor:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
        self.columns = self.dataframe.columns

    @staticmethod
    def add_to_dict(d: dict, data):
        keys = list(d.keys())
        for i in range(len(keys)):
            d[keys[i]] = data[i]

    def info(self):
        data = []
        d = dict.fromkeys(sorted([str(elem) for elem in list(set(self.dataframe.dtypes))]))
        for key in d.keys():
            di = {}
            for elem in self.columns:
                if str(self.dataframe[elem].dtype) == key:
                    di[elem] = len(set(self.dataframe[elem]))
                else:
                    pass
            data.append(di)
        self.add_to_dict(d, data)
        return d


class TimeModifier:
    def __init__(self, dataframe: pd.DataFrame, new_data=False) -> None:
        self.dataframe = dataframe
        self._new_data = new_data

    def change_columns(self, *columns):
        for column in columns:
            self.dataframe[column] = pd.to_datetime(self.dataframe[column])
        if self._new_data:
            return self.dataframe

    def set_new_columns(self, *columns):
        for column in columns:
            self.dataframe[str(column) + '_dtf'] = pd.to_datetime(self.dataframe[column])
        if self._new_data:
            return self.dataframe


class Transform:
    def __init__(self, dataframe: pd.DataFrame, new_data=False) -> None:
        self.dataframe = dataframe
        self._new_data = new_data

    def log_transformation(self, *columns, base=exp(1)):
        for column in columns:
            assert column not in list(Descriptor(self.dataframe).info()['object'].keys())
        for column in columns:
            self.dataframe[column] = [log(elem, base=base) for elem in list(self.dataframe[column])]
        if self._new_data:
            return self.dataframe


class Adder:
    def __init__(self, dataframe: pd.DataFrame, new_data=False) -> None:
        self.dataframe = dataframe
        self._new_data = new_data

    def __types(self):
        return list(str(self.dataframe.dtypes[i]) for i in range(len(self.dataframe.dtypes)))

    def duration(self, start, stop):
        if 'datetime64[ns]' not in Adder.__types(self):
            raise TypeError("No column match datetime64[ns] format")
        journey_durations = []
        for i in range(len(self.dataframe)):
            journey_time = (self.dataframe[stop][i] - self.dataframe[start][i]).seconds
            journey_durations.append(journey_time)
        self.dataframe['duration'] = journey_durations
        if self._new_data:
            return self.dataframe

    def week_day(self, *columns):
        if 'datetime64[ns]' not in Adder.__types(self):
            raise TypeError("No column match datetime64[ns] format")
        for column in columns:
            self.dataframe[column + '_day'] = self.dataframe[column].dt.weekday_name
        if self._new_data:
            return self.dataframe

    def time_zone(self, time_column, num_of_zones=6):
        timezones = []
        interval = 24 * 3600 / num_of_zones
        for i in range(len(self.dataframe)):
            delta = dt.datetime.strptime(str(self.dataframe[time_column][i]).split(' ')[1], "%H:%M:%S") - dt.datetime(
                1900, 1, 1, 0, 0, 0)
            timezones.append(int(delta.seconds // interval))
        self.dataframe[time_column + '_zone'] = timezones
        if self._new_data:
            return self.dataframe

    def duration_zone(self, num_of_zones=6):
        if 'duration' not in self.dataframe.columns:
            raise ValueError("No column named 'duration'")
        duration_zones = []
        ma, mi = max(self.dataframe['duration']), min(self.dataframe['duration'])
        interval = (ma - mi) / num_of_zones
        for i in range(len(self.dataframe)):
            if self.dataframe['duration'][i] == ma:
                duration_zones.append(num_of_zones - 1)
            else:
                duration_zones.append(int((self.dataframe['duration'][i] - mi) // interval))
        self.dataframe['duration_zone'] = duration_zones
        if self._new_data:
            return self.dataframe

    def zone(self, *columns, num_of_zones: int):
        for column in columns:
            assert column not in list(Descriptor(self.dataframe).info()['object'].keys())
        for column in columns:
            zones = []
            ma, mi = max(self.dataframe[column]), min(self.dataframe[column])
            interval = (ma - mi) / num_of_zones
            for i in range(len(self.dataframe)):
                if self.dataframe[column][i] == ma:
                    zones.append(num_of_zones - 1)
                else:
                    zones.append(int((self.dataframe['duration'][i] - mi) // interval))
            self.dataframe[column + '_zone'] = zones
        if self._new_data:
            return self.dataframe

    def quantile(self, *columns, num_of_quantiles: int):
        for column in columns:
            assert column not in list(Descriptor(self.dataframe).info()['object'].keys())
        quantiles = [(i + 1) / num_of_quantiles for i in range(num_of_quantiles)]
        for column in columns:
            column_values = sorted(list(self.dataframe[column]))
            quantile_indexes = [round(elem * (len(column_values) - 1)) for elem in quantiles]
            quantile_values = [column_values[index] for index in quantile_indexes]
            new_column_values = [binary_search(quantile_values, elem) for elem in list(self.dataframe[column])]
            self.dataframe[column + '_quantile'] = new_column_values
        if self._new_data:
            return self.dataframe


__all__ = ['Descriptor', 'TimeModifier', 'Transform', 'Adder']
