import pandas as pd
import numpy as np
from math import log
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

# TODO:
#  1. Add KS-distance method to Stat


class Discrete:

    """
    Constructs a frequency map of appearance of a given parameter into particular area
    """

    def __init__(self, dataframe: pd.DataFrame, *variables, parameter: str) -> None:
        self.dataframe = dataframe
        self.variables = variables
        self.parameter = parameter

    @staticmethod
    def __product(args):
        result = [[]]
        for arg in args:
            result = [x + [y] for x in result for y in arg]
        return [tuple(elem) for elem in result]

    def frequency(self):
        names = sorted(list(set(self.dataframe[self.parameter])))
        final_dict = dict.fromkeys(names)
        indexes = [list(self.dataframe.columns).index(variable) for variable in self.variables]
        splits = [list(range(1 + max(self.dataframe[variable]))) for variable in self.variables]
        for name in names:
            values = dict.fromkeys(Discrete.__product(splits), 0)
            data = np.array(self.dataframe.loc[self.dataframe[self.parameter] == name])
            for i in range(len(data)):
                coordinates = [data[i][index] for index in indexes]
                values[tuple(coordinates)] += 1
            final_dict[name] = values
        return final_dict


class PDF(Discrete):

    """
    Constructs a discrete pdf for a given parameter with one of the described methods
    """

    def __init__(self, dataframe: pd.DataFrame, *variables, parameter: str, normalize=True) -> None:
        super(PDF, self).__init__(dataframe, *variables, parameter=parameter)
        self.normalize = normalize
        self._frequency_data = self.frequency()

    @property
    def frequency_data(self):
        return self._frequency_data

    def __dimension(self):
        keys = list(self._frequency_data.keys())
        coordinates = list(self._frequency_data[keys[0]].keys())
        list_coordinates = [list(elem) for elem in coordinates]
        splits = [max(elem) + 1 for elem in np.array(list_coordinates).T]
        return splits

    def diagonal(self):
        assert len(set(PDF.__dimension(self))) == 1
        num = PDF.__dimension(self)[0]
        discrete_pdf = dict.fromkeys(list(self._frequency_data.keys()))
        for key in discrete_pdf:
            agg_result = dict.fromkeys(range(num), 0)
            single_data = self._frequency_data[key]
            for coordinate in single_data.keys():
                agg_result[max(coordinate)] += single_data[coordinate]
            if self.normalize:
                discrete_pdf[key] = [elem / sum(list(agg_result.values())) for elem in list(agg_result.values())]
            else:
                discrete_pdf[key] = list(agg_result.values())
        return discrete_pdf

    def summation(self):
        discrete_pdf = dict.fromkeys(list(self._frequency_data.keys()))
        for key in discrete_pdf:
            coordinates = [list(elem) for elem in list(self._frequency_data[key].keys())]
            agg_result = dict.fromkeys(sorted(list(set(sum(elem) for elem in coordinates))), 0)
            single_data = self._frequency_data[key]
            for coordinate in list(single_data.keys()):
                single_data[sum(coordinate)] = single_data.pop(coordinate)
            for coordinates_sum in single_data.keys():
                agg_result[coordinates_sum] += single_data[coordinates_sum]
            if self.normalize:
                discrete_pdf[key] = [elem / sum(list(agg_result.values())) for elem in list(agg_result.values())]
            else:
                discrete_pdf[key] = list(agg_result.values())
        return discrete_pdf

    def linear(self):
        discrete_pdf = dict.fromkeys(list(self._frequency_data.keys()))
        for key in discrete_pdf:
            if self.normalize:
                discrete_pdf[key] = [freq / sum(list(self._frequency_data[key].values()))
                                     for freq in list(self._frequency_data[key].values())]
            else:
                discrete_pdf[key] = list(self._frequency_data[key].values())
        return discrete_pdf


class Stat(PDF):

    """
    Constructs a divergence matrix, calculated between all ordered pairs of given parameter
    """

    def __init__(self, dataframe: pd.DataFrame, *variables, parameter: str, approach: str,
                 normalize=True, return_keys=True) -> None:
        super(Stat, self).__init__(dataframe, *variables, parameter=parameter, normalize=normalize)
        self.approach = approach
        self._discrete_pdf_data = getattr(self, approach)()
        self._keys, self._values = list(self._discrete_pdf_data.keys()), list(self._discrete_pdf_data.values())
        self.return_keys = return_keys

    @property
    def discrete_pdf_data(self):
        return self._discrete_pdf_data

    def __return_result(self, matrix: np.array):
        if self.return_keys:
            return self._keys, np.array(matrix)
        else:
            return np.array(matrix)

    def kl_divergence(self):
        divergence = [[entropy(p, q) for q in self._values] for p in self._values]
        return Stat.__return_result(self, divergence)

    def js_divergence(self):
        divergence = [[jensenshannon(p, q) for q in self._values] for p in self._values]
        return Stat.__return_result(self, divergence)

    def ab_divergence(self, a: float, b: float):
        assert self.normalize is True
        num = np.array(self._values).shape[1]
        divergence = []
        for p in self._values:
            inside_list = []
            for q in self._values:
                div = 0
                if a + b != 0:
                    div = sum((-1/(a*b))*(p[i]**a*q[i]**b-(a/(a+b))*p[i]**(a+b)-(b/(a+b))*q[i]**(a+b))
                              for i in range(num))
                elif a != 0 and b == 0:
                    div = sum((1/a**2)*(p[i]**a*log(p[i]**a/q[i]**a)-p[i]**a+q[i]**a) for i in range(num))
                elif a == -1*b and b != 0:
                    div = sum((1/a**2)*(log(q[i]**a/p[i]**a)+1/(q[i]**a/p[i]**a)-1) for i in range(num))
                elif a == 0 and b != 0:
                    div = sum((1/b**2)*(q[i]**b*log(q[i]**b/p[i]**b)-q[i]**b+p[i]**b) for i in range(num))
                elif a == b == 0:
                    div = sum(0.5*(log(p[i])-log(q[i]))**2 for i in range(num))
                inside_list.append(div)
            divergence.append(inside_list)
        return Stat.__return_result(self, divergence)


__all__ = ['Discrete', 'PDF', 'Stat']
