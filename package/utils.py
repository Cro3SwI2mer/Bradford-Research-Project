import pandas as pd


def column_dict(dataframe: pd.DataFrame, column: str):
    result = dict.fromkeys(sorted(list(set(dataframe[column]))), 0)
    for key in dataframe[column]:
        result[key] += 1
    return result


def normalize(matrix):
    matrix = matrix.T
    for row in matrix:
        s = sum(row)
        for i in range(len(list(row))):
            row[i] = row[i] / s
    return matrix.T


def add_diagonal(matrix, elem):
    for i in range(len(matrix)):
        matrix[i][i] += elem
    return matrix


def power(matrix, r):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = matrix[i][j] ** r
    return matrix


def extremums(matrix):
    indexes = []
    for i in range(len(matrix)):
        l = [j for j in range(len(matrix[i])) if matrix[i][j] == max(matrix[i])]
        l.append(i)
        indexes.append(set(l))
    return indexes


def merge_sets(sets):
    result = []
    for candidate in sets:
        for current in result:
            if candidate & current:
                current |= candidate
                result = merge_sets(result)
                break
        else:
            result.append(candidate)
    return result
