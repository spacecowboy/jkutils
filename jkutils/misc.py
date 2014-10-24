"""
Functions which do not sort well under another name. Will all be imported
by __init__.py to package level.
"""

from __future__ import division
import numpy as np


def normalize_panda(dataframe, cols):
    '''
    Normalize a pandas dataframe. Binary values are
    forced to 0-1, and continuous (the rest) variables
    are forced to zero mean and standard deviation = 1

    Parameters:
    - dataframe, the pandas dataframe to normalize column-wise
    - cols, (iterable) the column names in the dataframe to normalize.
    '''
    for col in cols:
        # Check if binary
        uniques = np.unique(dataframe[col])

        if len(uniques) == 2:
            # Binary, force into 0 and 1
            mins = dataframe[col] == np.min(uniques)
            maxs = dataframe[col] == np.max(uniques)

            dataframe[col][mins] = 0
            dataframe[col][maxs] = 1
        else:
            # Can still be "binary"
            if len(uniques) == 1 and (uniques[0] == 0 or uniques[0] == 1):
                # Yes, single binary value
                continue

            # Continuous, zero mean with 1 standard deviation
            mean = dataframe[col].mean()
            std = dataframe[col].std()

            dataframe[col] -= mean
            # Can be single value
            if std > 0:
                dataframe[col] /= std


def normalize_numpy(array, cols):
    '''
    Normalize a numpy array. Binary values are
    forced to 0-1, and continuous (the rest) variables
    are forced to zero mean and standard deviation = 1

    Parameters:
    - array, the array to normalize column-wise
    - cols, (iterable) the column indices in the array to normalize.
    '''
    for col in cols:
        # Check if binary
        uniques = np.unique(array[col])
        if len(uniques) == 2:
            # Binary, force into 0 and 1
            mins = array[col] == np.min(uniques)
            maxs = array[col] == np.max(uniques)

            array[mins, col] = 0
            array[maxs, col] = 1
        else:
            # Can still be "binary"
            if len(uniques) == 1 and (uniques[0] == 0 or uniques[0] == 1):
                # Yes, single binary value
                continue

            # Continuous, zero mean with 1 standard deviation
            mean = array[col].mean()
            std = array[col].std()

            array[col] -= mean
            # Can be single value
            if std > 0:
                array[col] /= std


def sample_wr(population, k):
    '''Selects k random elements (with replacement) from a population.
    Returns an array of indices.
    '''
    return np.random.randint(0, len(population), k)


def bagging(data, count=None):
    '''Samples len elements (with replacement) from data and returns a view of those elements.'''
    if count is None:
        count = len(data)
    return data[np.random.randint(0, len(data), count)]


def bagging_stratified(data, column, count=None):
    '''Samples with replacement from the data set but guarantees that
       the ratio of values in column remains the same.

       Column is expected to be a binary column (any two values)
    '''
    vals = np.unique(data[:, column])
    if len(vals) != 2:
        raise ValueError("Column {} is not a binary column. Number of values are: {}".format(column, len(vals)))

    group1 = data[data[:, column] == vals[0]]
    group2 = data[data[:, column] == vals[1]]

    if count is None:
        count = len(data)

    count1 = int(round(len(group1)*count/len(data)))
    count2 = int(round(len(group2)*count/len(data)))

    retval = np.append(bagging(group1, count1), bagging(group2, count2), axis=0)
    np.random.shuffle(retval)
    return retval


def divide_stratified(data, column, frac):
    '''Divides the data set in two pieces, one being frac*len(data).
       Stratifies for the designated column to guarantee that the ratio
       remains the same. Column must be binary but can have any values.

       Returns (example frac=1/3) a tuple which has two lists of indices:
       (subdata of size 2/3, subdata of size 1/3)
    '''
    if (frac <= 0 or frac >= 1):
        raise ValueError("Frac must be a fraction between 0 and 1, not: {}".format(frac))

    vals = np.unique(data[:, column])
    if len(vals) != 2:
        raise ValueError("Column {} is not a binary column. Number of values are: {}".format(column, len(vals)))

    idx = np.arange(0, len(data))

    np.random.shuffle(idx)

    group1 = idx[data[:, column] == vals[0]]
    group2 = idx[data[:, column] == vals[1]]

    group1_num = int(round(frac*len(group1)))
    group2_num = int(round(frac*len(group2)))

    group1_test = group1[:group1_num]
    group1_trn = group1[group1_num:]
    group2_test = group2[:group2_num]
    group2_trn = group2[group2_num:]

    trn = np.append(group1_trn, group2_trn)
    test = np.append(group1_test, group2_test)

    np.random.shuffle(trn)
    np.random.shuffle(test)

    return (trn, test)
