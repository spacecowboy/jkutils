"""
Functions which do not sort well under another name. Will all be imported
by __init__.py to package level.
"""

import random

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
