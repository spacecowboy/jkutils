'''
This module defines several methods that can be used to test
and compare different properties in machine learning. They are
intended to be used in cross validation and box plotting
scenarios.
'''

from __future__ import division, print_function
import numpy as np
import os
import ann
#from ann import get_C_index
from jkutils import plotting as plt

def test_parameter_values(net_constructor, data, inputcols, targetcols,
                           name, values, ntimes=10):
    '''
    Given a parameter and a list of values, will test ntimes for each
    parameter value. This trains on the entire data set each time so
    is suitable for parameters which should be treated as unrelated
    to overtraining.
    Returns the result as a dict, defined as dict[value] = resultlist

    Keyword arguments:
    net_constructor - A function that should return a new neural network with
    all properties set to suitable values. Will be called as:
    net_constructor(name, value)

    data - The data to do crossvalidation on. Should be a two-dimensional
    numpy array (or compatible).

    inputcols - A tuple/list of the column numbers which represent the input
    data.

    targetcols - A tuple/list expected to have two members. First being the
    column number of the survival times. The second being the column number
    of the event column.

    name - Name of the property to test different values for. Is up to
    net_constructor to set this property on the network

    values - A list of values to test 'name' with.

    ntimes - The number of times to train the network.
    '''
    # Store results in a hash
    results = {}

    # Try these values for generations parameter
    for value in values:
        print("Training with {} = {}".format(name, value))
        net = net_constructor(name, value)
        # Store results in a hash where the values are lists
        results[value] = []

        #Train n times for each value
        for x in range(ntimes):
            net.learn(data[:, inputcols], data[:, targetcols])
            predictions = np.array([net.output(x) for x in data[:, inputcols]])
            results[value].append(np.mean(ann.get_error(net.error_function,
                                                data[:, targetcols],
                                                predictions)))
            #results[value].append(get_C_index(data[:, targetcols], predictions))

    return results


def crossvalidate_mse(net_constructor, data, inputcols, targetcols, ntimes=5,
                      kfold=3, stratifycol=None):
    '''
    Does crossvalidation testing on a network the designated
    number of times. Expects a single target column. Stratifies for classes
    if specified.

    Keyword arguments:
    net_constructor - A function that should return a new neural network with
    all properties set to suitable values.

    data - The data to do crossvalidation on. Should be a two-dimensional
    numpy array (or compatible).

    inputcols - A tuple/list of the column numbers which represent the input
    data.

    targetcols - Column number which is the target column.

    ntimes - The number of times to divide the data.

    kfold - The number of folds to divide the data in. Total number of results
    will equal ntimes * kfold. Where each row has featured in a test set ntimes.

    startifycol - Optional. The column stratify for. Default None.

    Returns a tuple: (trnresultlist, valresultlist)
    where each list is ntimes * kfold long.
    '''
    trnresults = []
    valresults = []

    indices = np.arange(len(data))

    classindices = {}
    if stratifycol is not None:
        classes = np.unique(data[:, stratifycol])
        for c in classes:
            classindices[c] = indices[data[:, stratifycol] == c]
    else:
        classes = ['Dummy']
        classindices[classes[0]] = indices

    for n in range(ntimes):
        # Re-shuffle the data every time
        for c in classes:
            np.random.shuffle(classindices[c])

        for k in range(kfold):
            valindices = []
            trnindices = []

            # Join the data pieces
            for p in range(kfold):
                # validation piece
                if k == p:
                    for idx in classindices.values():
                        # Calc piece length
                        plength = int(round(len(idx) / kfold))
                        valindices.extend(idx[p*plength:(p+1)*plength])
                else:
                    for idx in classindices.values():
                        # Calc piece length
                        plength = int(round(len(idx) / kfold))
                        trnindices.extend(idx[p*plength:(p+1)*plength])

            # Ready to train
            net = net_constructor()
            net.learn(data[trnindices][:, inputcols],
                      data[trnindices][:, (targetcols,)])

            # Training result
            predictions = np.array([net.output(x) for x in data[trnindices][:, inputcols]]).ravel()
            mse = np.sum((data[trnindices][:, targetcols] - predictions)**2) / len(predictions) / 2.0
            trnresults.append(mse)

            # Validation result
            predictions = np.array([net.output(x) for x in data[valindices][:, inputcols]]).ravel()
            mse = np.sum((data[valindices][:, targetcols] - predictions)**2) / len(predictions) / 2.0

            valresults.append(mse)

    return (trnresults, valresults)

def crossvalidate(net_constructor, data, inputcols, targetcols, ntimes=5,
                  kfold=3, evalfunc = None, evalresults = None):
    '''
    Does crossvalidation testing on a network the designated
    number of times. Random divisions are stratified for events.

    Keyword arguments:
    net_constructor - A function that should return a new neural network with
    all properties set to suitable values. The model must a "learn" method,
    and one of the following: output or output_all, where they are called as:
    output(x), for x in data[:, inputcols]
    output_all(data[:, inputcols)

    data - The data to do crossvalidation on. Should be a two-dimensional
    numpy array (or compatible).

    inputcols - A tuple/list of the column numbers which represent the input
    data.

    targetcols - A tuple/list expected to have two members. First being the
    column number of the survival times. The second being the column number
    of the event column. The data sets are stratified for the event column.

    ntimes - The number of times to divide the data.

    kfold - The number of folds to divide the data in. Total number of results
    will equal ntimes * kfold. Where each row has featured in a test set ntimes.

    evalfunc - Function to apply at end of training. Called with the following
    signature:
    evalfunc(net, data, inputcols, targetcols, trnindices, valindices, evalresults)
    Results are expected to be placed in evalresults. Only called if both
    evalfunc and evalresults is not None.

    Returns a tuple: (trnresultlist, valresultlist)
    where each list is ntimes * kfold long.
    '''
    trnresults = []
    valresults = []

    # This might be a decimal number, remember to round it off
    indices = np.arange(len(data))

    classes = np.unique(data[:, targetcols[1]])
    classindices = {}
    for c in classes:
        classindices[c] = indices[data[:, targetcols[1]] == c]

    for n in range(ntimes):
        # Re-shuffle the data every time
        for c in classes:
            np.random.shuffle(classindices[c])

        for k in range(kfold):
            valindices = []
            trnindices = []

            # Join the data pieces
            for p in range(kfold):
                # validation piece
                if k == p:
                    for idx in classindices.values():
                        # Calc piece length
                        plength = int(round(len(idx) / kfold))
                        valindices.extend(idx[p*plength:(p+1)*plength])
                else:
                    for idx in classindices.values():
                        # Calc piece length
                        plength = int(round(len(idx) / kfold))
                        trnindices.extend(idx[p*plength:(p+1)*plength])

            # Ready to train
            net = net_constructor()
            net.learn(data[trnindices][:, inputcols],
                      data[trnindices][:, targetcols])

            # Training result
            try:
                predictions = np.array(net.output_all(data[trnindices][:, inputcols]))
            except:
                predictions = np.array([net.output(x) for x in data[trnindices][:, inputcols]])

            err = np.mean(ann.get_error(net.error_function,
                                    data[trnindices][:, targetcols],
                                    predictions))
            #c_index = get_C_index(data[trnindices][:, targetcols], predictions)
            trnresults.append(err)

            # Validation result
            try:
                predictions = np.array(net.output_all(data[valindices][:, inputcols]))
            except:
                predictions = np.array([net.output(x) for x in data[valindices][:, inputcols]])

            err = np.mean(ann.get_error(net.error_function,
                                    data[valindices][:, targetcols],
                                    predictions))

            #c_index = get_C_index(data[valindices][:, targetcols], predictions)
            valresults.append(err)

            if evalfunc is not None and evalresults is not None:
                evalfunc(net, data, inputcols, targetcols,
                         trnindices, valindices, evalresults)

    return (trnresults, valresults)


def plot_comparison(paramchoices, results, savefig=None, figname=None,
                    xlabel=None, figure=None):
    '''
    Plots the results of parameter runs as a boxplot.
    You might need to call plt.show() afterwards.

    Keyword arguments:
    paramchoices - A list of values that have been tested.

    results - A list of equal length as paramchoices. Each value is
    in turn a list of results corresponding to that parameter value.

    savefig - An optional function which saves the figure to a file.
    See "get_savefig".

    figname - An optional filename to save the figure as.

    figure - Optional figure to plot in

    xlabel - Optional label for x-axis
    '''
    if figure is None:
        plt.figure()
    plt.boxplot(results)

    #plt.ylim((0.5, 1.0))
    ax = plt.gca()
    ax.set_ylabel('Error')
    if xlabel:
        ax.set_xlabel(xlabel)
    # Nice x-axis values
    ax.set_xticklabels(paramchoices)
    # Rotate labels
    #plt.gcf().autofmt_xdate()
    # Save eps
    if savefig is not None and figname is not None:
        savefig(figname)
    elif savefig is not None:
        savefig()


def crossvalidate_parameter(net_constructor, data, inputcols, targetcols,
                    name, values, savefig=None,
                    ntimes=5, kfold=3):
    '''
    Runs crossvalidation tests on the parameter with the different
    values. Then plots the results as a boxplot and possibly saves
    the plots as files. Crossvalidations are stratified over
    censored events.

    Keyword arguments:
    net_constructor - A function that should return a new neural network with
    all properties set to suitable values. Will be called as:
    net_constructor(name, value)

    data - The data to do crossvalidation on. Should be a two-dimensional
    numpy array (or compatible).

    inputcols - A tuple/list of the column numbers which represent the input
    data.

    targetcols - A tuple/list expected to have two members. First being the
    column number of the survival times. The second being the column number
    of the event column.

    name - The name of the property to cross validate. Up to net_constructor
    to set the value of this parameter.

    values - A list of values to test the property with.

    savefig - An optional function to save figures to files. See
    "get_savefig()".

    ntimes - The number of times to divide the data.

    kfold - The number of folds to divide the data in. Total number of results
    will equal ntimes * kfold. Where each row has featured in a test set ntimes.
    '''
    labels = ['{}'.format(v) for v in values]
    results = []

    for val in values:
        print("Cross validating {} = {}".format(name, val))
        def inner_cons():
            net = net_constructor(name, val)
            return net

        results.append(crossvalidate(inner_cons, data, inputcols, targetcols,
                                     ntimes, kfold))

    valresults = [result[1] for result in results]
    trnresults = [result[0] for result in results]

    plot_comparison(labels, trnresults, savefig, name + '_trn',
                    xlabel = name)
    plot_comparison(labels, valresults, savefig, name + '_val',
                    xlabel = name)


def compare_parameter(net_constructor, data, inputcols, targetcols,
                      name, values, savefig=None, ntimes=10, plot=True,
                      figure=None):
    '''
    Keyword arguments:
    net_constructor - A function that should return a new neural network with
    all properties set to suitable values. Will be called as:
    net_constructor(name, value)

    data - The data to do crossvalidation on. Should be a two-dimensional
    numpy array (or compatible).

    inputcols - A tuple/list of the column numbers which represent the input
    data.

    targetcols - A tuple/list expected to have two members. First being the
    column number of the survival times. The second being the column number
    of the event column.

    name - Name of the property to test different values for. Up to
    net_constructor to set this property to its proper value.

    values - A list of values to test 'name' with.

    savefig - An optional function to save figures to files. See
    "get_savefig()".

    ntimes - The number of times to train the network.

    plot[true] - Set to false if no plotting desired
    figure - Optional existing figure to plot to

    Returns a tuple (variable, result)
    '''
    testresults = test_parameter_values(net_constructor, data, inputcols,
                                    targetcols, name, values, ntimes)

    labels = ['{}'.format(v) for v in sorted(testresults.keys())]

    sortedresults = [testresults[k] for k in sorted(testresults.keys())]

    if plot:
        plot_comparison(labels, sortedresults, savefig, name, xlabel = name,
                        figure=figure)

    return (sorted(testresults.keys()), sortedresults)
