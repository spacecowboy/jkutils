"""
Some utility functions for survival contexts, like calculating the C-index
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def calc_line(x, y):
    '''
    y = mx + c
    We can rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]]. Now use lstsq to solve for p:
    Returns m, c
    '''
    A = np.vstack([x, np.ones(len(x))]).T
    return np.linalg.lstsq(A, y)[0]


def scatter(data_x, data_y, events = None, gridsize = 30, mincnt = 0,
            plotSlope = False):
    '''
    It is assumed that the x-axis contains the target data, and y-axis the computed outputs.
    If events is not None, then any censored data (points with a zero in events) will not be able to go above the diagonal.
    Reason for that being that a censored data is "correct" if the output for that data is greater or equal to the diagonal point.
    The diagonal is calculated from the non-censored data (all if no events specified) using least squares linear regression.
    Gridsize determines how many hexagonal bins are used (on the x-axis. Y-axis is determined automatically to match)
    mincnt is the minimum number of hits a bin needs to be plotted.
    '''
    if not len(data_x) == len(data_y) or (events is not None and not len(data_x) == len(events)):
        raise ValueError('Lengths of arrays do not match!')

    xmin = data_x.min()
    xmax = data_x.max()
    ymin = data_y.min()
    ymax = data_y.max()

    if not plotSlope:
        #Nothing has been specified, we do it if we have events
        plotSlope = True if events is not None else False

    #For plotting reasons, we need to have these sorted
    if events is None:
        sorted_x_y = [[data_x[i], data_y[i]] for i in xrange(len(data_x))]
    else:
        sorted_x_y = [[data_x[i], data_y[i], events[i]] for i in xrange(len(data_x))]
    sorted_x_y.sort(lambda x, y: cmp(x[0], y[0])) #Sort on target data
    sorted_x_y = np.array(sorted_x_y)
    #Calculate the regression line (if events is None weneed it later)
    slope, cut = calc_line(sorted_x_y[:, 0], sorted_x_y[:, 1])

    if events is not None:
        #We must calculate the diagonal from the non-censored
        non_censored_x = sorted_x_y[:, 0][sorted_x_y[:, 2] == 1]
        non_censored_y = sorted_x_y[:, 1][sorted_x_y[:, 2] == 1]

        ymin = non_censored_y.min()
        ymax = non_censored_y.max()

        slope, cut = calc_line(non_censored_x, non_censored_y)

        #And then no censored point can climb above the diagonal. Their value is the percentage of their comparisons
        #in the C-index which are successful
        for i in xrange(len(sorted_x_y)):
            target, output, event = sorted_x_y[i]
            if event == 0:
                #Compare with all previous non-censored and calculate ratio of correct comparisons
                total = num_of_correct = 0
                for prev_target, prev_output, prev_event in sorted_x_y[:i]:
                    if prev_event == 1:
                        total += 1
                        if prev_output <= output: #cmp(prev_output, output) < 1
                            num_of_correct += 1

                #Now we have the ratio
                #If no previous non-censored exist (possible for test sets for example) we move to zero.
                total = 1.0 if total < 1 else total
                ratio = num_of_correct / total

                #Move the point
                diagonal_point = cut + slope * target
                sorted_x_y[i][1] = ymin + ratio * (diagonal_point - ymin)

    axWasNone = True
    ax = plt.gca()
    #if ax is None:
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111)
    #    axWasNone = True
    pc = ax.hexbin(sorted_x_y[:, 0], sorted_x_y[:, 1], bins = 'log', cmap = cm.jet,
                   gridsize = gridsize, mincnt = mincnt)
    ax.axis([xmin, xmax, ymin, ymax])
    #line_eq = "Line: {m:.3f} * x + {c:.3f}".format(m=slope, c=cut)
    #ax.set_title("Scatter plot heatmap, taking censored into account\n" + line_eq) if events is not None else \
    #    ax.set_title("Scatter plot heatmap\n" + line_eq)
    if axWasNone:
        cb = plt.gcf().colorbar(pc, ax = ax)
        cb.set_label('log10(N)')
    if plotSlope:
        ax.plot(sorted_x_y[:, 0], slope*sorted_x_y[:, 0] + cut, 'r-') #Print slope
    #ax.scatter(sorted_x_y[:, 0], sorted_x_y[:, 1], c='g')
    #ax.set_xlabel(x_label)
    #ax.set_ylabel(y_label)
    #ax.set_title(title)


def get_C_index(T, outputs):
    """
    Better to use the much faster C++-version in ann-package.

    Calculate the C-index of outputs compared to the target data T.
    T should be 2-dimensional, first column is the survival time and second
    column is the event variable. Y is also expected to be 2-dimensional, but
    only the first column needs to be filled. This should be the predicted
    survival times or prognostic indices.

    Other details:
    if T[x,0] < T[y,0] and X[x] < X[y] or T[x,0] > T[y,0] and X[x] > X[y],
    plus 1. Finally divide by the number of comparisons made.
    Non-censored points can be compared with all other non-censored points
    and all later censored points.
    Censored points can only be compared to earlier non-censored points.
    
    If X[x] = X[y], 0.5 is added. Ties in T are not valid comparisons.
    """

    total = 0
    sum = 0
    for x in range(len(T)):
        for y in range(len(T)):
            if x == y:
                #Don't compare with itself
                continue
            if T[x, 1] == 1 and (T[y, 1] == 1):
                #Non-censored, compare with all other non-censored
                if T[x, 0] < T[y, 0]:
                    #all non-censored will be compared eventually, but only once
                    total += 1
                    if outputs[x, 0] < outputs[y, 0]:
                        sum += 1
            elif T[x, 1] == 1 and (T[y, 1] == 0) and T[x, 0] < T[y, 0]:
                #Non-censored compared with later censored
                total += 1
                if outputs[x, 0] < outputs[y, 0]:
                    sum += 1
                elif outputs[x, 0] == outputs[y, 0]:
                    sum += 0.5

    print(("Sum: {}".format(sum)))
    print(("Total: {}".format(total)))
    return float(sum) / float(total)
