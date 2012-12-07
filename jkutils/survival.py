"""
Some utility functions for survival contexts, like calculating the C-index
"""

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

    print(("Sum: {}".format(sum)))
    print(("Total: {}".format(total)))
    return float(sum) / float(total)
