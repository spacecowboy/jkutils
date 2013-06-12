from scipy.stats import pearsonr
from matplotlib.patches import Ellipse
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

def wraparray(array, prefix=None, suffix=None):
    '''Returns an array with the string versions of the
    items of the given array wrapped by the given prefixes
    and suffixes.

    Example:

    wraparray([1,2,3], prefix='Ball ', suffix=' flies')
    > ['Ball 1 flies', 'Ball 2 flies', 'Ball 3 flies']'''
    if prefix is None:
        prefix = ''
    if suffix is None:
        suffix = ''
    xc = []
    for x in array:
        xc.append(prefix + str(x) + suffix)

    return xc

def plotcorr(x, headers=None, figure=None, axis=None):
    '''
    Plot a correlation matrix, much like the 'plotcorr' command in R.

    X must be a two dimensional array indexed by [variable, sample]
    so you typically have arrays of shape [15, 500] or so.

    If no figure or axis is given, one is created and scaled by the number of variables

    Returns the figure
    '''
    colors = ["#A50F15","#DE2D26","#FB6A4A","#FCAE91","#FEE5D9","white",
              "#EFF3FF","#BDD7E7","#6BAED6","#3182BD","#08519C"]

    if figure is None and axis is None:
        # Scale by size of array
        dim = 6 + len(x)/4
        figure = plt.figure(figsize=(dim,dim))
    if axis is None:
        axis = plt.gca()
    ax = axis
    ax.set_axis_bgcolor('white')
    ax.grid('off')
    lim = len(x)
    for i in range(len(x)):
        for j in range(len(x)):
            # Calculate correlation
            p = pearsonr(x[i], x[j])[0]
            # Pick a color
            c = p
            c += 1
            c *= (len(colors) - 1) / 2
            c = int(round(c))
            c = colors[c]

            # Angle
            if p >= 0:
                # Linear correlation
                angle = -45
            else:
                # Anti linear correlation
                angle = 45

            e = Ellipse(xy=(i, j), width=1, height=(1 - abs(p)), angle=angle)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            #e.set_alpha(rand())
            e.set_facecolor(c)

    ax.set_ylim((-0.5, len(x)-0.5))
    ax.set_xlim((-0.5, len(x)-0.5))
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(x)))
    ax.set_xticks(np.arange(len(x)))

    if (headers is not None and len(headers) == len(x)):
        if mpl.rcParams['text.usetex']:
            ax.set_yticklabels(wraparray(headers, '\huge{', '}'))
            ax.set_xticklabels(wraparray(headers, '\huge{', '}'))
        else:
            ax.set_yticklabels(headers, fontsize='xx-large')
            ax.set_xticklabels(headers, fontsize='xx-large')

    # Move bottom to top
    for tick in ax.xaxis.iter_ticks():
        tick[0].label2On = True
        tick[0].label1On = False
        tick[0].label2.set_rotation('vertical')
        tick[0].label2.set_size('xx-large')

    return figure


def rhist(data, ax=None, **keywords):
    """Creates a histogram with default style parameters to look like ggplot2
    Is equivalent to calling ax.hist and accepts the same keyword parameters.
    If style parameters are explicitly defined, they will not be overwritten.

    defaults = {
                'facecolor' : '0.3',
                'edgecolor' : '0.28',
                'linewidth' : '1',
                'bins' : 100
                }
    """

    defaults = {
                'facecolor' : '0.3',
                'edgecolor' : '0.28',
                'linewidth' : '1',
                'bins' : 100
                }

    for k, v in defaults.items():
        if k not in keywords: keywords[k] = v

    if ax is None:
        ax = plt.gca()

    return ax.hist(data, **keywords)


def rbox(x, ax=None, **keywords):
    """Creates a ggplot2 style boxplot, is eqivalent to calling ax.boxplot with the following additions:

    Keyword arguments:
    colors -- array-like collection of colours for box fills
    names -- array-like collection of box names which are passed on as tick labels

    """
    if ax is None:
        ax = plt.gca()

    #hasColors = 'colors' in keywords
    if 'colors' in keywords:
        if keywords['colors'] is None:
            hasColors = False
        else:
            colors = keywords['colors']
            keywords.pop('colors')
            hasColors = True
    else:
        colors = '#A6CEE3, #1F78B4, #B2DF8A, #33A02C, #FB9A99, #E31A1C, \
#FDBF6F, #FF7F00, #CAB2D6, #6A3D9A, #FFFF99, #B15928'.split(', ')
        hasColors = True

    if 'names' in keywords:
        ax.tickNames = plt.setp(ax, xticklabels=keywords['names'] )
        keywords.pop('names')

    bp = ax.boxplot(x, **keywords)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black', linestyle = 'solid')
    plt.setp(bp['fliers'], color='black', alpha = 0.9, marker= 'o',
             markersize = 3)
    plt.setp(bp['medians'], color='black')

    # convert x to a list of vectors, just as it's done in boxplot
    if hasattr(x, 'shape'):
        if len(x.shape) == 1:
            if hasattr(x[0], 'shape'):
                x = list(x)
            else:
                x = [x,]
        elif len(x.shape) == 2:
            nr, nc = x.shape
            if nr == 1:
                x = [x]
            elif nc == 1:
                x = [x.ravel()]
            else:
                x = [x[:,i] for i in xrange(nc)]
        else:
            raise ValueError("input x can have no more than 2 dimensions")
    if not hasattr(x[0], '__len__'):
        x = [x]

    numBoxes = len(x)
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        # TODO
        # Does not support notch yet
        for j in range(5):
          boxX.append(box.get_xdata()[j])
          boxY.append(box.get_ydata()[j])
        boxCoords = zip(boxX,boxY)

        if hasColors:
            boxPolygon = plt.Polygon(boxCoords, facecolor = colors[i % len(colors)])
        else:
            boxPolygon = plt.Polygon(boxCoords, facecolor = '0.95')

        ax.add_patch(boxPolygon)
    return bp
