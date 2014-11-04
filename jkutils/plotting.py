import os
from functools import wraps
from scipy.stats import pearsonr
from matplotlib.patches import Ellipse
import matplotlib as mpl
import numpy as np
# Import all into namespace
from matplotlib.pyplot import *
# But keep a handle to original as well
import matplotlib.pyplot as plt

plt_colors = ['#E41A1C', '#377EB8', '#4DAF4A',
              '#984EA3', '#FF7F00', '#FFFF33',
              '#A65628', '#F781BF', '#999999']

_set2 = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3',
         '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3']
_almost_black = '#262626'
_light_grey = '#fafafa'

def removespines(ax, spines_to_remove=None):
    '''Remove by default ['top', 'right']'''
    if spines_to_remove is None:
        spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

def removeticks(ax):
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

def setaxiscolors(ax):
    # Change the labels to the off-black
    ax.xaxis.label.set_color(_almost_black)
    ax.yaxis.label.set_color(_almost_black)

def setspinecolors(ax):
    for spine in ['top', 'bottom', 'right', 'left']:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(_almost_black)

def cleanaxis(ax=None):
    if ax is None:
        ax = plt.gca()
    removeticks(ax)
    removespines(ax)
    setaxiscolors(ax)
    setspinecolors(ax)

@wraps(plt.hist)
def hist(*args, **kwargs):
    restore = tweakstyle(below=False, bg='white', gridcolor='white', grid=False)
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = 'white'
    plt.hist(*args, **kwargs)
    ax = plt.gca()
    removespines(ax)
    removeticks(ax)
    setaxiscolors(ax)
    setspinecolors(ax)
    ax.grid(axis='y', linestyle='-', linewidth=0.5)
    restore()

@wraps(plt.semilogx)
def semilogx(*args, **kwargs):
    restore = tweakstyle(below=True, bg='#fafafa', grid=True,
                         gridcolor='lightgrey')
    plt.semilogx(*args, **kwargs)
    ax = plt.gca()
    removespines(ax)
    removeticks(ax)
    setaxiscolors(ax)
    restore()


@wraps(plt.semilogy)
def semilogy(*args, **kwargs):
    restore = tweakstyle(below=True, bg='#fafafa', grid=True,
                         gridcolor='lightgrey')
    plt.semilogy(*args, **kwargs)
    ax = plt.gca()
    removespines(ax)
    removeticks(ax)
    setaxiscolors(ax)
    restore()


@wraps(plt.loglog)
def loglog(*args, **kwargs):
    restore = tweakstyle(below=True, bg='#fafafa', grid=True,
                         gridcolor='lightgrey')
    plt.loglog(*args, **kwargs)
    ax = plt.gca()
    removespines(ax)
    removeticks(ax)
    setaxiscolors(ax)
    restore()


@wraps(plt.plot)
def plot(*args, **kwargs):
    restore = tweakstyle(below=True, bg='#fafafa', grid=True,
                         gridcolor='lightgrey')
    plt.plot(*args, **kwargs)
    ax = plt.gca()
    removespines(ax)
    removeticks(ax)
    setaxiscolors(ax)
    restore()

@wraps(plt.plot)
def bar(*args, **kwargs):
    restore = tweakstyle(bg='white')
    plt.bar(*args, **kwargs)
    ax = plt.gca()
    ax.grid(axis='y', color='white', linestyle='-', linewidth=0.5)
    removespines(ax)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('none')
    setaxiscolors(ax)
    setspinecolors(ax)
    restore()

@wraps(plt.scatter)
def scatter(*args, **kwargs):
    kwargs['alpha'] = kwargs.get('alpha', 0.5)
    kwargs['facecolor'] = kwargs.get('facecolor', _set2[0])
    kwargs['edgecolor'] = kwargs.get('edgecolor', 'black')
    kwargs['linewidths'] = kwargs.get('linewidths', 0.3)

    restore = tweakstyle(below=True, bg='#fafafa', grid=True,
                         gridcolor='lightgrey')

    plt.scatter(*args, **kwargs)
    ax = plt.gca()
    removespines(ax, ["top", "right", "left", "bottom"])
    removeticks(ax)
    setaxiscolors(ax)
    restore()


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

def plotcorr(x, headers=None, figure=None, axis=None, legend=False,
             bbox_anchor=None, non_zero=False):
    '''
    Plot a correlation matrix, much like the 'plotcorr' command in R.

    X must be a two dimensional array indexed by [variable, sample]
    so you typically have arrays of shape [15, 500] or so.

    If no figure or axis is given, one is created and scaled by the number of variables

    If specified, non_zero will only compare cases where both variables
    are non-zero.

    Returns the figure
    '''
    colors = ["#A50F15","#DE2D26","#FB6A4A","#FCAE91","#FEE5D9","white",
              "#EFF3FF","#BDD7E7","#6BAED6","#3182BD","#08519C"]

    if bbox_anchor is None:
        bbox_anchor = (1, 0, 1, 1)

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
            # all of them
            valid_rows = x[i] == x[i]
            if non_zero:
                # non-zero indices
                inz = x[i] != 0
                jnz = x[j] != 0
                valid_rows = inz * jnz
            # Calculate correlation (p is actually rho)
            p = pearsonr(x[i][valid_rows], x[j][valid_rows])[0]
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
            #ax.set_yticklabels(wraparray(headers, '\huge{', '}'))
            ax.set_yticklabels(wraparray(headers, r'\normalsize{', '}'))
            ax.set_xticklabels(wraparray(headers, r'\normalsize{', '}'))
        else:
            #xx-large
            ax.set_yticklabels(headers, fontsize='medium')
            ax.set_xticklabels(headers, fontsize='medium')

    # Move bottom to top
    for tick in ax.xaxis.iter_ticks():
        tick[0].label2On = True
        tick[0].label1On = False
        tick[0].label2.set_rotation('vertical')
        tick[0].label2.set_size('medium')

    if not legend:
        return figure

    # Explanatory legend
    labels = []
    artists = []
    centers = np.linspace(-1, 1, len(colors))
    diff = (centers[1] - centers[0]) / 2
    for i, c in enumerate(colors):
        #if i != 0 and i != len(colors)/2 and i % 2 != 0:
        #    continue
        l, r = centers[i] - diff, centers[i]  + diff
        if i == 0:
            # No left edge
            l = centers[i]
        elif i == len(colors) - 1:
            # No right edge
            r = centers[i]
        labels.append(r"${:.1f} < p < {:.1f}$".format(l, r))
        if mpl.rcParams['text.usetex']:
            labels[-1] = r"\small{" + r"${:.1f} < \rho < {:.1f}$".format(l, r) + "}"

        # Angle
        if centers[i] <= 0:
            angle = -45
        else:
            angle = 45
        e = Ellipse(xy=(0, 0), width=1, height=(1 - abs(centers[i])),
                    angle=angle)
        e.set_facecolor(c)
        artists.append(e)

    from matplotlib.legend_handler import HandlerPatch

    def make_ellipse(legend, orig_handle,
                     xdescent, ydescent,
                     width, height, fontsize):
        #size = height+ydescent
        size = width+xdescent
        p = Ellipse(xy=(0.5*width-0.5*xdescent, 0.5*height-0.5*ydescent),
                    width = size*orig_handle.width,
                    height=size*orig_handle.height,
                    angle=orig_handle.angle)
        return p

    title = 'p = Pearson correlation'
    fontsize = 'medium'
    if mpl.rcParams['text.usetex']:
        title = r'\small{$\rho$ = Pearson correlation}'
        fontsize = None

    leg = ax.legend(artists, labels,
                    handler_map={Ellipse: HandlerPatch(patch_func=make_ellipse)},
                    labelspacing=1.3, fontsize=fontsize, loc='upper left',
                    title=title, bbox_to_anchor=bbox_anchor)
    leg.legendPatch.set_facecolor('white')

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

def boxplot(x, ax=None, **keywords):
    """Creates a ggplot2 style boxplot, is eqivalent to calling ax.boxplot with the following additions:

    Keyword arguments:
    colors -- array-like collection of colours for box fills
    names -- array-like collection of box names which are passed on as tick labels

    """
    return rbox(x, ax, **keywords)

def rbox(x, ax=None, **keywords):
    """Creates a ggplot2 style boxplot, is eqivalent to calling ax.boxplot with the following additions:

    Keyword arguments:
    colors -- array-like collection of colours for box fills
    names -- array-like collection of box names which are passed on as tick labels

    """
    restore = tweakstyle(below=True, bg='#fafafa', grid=True,
                         gridcolor='lightgrey')

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

    # For bug in Matplotlib 1.4, not respecting flierprops
    keywords['sym'] = 'ko'

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
                x = [x[:,i] for i in range(nc)]
        else:
            raise ValueError("input x can have no more than 2 dimensions")
    if not hasattr(x[0], '__len__'):
        x = [x]

    numBoxes = len(x)
    boxpolygons = []
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        # TODO
        # Does not support notch yet
        for j in range(5):
          boxX.append(box.get_xdata()[j])
          boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX,boxY))

        if hasColors:
            boxPolygon = plt.Polygon(boxCoords, facecolor = colors[i % len(colors)])
        else:
            boxPolygon = plt.Polygon(boxCoords, facecolor = '0.95')

        ax.add_patch(boxPolygon)
        boxpolygons.append(boxPolygon)

    removespines(ax)
    removeticks(ax)
    setaxiscolors(ax)
    restore()

    bp['boxpolygons'] = boxpolygons
    return bp

def get_savefig(savedir, prefix=None, filename=None, extensions=None):
    '''
    Returns a function which saves the current matplotlib figure
    when called. Will set suitable values for bbox_inches.
    Files are saved with eps and png extensions in the
    designated directory and prefixed with the specified
    prefix as "prefix_filename.extension"

    DPI defaults to 300 to get high resolution png files.

    Keyword arguments:
    savedir - Folder in which to save figures

    prefix - Optional prefix for files.

    filename - Default filename to use if none is given

    extensions - An iterable of file-extensions. If None,
                 defaults to [pdf, png, eps]
    '''
    # First make sure savedir exists
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    if extensions is None:
        # Save pdf, png first as eps crashes on big images
        extensions = ['pdf', 'png', 'eps']

    # Define function which saves figures there
    @wraps(plt.savefig)
    def savefig(*args, **kwargs):
        '''
        Use as plt.savefig. File extension will be ignored, and saved
        as eps and png.
        Should only be given a raw file name, not an entire path.

        Use as savefig(filename) or savefig() if a default filename
        has been defined.

        Accepts any arguments that plt.savefig accepts.
        '''
        # Make sure we use bbox_inches
        if 'bbox_inches' not in kwargs:
            kwargs['bbox_inches'] = 'tight'

        # Make the images high resolution if not otherwise specified
        if 'dpi' not in kwargs:
            kwargs['dpi'] = 300

        # Default filename
        fname = filename
        if args is None or len(args) == 0:
            args = [] # Just make sure it's a list
        else:
            args = list(args)
            fname, ext = os.path.splitext(args.pop(0))
            #prefixing with path and prefix
            fileprefix = prefix
            if prefix is None:
                fileprefix = ''
            elif not prefix.endswith("_"):
                fileprefix += "_"
            fname = fileprefix + fname
            fname = os.path.join(savedir, fname)

        if fname is None:
            raise ValueError("A filename must be specified!")

        for ext in extensions:
            plt.savefig(*([fname + '.' + ext] + args), **kwargs)

    return savefig


def setstyle(**kwargs):
    '''Sets a more reasonable style for matplotlib.
    Any parameter for rcParams can be sent as keyword arguments
    to this function. This can be useful in iPython Notebook
    where you want to set a specific savefig.dpi to get pictures
    of a certain size for example. To handle dots in names,
    you can do it like:

    setstyle(**{'savefig.dpi':50})

    Please remember that not all functions take these values
    into account, like boxplot. In that case, you can
    access the colors for lines as:

    import matplotlib as mpl
    colors = mpl.rcParams['axes.color_cycle']

    And set some good colors by doing:
    setstyle(**{'axes.color_cycle':jkutils.plt_colors})

    For boxplot, see the rbox function for a colorized boxplot.

    Some very useful keyword arguments are:

    bg == axes.facecolor
    lw == axes.linewidth
    below == axes.axisbelow
    '''
    # Tex
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True
    # Axes
    mpl.rcParams['axes.facecolor'] = 'fafafa'
    mpl.rcParams['axes.edgecolor'] = 'white'
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['axes.labelsize'] = 'large'
    mpl.rcParams['axes.labelcolor'] = '555555'
    mpl.rcParams['axes.axisbelow'] = True
    # Set some more sensible color defaults for plots
    mpl.rcParams['axes.color_cycle'] = '#A6CEE3, #1F78B4, #B2DF8A, \
#33A02C, #FB9A99, #E31A1C, #FDBF6F, #FF7F00, #CAB2D6, \
#6A3D9A, #FFFF99, #B15928'.split(', ')
    # Ticks
    mpl.rcParams['xtick.color'] = '555555'
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.color'] = '555555'
    mpl.rcParams['ytick.direction'] = 'out'
    # Grid
    mpl.rcParams['grid.color'] = 'lightgrey'
    mpl.rcParams['grid.linestyle'] = '-'
    mpl.rcParams['axes.grid'] = True
    # Legend
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['legend.numpoints'] = 1
    # Figure
    mpl.rcParams['figure.figsize'] = (5, 4)
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['figure.edgecolor'] = '0.50'
    mpl.rcParams['figure.subplot.bottom'] = 0.15
    mpl.rcParams['figure.subplot.top'] = 0.85
    mpl.rcParams['figure.subplot.hspace'] = 0.5
    # Latex font everywhere
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern'
    # Savefig
    mpl.rcParams['savefig.dpi'] = 100

    # Set user specified stuff
    tweakstyle(**kwargs)

def tweakstyle(bg = None, lw = None, below = None, gridcolor=None, grid=None,
               **kwargs):
    '''Tweak a specific part of the plotting style, but don't overwrite
     other changes.

    Some very useful keyword arguments are:

    bg == axes.facecolor
    lw == axes.linewidth
    below == axes.axisbelow
    grid = axes.grid
    gridcolor == grid.color
    '''

    oldvals = mpl.rcParams.copy()
    def restore():
        '''Restore to values before tweak.'''
        for k, v in oldvals.items():
            try:
                mpl.rcParams[k] = v
            except ValueError:
                # Ignore possible bad values
                pass

    if bg is not None:
        mpl.rcParams['axes.facecolor'] = bg
    if lw is not None:
        mpl.rcParams['axes.linewidth'] = lw
    if below is not None:
        mpl.rcParams['axes.axisbelow'] = below
    if gridcolor is not None:
        mpl.rcParams['grid.color'] = gridcolor
    if grid is not None:
        mpl.rcParams['axes.grid'] = grid

    # The rest
    for k, v in kwargs.items():
        mpl.rcParams[k] = v

    # Return restore function
    return restore
