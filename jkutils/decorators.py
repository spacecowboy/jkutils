'''Some decorators that are handy to use at times'''

import os, pickle, errno
from functools import wraps
from .filehandling import mkdir_p


def lazyproperty(fn):
    """Wraps the normal property decorator and makes
    a property lazy. It property will only be accessed
    or calculated once for this object's lifetime"""
    attr_name = '_lazy_' + fn.__name__
    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


def lazyfile(path='./.lazyfile/', idgen=None):
    """Make a method lazy by saving/loading its result
    from a file.

    Args:
    path
    Folder where files are placed

    idgen
    A callable object which should return an id
    of some sort. Will be used in a string so don't return
    stupid characters. Called with the args and kwargs given
    to the function itself. This means you can generate
    a unique id based on the args if you want. Two
    examples:

    def idgen1(*args, **kwargs):
        return 0 # Increment this manually before real runs

    def idgen2(myarg):
        return hash(myarg) # Change with the arguments

    Caution:
    Because this decorator takes optional
    arguments, you must always write PARENTHESIS:

    @lazyfile()
    def afunc():
        return "bob"

    @lazyfile(path='/home/bob/temp')
    def bfunc(bah):
        return bah

    @lazyfile(idgen=idgen)
    def cfunc(args):
        return calc(args)

    An alternative approach is first generate your decorator
    then use it the normal way without parenthesis:

    lazynate = lazyfile(idgen=idgen)

    @lazynate
    def dfunc(args):
        return calc(args)

    The result is pickled and stored at the specified path
    in file ".lazyfile_FUNCTIONNAME". The (entire) path will
    be created if it does not already exist.

    The following will NOT work because it will return the
    wrong wrapped decorator:

    @lazyfile
    def wrong():
        return "Will never run"
    """
    def _wrapped_lazyfunc(fn):
        @wraps(fn)
        def _lazyfunc(*args, **kwargs):
            result = None
            fname = '._lazyfile_{}'.format(fn.__name__)
            if idgen is not None:
                fname += '_{}'.format(idgen(*args, **kwargs))
            fpath = os.path.join(path, fname)
            # First, we try to load from the file if we can
            try:
                with open(fpath, 'rb') as savefile:
                    result = pickle.load(savefile)
            except (IOError):
                # No file exists (yet)
                pass

            # If nothing was loaded, we calculate it and save if we can
            if result is None:
                result = fn(*args, **kwargs)
                # Don't accept errors here
                mkdir_p(path)
                with open(fpath, 'wb') as savefile:
                    pickle.dump(result, savefile, pickle.HIGHEST_PROTOCOL)

            return result
        return _lazyfunc
    return _wrapped_lazyfunc
