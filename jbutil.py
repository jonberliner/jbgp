from itertools import chain
from datetime import datetime
from matplotlib.pyplot import get_cmap
from numpy import asarray, prod, zeros, repeat
from numpy.random import binomial
from pandas import DataFrame, read_pickle
from glob import glob
import pdb


def stitch_pickled(critstring):
    """load all files that match criteria of critstring, unpickle, and convert
       into pandas DataFrame"""
    assert (critstring[-4:] == '.pkl' or critsting[-7:] =='.pickle'),\
            'critsting must end in pickle extension'
    subpkls = glob(critstring)
    subfits = [read_pickle(subpkl) for subpkl in subpkls]
    stitched_df = DataFrame(subfits)
    return stitched_df


def merge_first(to, frm, field, on):
    assert isinstance(to, DataFrame)
    assert isinstance(frm, DataFrame)
    assert isinstance(field, str)
    assert isinstance(on, str)

    to.set_index(on, inplace=True)
    ids = to.index.unique()
    for id in ids:
        to.loc[id, field] = frm[frm[on]==id][field].iat[0]
    to.reset_index(inplace=True)
    return to


def datetimestamp(delim='_'):
    fmt = delim.join(['%Y', '%m', '%d', '%H', '%M'])
    return datetime.now().strftime(fmt)


def factors(n):
    result = []
    # test 2 and all of the odd numbers
    # xrange instead of range avoids constructing the list
    for i in chain([2],xrange(3,n+1,2)):
        s = 0
        while n%i == 0: #a good place for mod
            n /= i
            s += 1
        result.extend([i]*s) #avoid another for loop
        if n==1:
            return result


def cmap_discrete(N, cmap):
    cm = get_cmap(cmap)
    return [cm(1.*i/N) for i in xrange(N)]


def ndm(*args):
    """generates a sparse mesh from numpy vecs"""
    return [x[(None,)*i+(slice(None),)+(None,)*(len(args)-i-1)] for i, x in enumerate(args)]



def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays in the list arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    from:
    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """
    arrays = [asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = prod([x.size for x in arrays])
    if out is None:
        out = zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def rank(array, descending=True):
    order = array.argsort()
    if descending: order = order[::-1]
    return order.argsort()


def bestrun(array, desired=1):
    """return the longest run of elt 'desired' in array 'array'"""
    ibestrun = -1
    bestrun = 0
    irun = 0
    run = 0
    for ielt, elt in enumerate(array):
        # count this run
        if elt == desired:
            run += 1
        else:
            run = 0
            irun = ielt
        # see if best run
        if run > bestrun:
            bestrun = run
            ibestrun = irun
    return (bestrun, ibestrun)


