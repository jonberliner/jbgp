from itertools import chain
from datetime import datetime
from matplotlib.pyplot import get_cmap


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
