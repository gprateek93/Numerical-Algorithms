import matplotlib.pyplot as plt
import numpy as np

def fcontour(f, x1range, x2range, **kwargs):
    """Draw the curve f(x) = 0 over the specified range.

    Arguments:
    f -- the function defining the curve
    x1range -- a 2-element list [x1min, x1max]
    x2range -- a 2-element list [x2min, x2max]

    Keyword arguments (e.g. for formatting) are passed on to
    matplotlib.pyplot.contour().
    """
    x1s = np.linspace(x1range[0], x1range[1])
    x2s = np.linspace(x2range[0], x2range[1])
    fs = [[f(np.array([x1,x2])) for x1 in x1s] for x2 in x2s]
    plt.contour(x1s, x2s, fs, [0], **kwargs)
    plt.axis('scaled')
