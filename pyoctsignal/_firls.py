import scipy.signal as signal

def firls(N:int, frequencies, a, weight=None):
    """
    FIR filter design using least squares method.  Returns a length N+1
    linear phase filter such that the integral of the weighted mean
    squared error in the specified bands is minimized.

    The vector F specifies the frequencies of the band edges,
    normalized so that half the sample frequency is equal to 1.  Each
    band is specified by two frequencies, to the vector must have an
    even length.

    The vector A specifies the amplitude of the desired response at
    each band edge.

    The optional argument W is a weighting function that contains one
    value for each band that weights the mean squared error in that
    band.

    A must be the same length as F, and W must be half the length of F.
    N must be even.  If given an odd value, 'firls' increments it by 1.

    The least squares optimization algorithm for computing FIR filter
    coefficients is derived in detail in:

    I. Selesnick, "Linear-Phase FIR Filter Design by Least Squares,"
    http://cnx.org/content/m10577
    """

    return signal.firls(N, frequencies, a, weight=weight, fs=2)
