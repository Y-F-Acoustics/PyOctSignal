import numpy as np
import scipy.signal as signal

def fir2(n: int, f, m, grid_n: int=None, ramp_n: int=None, window='hamming'):
    """
    Produce an order n FIR filter with arbitrary frequency response m over frequency bands f, returning the n+1 filter coefficients in b. 
    
    The vector f specifies the frequency band edges of the filter response and m specifies the magnitude response at each frequency.
    
    The vector f must be nondecreasing over the range [0,1], and the first and last elements must be 0 and 1, respectively. 
    A discontinuous jump in the frequency response can be specified by duplicating a band edge in f with different values in m.
    
    The resolution over which the frequency response is evaluated can be controlled with the grid_n argument.
    The default is 512 or the next larger power of 2 greater than the filter length.
    
    The band transition width for discontinuities can be controlled with the ramp_n argument.
    The default is grid_n/25. Larger values will result in wider band transitions but better stopband rejection.
    
    An optional shaping window can be given as a vector with length n+1. If not specified, a Hamming window of length n+1 is used.
    
    To apply the filter, use the return vector b with the filter function, for example y = filter (b, 1, x).
    """

    # Verify frequency and magnitude vectors are reasonable
    try:
        t = len(f)
    except TypeError:
        raise ValueError("fir2: frequency must be nondecreasing starting from 0 and ending at 1")
    else:
        if (t < 2) or (f[0] != 0) or (f[t-1] != 1) or (np.any(np.diff(f) < 0)):
            raise ValueError("fir2: frequency must be nondecreasing starting from 0 and ending at 1")
        
    try:
        truth = (t != len(m))
    except TypeError:
        raise ValueError("fir2: frequency and magnitude vectors must be the same length")
    else:
        if truth == True:
            raise ValueError("fir2: frequency and magnitude vectors must be the same length")


    # Default grid size is 512, unless n+1 >= 1024
    if grid_n is None:
        if n+1 < 1024:
            grid_n = 512
        else:
            grid_n = n + 1

    # ML behavior appears to always round the grid size up to a power of 2
    grid_n = 2 ** int(np.log2(2 ** np.ceil(np.log2(grid_n)))) # grid_n = 2 ^ nextpow2(grid_n)

    # Error out if the grid size is not beig enough for the window
    if 2**grid_n < n+1:
        raise ValueError("fir2: grid size must be greater than half the filter order")

    # Find the grid spacing and ramp width
    if ramp_n is None:
        ramp_n = int(np.fix(grid_n/25))
    
    if (type(grid_n) != int) or (type(ramp_n) != int):
        raise ValueError("fir2: grid_n and ramp_n must be integers")

    if type(window) in [list, np.ndarray]:
        if len(window) != n+1:
            raise ValueError("fir2: window must be of length n+1")
    elif type(window) == str:
        window = signal.get_window(window, n+1, fftbins=False)
    
    # Apply ramps to discontinuities
    if ramp_n > 0:
        # remember originalfrequency points prior to applying ramps
        basef = f.copy()
        basem = m.copy()

        # separate identical frequencies, but keep the midpoint
        idx = np.where(np.diff(f) == 0) # idx = find(diff(m) == 0)
        if (type(f) == list):
            f = np.array(f)
        if (type(basef) == list):
            basef = np.array(basef)
        f[idx[0]] = f[idx[0]] - ramp_n/grid_n/2
        f[idx[0]+1] = f[idx[0]+1] + ramp_n/grid_n/2
        f = np.transpose(np.hstack((f, basef[idx])))

        # Make sure the grid points stay monolithic in [0, 1]
        f[f < 0] = 0
        f[f > 1] = 1
        f = np.unique(np.hstack((f, basef[idx])))

        # Preserve window shape even though f may have changed
        m = np.interp(f, basef, basem)

    # Interpolate between grid points
    grid = np.interp(np.linspace(0, 1, num=grid_n+1), f, m)

    # Transform frequency response into time response and
    # center the response about n/2, truncating the excess
    if np.remainder(n, 2) == 0:
        b = np.fft.ifft(np.hstack((grid, grid[grid_n:1:-1])))
        mid = (n+1)/2
        b = np.real(np.hstack((b[len(b)-int(np.floor(mid)):len(b)], b[0:int(np.ceil(mid))])))
    else:
        b = np.fft.ifft(np.hstack((grid, np.zeros(grid_n*2), grid[grid_n:1:-1])))
        b = 2 * np.real(np.hstack((b[len(b)-n:-1:2], b[1:n+1:2])))
        
    # Multiplication in the time domain is convolution in frequency,
    # so multiply y our window now to smooth the frequency response.
    # Also, for matlab compatibility, we return return values in 1 row
    b = b * window

    return b
