import numpy as np
import scipy.signal as signal
from scipy import interpolate
from ._fir2 import fir2

def fir1(n:int, w, ftype:str='default', window='hamming', scaleopt:str='scale'):
    """

    """

    # Assign default window, filter type and scale
    # If single band egde, the first band defaults to a pass band to
    # create a lowpass filter. If multiple band edges, the first band
    # defaults to a stop band so that the two band case defaults to a
    # band pass filter. Ick.
    if type(w) in [list, tuple]:
        w = np.array(w)
    elif type(w) in [int, float]:
        w = np.array((w, ))
    
    # window = [] <- no code input in this file to prevent overwrite default hamming window.
    scale = 1 # scale=1
    ftypenum = (np.shape(w)[0] == 1) # ftype = (length(w)==1)

    # sort arglist, normalize any string
    if ftype == 'default':
        if np.shape(w)[0] == 1:
            ftype = 'low'
        elif np.shape(w)[0] == 2:
            ftype = 'bandpass'
        elif np.shape(w)[0] >= 3:
            ftype = 'DC-0'

    if ftype in ['low', 'stop', 'DC-1']:
        ftypenum = 1   # case {'low', 'stop', 'dc-1'}, ftype=1
    elif ftype in ['high', 'pass', 'bandpass', 'DC-0']:
        ftypenum = 0   # case {'high', 'pass', 'bandpass', 'stop'}, ftype=0
    else:
        raise ValueError("fir1: ftype must be 'low', 'high', 'stop', 'bandpass', 'DC-0' or 'DC-1'.")

    if scaleopt == 'scale':
        scale = 1     # case {'scale'}, scale=1
    elif scaleopt == 'noscale':
        scale = 0     # case {'noscale'}, scale=0
    else:
        raise ValueError("fir1: scaleopt must be 'scale' or 'noscale'.")

    # Build response function according to fir2 requirements
    bands = int(np.shape(w)[0]+1)   # bands = length(w)+1
    f = np.zeros(2*bands) # f = zeros(1, 2*bands)
    f[0] = 0                   # f(1) = 0
    f[2*bands-1] = 1           # f(2*bands) = 1
    f[1:2*bands-1:2] = w         # f(2:2:2*bands-1) = w
    f[2:2*bands-1:2] = w         # f(3:2:2*bands-1) = w
    m = np.zeros(2*bands) # m = zeros(1,2*bands)
    m[0:2*bands+1:2] = np.remainder(np.arange(1, bands+1)-(1-ftypenum), 2) # m(1:2:2*bands) = rem([1:bands]-(1-ftype),2)
    m[1:2*bands+1:2] = m[0:2*bands+1:2] # m(2:2:2*bands) = m(1:2:2*bands)

    # Increment the order if the final band is a pass band. Something
    # about having a nyquist frequency of zero causing problems.
    if np.remainder(n,2) == 1 and m[2*bands-1] == 1:
        print("n must be even for highpass and bandstop filters. Incrementing.")
        n = n + 1

        if type(window) == list:
            window = np.array(window)
        
        if type(window) == np.array:
            # Extend the window using interpolation
            M = np.shape(window)[0]     # M = length(window)
            if M == 1:
                window = np.hstack((window, window)) # window = [window; window];
            elif M < 4:
                window = np.interp(np.linspace(0, 1, num=M+1), np.linspace(0, 1, num=M), window) # window = interp1(linspace(0,1,M),window,linspace(0,1,M+1),'linear')
            else:
                func = interpolate.interp1d(np.linspace(0, 1, num=M), window) 
                window = func(np.linspace(0, 1, num=M+1)) # window = interp1(linspace(0,1,M),window,linspace(0,1,M+1),'spline')
    
    # Compute the filter
    print(f)
    b = fir2(n, f, m, grid_n=None, ramp_n=2, window=window)

    # Normalize filter magnitude
    if scale == 1:
        # Find the middle of the first band egde
        # Find the frequency of the normalizing gain
        if m[0] == 1:
            # If the first band edge is a passband, use DC gain
            w_o = 0
        elif f[3] == 1:
            # for a highpass filter,
            # use the gain at half the sample frequency
            w_o = 1
        else:
            # otherwise, use the gain at the center
            # frequency of the first passband
            w_o = f[2] + (f[3]-f[2])/2

        # compute |h(w_o)|^-1
        renorm = 1/np.abs(np.polyval(b, np.exp(-1j*np.pi*w_o))) # renorm = 1/abs(polyval(b, exp(-1i*pi*w_o)))

        # Normalize the filter
        b = renorm * b

    return b