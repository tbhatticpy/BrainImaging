import numpy as np
try:
    import functools
except:
    pass    

from typing import Tuple, Sequence, Optional
from scipy.stats import rv_continuous
import statsmodels.api as sm


class singleExp_gen(rv_continuous):
    
    '''Exponential distribution''' 
    def _pdf(self, x: np.ndarray[float], gamma: float) -> np.ndarray[float]:         
        return gamma * np.exp(-gamma*x)
    
singleExp = singleExp_gen(name='singleExp', a=0)
    
class biExp_gen(rv_continuous):
    
    ''' Bi-exponential distribution '''
    def _pdf(self, x: np.ndarray[float], gamma1: float, gamma2: float, delta1: float, delta2: float):
        return (1/(delta1+delta2))*(delta1*gamma1*np.exp(-(gamma1*x)) + delta2*gamma2*np.exp(-(gamma2*x))) 

biExp = biExp_gen(name='biExp', a=0)




eps = 1e-25





class TukeyWeightedNorm(sm.robust.norms.TukeyBiweight):
    ''' 
    Custom Norm Class using Tukey's biweight (bisquare).
    '''
    
    def __init__(self, weights: np.ndarray[float], c: float=4.685, **kwargs):
        super().__init__(**kwargs)
        self.weights_vector = np.array(weights)
        self.flag = 0
        self.c = c
        
    def weights(self, z: np.ndarray[float]):
        """
            Instead of weights equal to one return custom
        INPUT:
            z : 1D array
        OUTPUT:
            weights: np.ndarray
        """
        if self.flag == 0:
            self.flag = 1
            return self.weights_vector.copy()
        else:
            subset = self._subset(z)
            return (1 - (z / self.c)**2)**2 * subset

def _fit_exponents(data): 
    
    ''' Fit data with MLE'''
    
    gamma1, gamma2, delta1, delta2, _, _ = biExp.fit(data, floc=0, fscale=1)
    gamma, _, _ = singleExp.fit(data, floc=0, fscale=1)
    
    biexp_params = [gamma1, gamma2, delta1, delta2]
    exp_params = gamma
    
    return biexp_params, exp_params


def _fit_dfa_exponent(window_lengths: np.ndarray[float], fluct: np.ndarray[float], weighting: str, N_samp: int, fitting: str='Tukey') -> Tuple[np.ndarray[float], np.ndarray[float]]:
    match weighting:
        case 'sq1ox':
            sigma = np.sqrt(window_lengths/N_samp)
        case '1ox':
            sigma = window_lengths/N_samp
        case _:
            raise RuntimeError(f'Weighting {weighting} is not available!')

    match fitting:
        case 'Tukey':
            model = functools.partial(sm.RLM, M=TukeyWeightedNorm(weights=sigma, c=4.685))
        case 'weighted':
            model = functools.partial(sm.WLS, weights=1.0/(sigma**2))
        case 'linfit':
            model = functools.partial(sm.OLS)
        case _:
            raise RuntimeError(f'Fitting {fitting} is not available!')
        

    n_ch = fluct.shape[0]

    fluct_log = np.log2(fluct)
    x = sm.tools.add_constant(np.log2(window_lengths))

    dfa_values = np.zeros(n_ch)
    intercept  = np.zeros(n_ch)    

    for chan_idx, chan_fluct in enumerate(fluct_log):
        intercept[chan_idx], dfa_values[chan_idx] = model(chan_fluct, x).fit().params
    
    return dfa_values, intercept


def _calc_rms(x, scale):
    """
    Windowed Root Mean Square (RMS) with linear detrending.
    Based on Dokato implementation.
    
    INPUT:
      x : 1D array
      scale : int, length of the window in which RMS will be calculaed
    OUTPUT:
      rms : numpy.array, RMS data in each window with length len(x)//scale
    """
    
    # making an array with data divided in windows
    shape = (x.shape[0]//scale, scale)
    X = np.lib.stride_tricks.as_strided(x,shape=shape)
    
    # vector of x-axis points to regression
    scale_ax = np.arange(scale)
    rms = np.zeros(X.shape[0])
    for e, xcut in enumerate(X):
        # fit a line to it
        coeff = np.polyfit(scale_ax, xcut, 1)
        xfit = np.polyval(coeff, scale_ax)
        # detrending and computing RMS of each window
        rms[e] = np.sqrt(np.mean((xcut-xfit)**2))
        
    return rms



def _dfa_conv(data, win_lengths):
    '''
    Conventional approach to calculating DFA.
    Based on Dokato implementation, which follows Hardstone 2012 Front Phys
    and Ihlen 2012 Front Physiol.
    
    Input:
        data_orig: 1D array of amplitude time series
        win_lenghts: 1D array of window lengths in samples.
    Output:
        fluct: Fluctuation function.
        slope: Slopes.

    '''
    xp = np

    y     = xp.cumsum(data - xp.mean(data))
    fluct = xp.zeros(len(win_lengths))
    
    for e, sc in enumerate(win_lengths):
        fluct[e] = xp.sqrt(xp.mean(_calc_rms(y, sc)**2))   
        
    slope = xp.zeros(0)      ### dummy - maybe replace it, if needed? 
    
    return fluct, slope



def _dfa_boxcar(data_orig, win_lengths):
    '''            
    Computes DFA using FFT-based method. (Nolte 2019 Sci Rep)
    Input: 
        data_orig:   1D array of amplitude time series.
        win_lenghts: 1D array of window lengths in samples.
    Output:
        fluct: Fluctuation function.
        slope: Slopes.
    
    '''

    xp = np

    data = xp.array(data_orig, copy=True)
    win_arr = xp.array(win_lengths)
    
    data -= data.mean(axis=-1, keepdims=True)
    data_fft = xp.fft.fft(data)

    n_chans, n_ts = data.shape
    is_odd = n_ts % 2 == 1

    nx = (n_ts + 1)//2 if is_odd else n_ts//2 + 1
    data_power = 2*xp.abs(data_fft[:, 1:nx])**2

    if is_odd == False:
        data_power[:,~0] /= 2
        
    ff = xp.arange(1, nx)
    g_sin = xp.sin(xp.pi*ff/n_ts)
    
    hsin = xp.sin(xp.pi*xp.outer(win_arr, ff)/n_ts)
    hcos = xp.cos(xp.pi*xp.outer(win_arr, ff)/n_ts)

    hx = 1 - hsin/xp.outer(win_arr, g_sin)
    h = (hx / (2*g_sin.reshape(1, -1)))**2

    f2 = xp.inner(data_power, h)

    fluct = xp.sqrt(f2)/n_ts

    hy = -hx*(hcos*xp.pi*ff/n_ts - hsin/win_arr.reshape(-1,1)) / xp.outer(win_arr, g_sin)
    h3 = hy/(4*g_sin**2)

    slope = xp.inner(data_power, h3) / f2*win_arr
    
    return fluct, slope
    




def get_dfa_parameters(data1,min_size=500,max_width=0.2,N_bins=20,overlap=0.25,
                   DF_meth='mean',use_profile=True,weighting='sqrt'):
    N_parc, N_samp = data1.shape    
    
    upper = np.round(N_samp * max_width )
    upper_log = np.log10(upper)
    lower_log = np.log10(min_size)
    
    width_log = (upper_log - lower_log)/N_bins
    
    windows = (np.round(10**np.array([lower_log + width_log*i for i in range(20)]))).astype('int')
    N_win   = len(windows)
    min_idx = int(np.min(np.abs(windows-min_size)))        
    
    
    dfa_pars = {}  
    dfa_pars['N_bins']       = N_bins
    dfa_pars['overlap']      = overlap
    dfa_pars['min_idx']      = min_idx
    dfa_pars['max_width']    = max_width
    dfa_pars['DF_meth']      = DF_meth
    dfa_pars['use_profile']  = use_profile
    dfa_pars['weighting']    = weighting
    dfa_pars['N_win']        = N_win
    dfa_pars['window_lengths']   = windows  

    return dfa_pars




def time_shift(y: np.ndarray):
    """ computes a time-shifted version of a 2D array of size[ch x samp] """
    xp          = np
    n_ch, n_s   = y.shape
    shifts      = xp.random.uniform(0, n_s, n_ch)
    shifts      = shifts.astype(int) 
    data_shift  = xp.zeros([n_ch, n_s],y.dtype)   
    for j in range(n_ch):                                                                   # shift data for surrogates
        s=shifts[j]
        data_shift[j,:] = xp.concatenate((y[j,s:],y[j,:s]))
    return data_shift



def normalize_signal(x):
    """ normalizes a n-dimensional array """
    xp = np
    x_abs = xp.abs(x)
    x_norm = x.copy()
    x_norm /= (x_abs + eps)

    return x_norm



def cplv(x, y: Optional = None, is_normed: bool = False, zero_diag: bool = True, surr: bool = False):
    """
        Computes complex PLV between all channels of input data.
        Input data can be one or two arrays. 
        If input data is cupy array(s), cupy will be used for computations.
    INPUT:
        x: 2D array of size [channels x samples], complex-valued.
        y: Optional, 2D array of complex values
        is_normed: whether data is normalized; if not, then normalization is applied        
        zero_diag: Whether to set the diagonal to zero
        surr: Whether to compute a surrogate value from time-shifted data.
    """
    xp = np

    n_ch, n_s = x.shape

    if is_normed:
        x = x
        y = x if y is None else y
    else:
        x = normalize_signal(x)
        y = x if y is None else normalize_signal(y)
        
    if surr:
        y = time_shift(y)

    avg_diff = xp.inner(x, xp.conj(y)) / n_s
    
    if zero_diag:
        xp.fill_diagonal(avg_diff, 0)

    return avg_diff



def dfa(data: np.ndarray, window_lengths: Sequence, method: str='boxcar', 
            use_gpu: bool=False, fitting ='Tukey', weighting = 'sq1ox') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:    
    """
    Compute DFA with conventional (windowed) or FFT-based 'boxcar' method.
    
    INPUT:
        data:           2D array of size [N_channels x N_samples]. (Amplitude envelope)
        window_lengths: sequence of window sizes, should be in samples.
        method:         either 'conv' or 'boxcar' 
        use_gpu:        If True, input np.array is converted to cp in function
        fitting:        'linfit' for regular unweighted linear fit, 
                        'Tukey' for biweight/bisquare,
                        'weighted' for weighted linear fit.
        weighting:      'sq1ox' or '1ox' 
                    
    OUTPUT:        
        fluctuation: 2D array of size N_channels x N_windows), 
        slope:       2D array of size N_channels x N_windows), 
        DFA:         1D vector of size N_channels 
        residuals:   1D vector of size N_channels        
    """

    xp = np
    
    
    allowed_methods = ('boxcar','conv' )
    if not(method in allowed_methods):
        raise RuntimeError('Method {} is not allowed! Only {} are available'.format(method, ','.join(allowed_methods)))

    allowed_weightings = ('sq1ox', '1ox')
    if not(weighting in allowed_weightings):
        raise RuntimeError('Weighting {} is not allowed! Only {} are available'.format(weighting, ','.join(allowed_weightings)))

    if method == 'conv':
        fluct, slope =  _dfa_conv(data, window_lengths)
    elif method == 'boxcar':
        fluct, slope =  _dfa_boxcar(data, window_lengths)
        
    if not(xp is np):
        fluct = xp.asnumpy(fluct)
        slope = xp.asnumpy(slope)
    
    dfa_exponents, residuals = _fit_dfa_exponent(window_lengths, fluct, weighting=weighting, N_samp=data.shape[-1], fitting=fitting)

    return fluct, slope, dfa_exponents, residuals





