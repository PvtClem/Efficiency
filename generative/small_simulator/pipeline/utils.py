import numpy as np
import uproot
from glob import glob
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import minimum_phase
from scipy.special import erf
import torch
from . import mod_recons_tools as mrt

R2D = 180. / np.pi
altitude = 1264
kb = 1.38064852e-23
c = 299792458
B_dec = 0.
B_inc = np.pi/2. + 1.0609856522873529
Bvec = np.array([np.sin(B_inc)*np.cos(B_dec),np.sin(B_inc)*np.sin(B_dec),np.cos(B_inc)])

def cart2sph(k:np.ndarray)-> tuple:
    """
    Convert cartesian coordinate to spherical coordinate
    """
    if type(k) is np.ndarray:
        r = np.linalg.norm(k, axis=1)
        tp = np.linalg.norm(k[:, :2], axis=1)
        theta = np.arctan2(tp, k[:, 2])
        phi = np.arctan2(k[:, 1], k[:, 0])
    elif type(k) is torch.Tensor:
        r = torch.linalg.norm(k, axis=1)
        tp = torch.linalg.norm(k[:, :2], axis=1)
        theta = torch.arctan2(tp, k[:, 2])
        phi = torch.arctan2(k[:, 1], k[:, 0])
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")
    return r, theta, phi
    
def sph2cart(theta:np.ndarray, phi:np.ndarray, r=1):
    """
    Convert spherical coordinate to cartesian coordinate
    """
    if isinstance(theta, (np.floating, float, np.ndarray)):
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return np.stack((x, y, z), axis=-1)
    elif type(theta) is torch.Tensor:
        x = r*torch.sin(theta)*torch.cos(phi)
        y = r*torch.sin(theta)*torch.sin(phi)
        z = r*torch.cos(theta)
        return torch.stack((x, y, z), dim=-1)
    else:
        raise TypeError(f"Input must be a numpy array or a torch tensor, not {type(theta)}.")


def compute_kxB_kxkxB(k, Bvec=Bvec):
    kxB = np.cross(k, Bvec)
    kxB = kxB / np.linalg.norm(kxB, axis=-1, keepdims=True) 
    kxkxB = np.cross(k, kxB)
    kxkxB = kxkxB / np.linalg.norm(kxkxB, axis=-1, keepdims=True)
    return kxB, kxkxB


def _butter_bandpass_filter(data, lowcut, highcut, fs):
    """subfunction of filt
    """
    b, a = butter(5, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')  # (order, [low, high], btype)
    return lfilter(b, a, data) #causal
    #return filtfilt(b, a, data) #non causal
    
def soft_brickwall_bandpass(X, fs, flow, fhigh, p=8, nfft=None, causal='linear', ntaps=None, axis=-1):
    """
    Soft brickwall bandpass filter for 1D or 2D signals.

    Parameters
    ----------
    X : array, shape (..., n_samples)
        Input signal(s). Can be 1D or multi-D (e.g., (n_signals, n_samples)).
    fs : float
        Sampling frequency.
    flow, fhigh : float
        Passband [flow, fhigh] in Hz.
    p : int
        Exponent controlling steepness (>=2, larger = steeper).
    nfft : int or None
        FFT length for prototype (>= signal length, power of 2 recommended).
    causal : {'linear','min'}
        'linear' = linear-phase causal FIR (delay ~ nfft/2).
        'min'    = minimum-phase causal FIR (approximate magnitude, less delay).
    ntaps : int or None
        Length of truncated FIR for 'min' option.
    axis : int
        Axis of time dimension in X.

    Returns
    -------
    Y : array, same shape as X
        Filtered signal(s).
    h : array
        Filter coefficients used.
    """
    X = np.asarray(X)
    n = X.shape[axis]
    if nfft is None:
        nfft = 2**int(np.ceil(np.log2(n)))  # next power of 2
    
    # frequency mask
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    Hpos = np.exp(-(flow/(freqs+1e-12))**p) * np.exp(-(freqs/fhigh)**p)
    Hpos[freqs == 0] = 0.0
    
    # symmetric impulse response (non-causal prototype)
    h_lin = np.fft.irfft(Hpos, nfft)
    
    if causal == 'linear':
        delay = nfft // 2
        h = np.roll(h_lin, delay)  # linear-phase causal FIR
    elif causal == 'min':
        if ntaps is None:
            ntaps = min(2048, nfft // 4)
        center = nfft // 2
        start = center - ntaps // 2
        h_cut = h_lin[start:start + ntaps]
        h = minimum_phase(h_cut, method='homomorphic')
    else:
        raise ValueError("causal must be 'linear' or 'min'")
    
    # apply FIR filter via convolution along chosen axis
    # np.apply_along_axis handles multiple signals cleanly
    Y = np.apply_along_axis(lambda sig: np.convolve(sig, h, mode='full')[:n], axis, X)
    
    return Y, h

def trace_strong_filtering(data, lowcut, highcut, fs, hard_cut=False):
    """Bandpass filter the trace data.
    """
    if lowcut<1e5:
        fact = 1
    else:
        fact = 1e6

    filtered_data = _butter_bandpass_filter(data, lowcut, highcut, fs)
    if hard_cut:
        filtered_data, h = soft_brickwall_bandpass(filtered_data, fs, 15*fact, 250*fact, p=16)
    return filtered_data

def compute_Xmax_ref(Xmax_pos, Xant, k, kxB, kxkxB):
    Xmax2Xant = Xant - Xmax_pos
    x_sph = ((Xmax2Xant * kxB).sum(axis=1))
    y_sph = ((Xmax2Xant * kxkxB).sum(axis=1))
    l = np.linalg.norm(Xmax2Xant, axis=1)
    eta = np.arctan2(y_sph, x_sph)
    omega = np.arccos(
        np.clip( 
             (k * Xmax2Xant).sum(axis=1) / l, 
             -1, 1)
        )
    return omega, eta, l

def av_ref_index_flat_slow(xmax_pos_z, altitude=altitude):
    """
    Calculate the average refractive index from shower maximum to the antennas.
    Xmax_pos_z: Height of Xmax in the antenna coordinate system !! Not altitude !!
    """
    C = 0.1218 # in km^-1
    k = 3.25e-4 # no unit
    z_ground = altitude / 1e3 # in km
    average_refract = 1 + k/(C*xmax_pos_z) * np.exp(-C * z_ground) * (1 - np.exp(-C * xmax_pos_z)) # average refractive index at altitude xmax_pos_z    return n_ref    
    return average_refract

def av_ref_index_curved_slow(xmax_pos_z, R_x, altitude=altitude):
    """
    Calculate the average refractive index from shower maximum to the antennas.
    Xmax_pos_z: Height of Xmax in the antenna coordinate system !! Not altitude !! !! Not heigh above ground !!
    R_x: Distance from shower maximum to the antenna in km
    """
    
    gamma = np.arcsin(xmax_pos_z / R_x) + np.pi/2
    print(gamma)
    sin_gamma = np.sin(gamma)
    cos_gamma = np.cos(gamma)
    C = 0.1218 # in km^-1
    k = 3.25e-4 # no unit
    R_earth = 6371*1e10 # in km
    z_ground = altitude / 1e3 # in km

    R_s = R_earth/(sin_gamma * sin_gamma) #

    Konstant = np.sqrt(np.pi * R_s / (2 * C))
    in_exp = C * R_s * (1 - sin_gamma*sin_gamma)/2
    Low_erf = np.sqrt(R_s * C / 2) * np.abs(cos_gamma)
    Low_erf_2 = np.sqrt(R_earth * C * cos_gamma * cos_gamma / (2 * sin_gamma * sin_gamma))
    High_erf = R_x /np.sqrt(2 * R_s / C) - Low_erf 
    High_erf_2 = np.sqrt(R_earth * C / 2 * sin_gamma * sin_gamma) * (R_x / R_earth - cos_gamma/(sin_gamma * sin_gamma))
    # assert np.isclose(Low_erf, Low_erf_2).all(), f"{Low_erf} vs {Low_erf_2}"
    # assert np.isclose(High_erf, High_erf_2).all(), f"{High_erf} vs {High_erf_2}"
    print(Low_erf)
    print(High_erf)
    delta_erf = (erf(High_erf_2) + erf(Low_erf_2))
    delta_erf = 2/np.sqrt(np.pi) * np.exp(-Low_erf**2) * (High_erf + Low_erf) # for small arguments
    I_Rx = Konstant  * np.exp(-C * z_ground) * np.exp(-in_exp) * delta_erf
    return 1 + k/R_x * I_Rx

def cherenkov_flat_slow(xmax_pos_z, altitude=altitude):
    """Calculate the Cherenkov angle from shower maximum to the antennas.
    """
    n_ref = av_ref_index_flat_slow(xmax_pos_z, altitude)
    cherenkov_angle = np.arccos(1/n_ref)
    return cherenkov_angle

def cherenkov_curved_slow(xmax_pos_z, R_x):
    """Calculate the Cherenkov angle from shower maximum to the antennas.
    """
    n_ref = av_ref_index_curved_slow(xmax_pos_z, R_x)
    cherenkov_angle = np.arccos(1/n_ref)
    return cherenkov_angle


def convert_to_grams(Xmax, theta, phi, ShowerCoreHeight_=1264):
        k = -sph2cart(theta, phi, 1)
        core = Xmax - k*(Xmax[2]-ShowerCoreHeight_)/k[2]
        XmaxDistance = np.linalg.norm(core - Xmax)
        LongitudinalDistance = mrt.ComputeLongitudinalDistance((phi)*R2D, (np.pi-theta)*R2D, 100e3, ShowerCoreHeight_, *(Xmax))
        biased_grams = mrt.ComputeDistanceGrammage((np.pi-theta)*R2D, XmaxDistance, LongitudinalDistance, ShowerCoreHeight_)
        return biased_grams