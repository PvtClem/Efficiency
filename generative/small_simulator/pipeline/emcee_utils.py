import emcee
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import hilbert

import sys

sys.path.append("/pbs/home/a/aferrier/WorkingDir/DU_response_computation/")
from matplotlib.lines import Line2D

#from PWF_reconstruction.recons_PWF import PWF_semianalytical
from .utils import R2D
from .input_formating import event_swf_time

from .inference import to_voltage
import torch


SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)



# =============================================================================
# PREPROCESSING & SNR SELECTION
# =============================================================================

def preprocess_traces(all_traces: np.ndarray, fs_input: float, fs_output, t_SN, t_EW, t_Z, tf, noise_computer, antenna_pos, Xmax_pos, smearing: float = 0.):
    """Preprocess traces with bandpass filtering and add noise."""
    noise_voltage = sample_galactic_noise(noise_computer, all_traces.shape[0], lst_hour=18)
    traces_voltage, _ = to_voltage(all_traces, antenna_pos, Xmax_pos, fs_input, fs_output, t_SN, t_EW, t_Z, tf, duration=2.048, to_adc=False)
    amps = np.linalg.norm(traces_voltage, axis=1).max(axis=-1)
    measured_signal = traces_voltage + noise_voltage
    smearing_factor = 1 + np.random.normal(0, smearing, size=(measured_signal.shape[0], 1, 1))
    measured_signal *= smearing_factor
    
    return traces_voltage, measured_signal, amps, smearing_factor-1


def pick_top_snr_indices(snr: np.ndarray, top_k: int):
    """Select indices of top-k SNR antennas."""
    strong_idx = np.isin(np.arange(len(snr)), np.argsort(snr)[-top_k:])
    return strong_idx


def pick_strong_antennas(amps: np.ndarray, threshold: int):
    """Select antennas above amplitude threshold."""
    strong_idx = amps > threshold
    return strong_idx


def sample_galactic_noise(noise_computer, n_antennas, lst_hour=18):

    noise_traces, noise_fft = noise_computer.noise_samples(
        lst_hour=lst_hour, n_samples=n_antennas, micro=True,
    )
    return noise_traces


def compute_std(traces):
    """Estimate per-polarization noise std from off-peak regions."""
    hilbert_traces = hilbert(traces, axis=-1)
    envelope = np.linalg.norm(np.abs(hilbert_traces), axis=1)
    loc_max = envelope.argmax(axis=1)
    med_pos = int(np.median(loc_max))
    mask = np.ones(traces.shape[-1], dtype=bool)
    mask[max(med_pos - 20, 0):min(med_pos + 40, len(mask))] = False
    std_polar = np.mean(np.std(traces[:, :, mask], axis=-1), axis=0)
    return std_polar

##############################################################################
###### Sampler helpers
##############################################################################

class EventContext:
    """
    Precomputes and caches all invariant quantities for a single event.
    
    This replaces the pattern of passing a DataFrame into the inner loop
    and calling .values, .copy(), column assignment, etc. 21,000 times.
    
    All arrays are stored as float32 for consistency with the model.
    """
    
    def __init__(self, df_ev, config_inputs, model, fs_ds,
                 measured_signal, noise_std, smearing, 
                 t_SN, t_EW, t_Z, tf,
                 times_noisy=None, sigma_t=None, alpha=0.5,
                 n_2=10, pad_left=0, pad_right=0,
                 k_guess=None, Xs_guess=None,
                 angle_conf=0.5, log_E_conf=1.5, Xs_conf=50e3,
                 amps=None):
        """
        Precompute everything that doesn't change across MCMC iterations.
        
        Parameters
        ----------
        df_ev : pd.DataFrame.     Event dataframe (used once, then discarded).
        config_inputs : list of str.     Input feature names for the model.
        model : MLP_metamodel.     The neural network model.
        fs_ds : float.     Sampling frequency.
        measured_signal : np.ndarray, shape (n_ant, 3, n_samples).     Noisy voltage traces.
        noise_std : float.     Noise standard deviation.
        smearing : float.     Smearing factor for heteroscedastic noise model.
        t_SN, t_EW, t_Z : objects.     Antenna response data tables.
        tf : np.ndarray.     Transfer function.
        times_noisy : np.ndarray, optional.     Noisy arrival times for hybrid method.
        sigma_t : float, optional.     Time uncertainty for hybrid method.
        alpha : float.     Balance between shape and time likelihood.
        k_guess : np.ndarray, shape (3,), optional.     Initial guess shower direction (for amplitude prior).
        Xs_guess : np.ndarray, shape (3,), optional.     Initial guess Xmax position (for amplitude prior).
        angle_conf : float.     Angular prior width in degrees (for amplitude prior).
        log_E_conf : float.     Energy prior width in log10 decades (for amplitude prior).
        Xs_conf : float.     Xmax position prior width in meters (for amplitude prior).
        """
        # Store config
        self.config_inputs = config_inputs
        self.model = model
        self.fs_ds = fs_ds
        self.n_2 = n_2
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.alpha = alpha
        self.smearing = smearing
        self.noise_std = noise_std
        self.sigma_t = sigma_t
        
        # --- Extract numpy arrays from DataFrame ONCE ---
        self.antenna_pos = df_ev[['du_pos_x', 'du_pos_y', 'du_pos_z']].values.astype(np.float64)
        self.xmax_pos = df_ev[['xmax_pos_x', 'xmax_pos_y', 'xmax_pos_z']].values.astype(np.float64)
        self.n_ant = self.antenna_pos.shape[0]
        
        # Store all columns needed by make_input_array as a dict of numpy arrays
        self.base_values = {}
        for col in df_ev.columns:
            try:
                self.base_values[col] = df_ev[col].values.astype(np.float64).copy()
            except (ValueError, TypeError):
                self.base_values[col] = df_ev[col].values.copy()
        
        # Measured signal (keep as provided, typically float64)
        self.measured_signal = np.ascontiguousarray(measured_signal)
        
        # Times for hybrid method
        if times_noisy is not None:
            self.times_noisy = times_noisy.astype(np.float64)
            self.times_noisy_centered = self.times_noisy - self.times_noisy.mean()
        else:
            self.times_noisy = None
            self.times_noisy_centered = None
        
        # --- Precompute antenna response data ---
        self.t_SN = t_SN
        self.t_EW = t_EW
        self.t_Z = t_Z
        self.tf = tf
        
        # --- Pre-slice measured signal for shape comparison ---
        pl = self.pad_left
        pr = self.pad_right if self.pad_right > 0 else None
        if pr is not None:
            self.measured_trimmed = self.measured_signal[:, :, pl:-pr]
        else:
            self.measured_trimmed = self.measured_signal[:, :, pl:]
        
        # Precompute the "delayed" (trimmed by n_2) version for _min_error_trace
        self.measured_delayed = self.measured_trimmed[:, :, n_2:-n_2]
        self.hilbert_delayed = np.linalg.norm(np.abs(hilbert(self.measured_delayed, axis=-1)), axis=1)

        # Noise variance terms (precompute denominator parts)
        if isinstance(noise_std, np.ndarray):
            self.noise_var_2 = noise_std[None, :, None] ** 2
        else:
            self.noise_var_2 = noise_std ** 2
        
        # Check if 'n_eff' is in config_inputs (for SWF with refraction)
        self.has_n_eff = 'n_eff' in config_inputs
        if self.has_n_eff:
            self.n_eff_idx = config_inputs.index('n_eff')
        
        # Pre-allocate delay search arrays
        self.delays_range = np.arange(-n_2, n_2 + 1)
        
        # --- Set model to eval mode and ensure no_grad context ---
        self.model.eval()
        
        # --- torch.compile for faster inference (PyTorch 2.0+) ---
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("  torch.compile applied (mode=reduce-overhead)")
            except Exception as e:
                print(f"  torch.compile skipped: {e}")
        
        # --- Precompute model normalizer on correct device ---
        self.device = getattr(model, 'device', 'cpu')
        
        # Pre-extract the duration and frequency axis for pred_trace_kxB
        duration = 2.048
        self.trace_len = int(fs_ds * duration)
        self.freqs_np = np.linspace(0, fs_ds / 2, int(fs_ds * duration / 2) + 1)
        self.freqs_torch = torch.tensor(self.freqs_np, dtype=torch.float32, device=self.device)
        
        # Speed of light for SWF
        self.c = 299792458.0
        
        # --- Amplitude-based reconstruction context ---
        # Alias for API compatibility with log_likelihood_amplitude
        self.du_pos = self.antenna_pos
        
        # Measured amplitudes and per-polarization noise std
        self.measured_amps = get_amps(measured_signal) if amps is None else amps
        self.std_polar = noise_std if isinstance(noise_std, np.ndarray) else np.array([noise_std, noise_std, noise_std], dtype=np.float64)
        
        # Array center
        self.mean_pos = self.antenna_pos.mean(axis=0)
        
        # Prior knowledge from other reconstruction methods
        if k_guess is not None:
            self.k_guess = np.asarray(k_guess, dtype=np.float64)
            cos_theta = -self.k_guess[2]
            self.D_guess = Xmax_Distance_fit(cos_theta) * 1e3   # km → m
            self.D_conf = Xmax_Distance_conf(cos_theta) * 1e3
        else:
            self.k_guess = None
            self.D_guess = None
            self.D_conf = None
        
        if Xs_guess is not None:
            self.Xs_guess = np.asarray(Xs_guess, dtype=np.float64)
        else:
            self.Xs_guess = None
        
        self.angle_conf = angle_conf
        self.log_E_conf = log_E_conf
        self.Xs_conf = Xs_conf
        
        print(f"  EventContext initialized: {self.n_ant} antennas, "
              f"{self.measured_signal.shape[-1]} samples, "
              f"device={self.device}")
        
        self.Cov_traces = np.sum(self.noise_var_2)/9 * np.eye(self.hilbert_delayed.shape[-1], dtype=np.float64) +\
                            self.smearing**2 * self.hilbert_delayed[:,:,None] * self.hilbert_delayed[:,None,:]
        self.det_Cov_traces = np.linalg.det(self.Cov_traces)
        self.inv_Cov_traces = np.linalg.inv(self.Cov_traces)

    def copy(self):
        """Create a copy of the context for use in parallel sampling."""
        new_ctx = EventContext.__new__(EventContext)  # Create uninitialized instance
        new_ctx.__dict__.update(self.__dict__)  # Shallow copy of all attributes
        return new_ctx




def min_error_trace(preds, measured_delayed, n_2):
    """
    Optimized delay search using sliding window view.
    
    Uses precomputed measured_delayed from EventContext.
    """
    l = preds.shape[-1]
    window_len = l - 2 * n_2
    preds_delayed = sliding_window_view(preds, window_len, axis=2)
    
    errors = ((preds_delayed - measured_delayed[:, :, None, :]) ** 2).sum(axis=(1, 3))
    
    delays = np.arange(-n_2, n_2 + 1)
    delay_best = delays[np.argmin(errors, axis=1)]
    return delay_best


def apply_shift(x, ctx):
    """
    Apply parameter shifts using pure numpy arrays instead of DataFrame operations.
    
    Returns modified copies of the arrays that make_input_array needs.
    
    ~100x faster than apply_shift() which copies a DataFrame each call.
    
    OPTIMIZATION: Only copies arrays that actually change (energy, xmax, angles).
    Uses a shallow dict copy + targeted array copies.
    """
    dx, dy, dz, dE, dtheta, dphi = x
    
    # Shallow copy of dict (O(n_columns) pointer copies, no array copies)
    values = dict(ctx.base_values)
    
    # Only copy+modify arrays that change with the shift
    if 'energy_em' in values:
        values['energy_em'] = ctx.base_values['energy_em'] * (1 + dE)
    if 'energy_primary' in values:
        values['energy_primary'] = ctx.base_values['energy_primary'] * (1 + dE)
    
    values['xmax_pos_x'] = ctx.base_values['xmax_pos_x'] + dx
    values['xmax_pos_y'] = ctx.base_values['xmax_pos_y'] + dy
    values['xmax_pos_z'] = ctx.base_values['xmax_pos_z'] + dz
    values['zenith'] = ctx.base_values['zenith'] + dtheta * np.pi / 180
    values['azimuth'] = ctx.base_values['azimuth'] + dphi * np.pi / 180
    
    # Shifted Xmax positions (antenna positions don't change)
    xmax_shifted = ctx.xmax_pos + np.array([[dx, dy, dz]])
    
    return values, ctx.antenna_pos, xmax_shifted



#################################################################################
###### Other utilities
#################################################################################

def get_amps(traces):
    """Get peak Hilbert envelope amplitude per antenna."""
    hilbert_traces = hilbert(traces, axis=-1)
    envelope = np.linalg.norm(np.abs(hilbert_traces), axis=1)
    amps = envelope.max(axis=1)
    return amps


def Xmax_Distance_fit(cos_theta):
    """Empirical Xmax distance vs cos(theta) fit."""
    a, b, c = 12.72862024, -16.58390211, -20.95807057
    return a * 1 / cos_theta + b * np.log(cos_theta) + c


def Xmax_Distance_conf(cos_theta):
    """Empirical distance confidence vs cos(theta)."""
    return 20 + 1.0 / (cos_theta ** 1.5)



#def pwf_guess(time_noisy, du_pos):
    """
    Get a PWF-based guess for the direction
    """
    #theta_pwf, phi_pwf = PWF_semianalytical(du_pos, time_noisy)
    #return theta_pwf, phi_pwf


def omega_init(X_guess, du_pos, k_swf):
    V_du = du_pos - X_guess
    V_du_norm = np.linalg.norm(V_du, axis=1)
    cos_omega = (k_swf[None,:] * V_du).sum(axis=1) / V_du_norm
    omega = np.arccos(cos_omega)*R2D
    max_omega = np.max(omega)
    alpha = 1.1/max_omega
    A0 = V_du_norm.mean()
    Delta = A0*(1-1/alpha)
    X_guess_new = X_guess + Delta * k_swf
    return X_guess_new

def recons_swf(du_times, du_pos, k_guess, n_walkers=20, n_steps=2000):
    cos_theta = -k_guess[2]


    angle_conf = 1 #1°
    D_guess = Xmax_Distance_fit(cos_theta) * 1e3 # Convert from km to m
    D_conf = 2*Xmax_Distance_conf(cos_theta) * 1e3 # Convert from km to m
    def log_prior(X_cand):
        mean_pos = du_pos.mean(axis=0)
        Xmax_in_ref = X_cand - mean_pos
        D = np.sqrt( np.square(Xmax_in_ref).sum())
        Angle = np.arccos((Xmax_in_ref/D * (-k_guess)).sum())
        angle_prior = -(Angle*R2D)**2/(2 * angle_conf**2) 
        D_prior = -(D - D_guess)**2/(2*D_conf**2)
        return angle_prior + D_prior
    
    def log_likelihood(X_cand):
        swf_times = event_swf_time(X_cand, du_pos)
        residuals = du_times - swf_times
        residuals -= residuals.mean()
        return -np.square(residuals).sum()/(2*7e-9**2) # 1 ns std dev
    
    def log_posterior(X_cand):
        return log_prior(X_cand) + log_likelihood(X_cand)
    
    ndim = 3
    D0 = 100e3
    X0 = du_pos.mean(axis=0) + D0 * (-k_guess)
    pos_init = X0 + np.random.randn(n_walkers, ndim) * 1000
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    sampler.run_mcmc(pos_init, n_steps)

    samples = sampler.get_chain(discard=1000, flat=True, thin=10)
    best_idx = np.argmax(sampler.get_log_prob(discard=1000, flat=True, thin=10))
    best_xmax = samples[best_idx]

    return best_xmax



# =============================================================================
# Plots

import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

def hdi_1d(samples, cred_mass=0.68):
    """Compute highest density interval (HDI) for a 1D sample."""
    x = np.sort(samples)
    n = len(x)
    interval = int(np.floor(cred_mass * n))
    widths = x[interval:] - x[:n - interval]
    i = np.argmin(widths)
    return x[i], x[i + interval]


def map_kde_1d(samples, grid_size=2000, subsample=20000):
    if samples.ndim == 1:
        samples = samples[:, None]

    N, D = samples.shape
    maps = np.zeros(D)

    for d in range(D):

        x = samples[:, d]

        # optional subsampling
        if len(x) > subsample:
            idx = np.random.choice(len(x), subsample, replace=False)
            x = x[idx]

        kde = gaussian_kde(x)

        xmin, xmax = x.min(), x.max()
        grid = np.linspace(xmin, xmax, grid_size)

        density = kde(grid)
        maps[d] = grid[np.argmax(density)]

    return maps



def hdi_1d(samples, cred_mass=0.68):
    """Compute highest density interval (HDI) for a 1D sample."""
    x = np.sort(samples)
    n = len(x)
    interval = int(np.floor(cred_mass * n))
    widths = x[interval:] - x[:n - interval]
    i = np.argmin(widths)
    return x[i], x[i + interval]


def map_kde_1d(samples, grid_size=2000, subsample=20000):
    if samples.ndim == 1:
        samples = samples[:, None]

    N, D = samples.shape
    maps = np.zeros(D)

    for d in range(D):

        x = samples[:, d]

        # optional subsampling
        if len(x) > subsample:
            idx = np.random.choice(len(x), subsample, replace=False)
            x = x[idx]

        kde = gaussian_kde(x)

        xmin, xmax = x.min(), x.max()
        grid = np.linspace(xmin, xmax, grid_size)

        density = kde(grid)
        maps[d] = grid[np.argmax(density)]

    return maps

def corner_sns(samples, labels=None, truths=None, title=None, show=True,
               show_titles=True, title_kwargs=None, recons=None, **kwargs):
    """Corner plot using seaborn pairplot with KDE contours, mimicking corner.corner syntax.
    
    Parameters
    ----------
    samples : array-like, shape (n_samples, n_dim)
        The samples to plot.
    labels : list of str, optional
        Labels for each dimension.
    truths : list of float, optional
        True values to mark on the plot.
    title : str, optional
        Super title for the figure.
    show : bool, optional
        Whether to call plt.show().
    show_titles : bool, optional
        Whether to show titles on diagonal subplots with median and quantiles.
    title_kwargs : dict, optional
        Keyword arguments passed to ax.set_title() for diagonal titles.
    **kwargs : dict
        Additional keyword arguments (ignored for compatibility).
    
    Returns
    -------
    g : sns.PairGrid
        The seaborn PairGrid object.
    """
    if title_kwargs is None:
        title_kwargs = {}
    
    samples = np.atleast_2d(samples)
    if samples.ndim == 3:
        samples = samples.reshape(-1, samples.shape[-1])
    
    n_dim = samples.shape[1]
    if labels is None:
        labels = [f'x_{i}' for i in range(n_dim)]
    
    df_samples = pd.DataFrame(samples, columns=labels)
    
    # Create PairGrid manually instead of pairplot
    g = sns.PairGrid(df_samples, corner=True, diag_sharey=False)
    
    # Diagonal: KDE plots
    g.map_diag(sns.kdeplot, color='#4C72B0', fill=True, alpha=0.4, linewidth=1.5)
    
    if recons is None:
        recons = df_samples.median()
        left_bound = df_samples.quantile(0.16) 
        right_bound = df_samples.quantile(0.84)
    else:
        recons = pd.Series(np.array(recons), index=labels)
        
        left_bound = {}
        right_bound = {}

        for label in labels:
            lo, hi = hdi_1d(df_samples[label].values, cred_mass=0.68)
            left_bound[label] = lo
            right_bound[label] = hi

        left_bound = pd.Series(left_bound)
        right_bound = pd.Series(right_bound)
    if truths is not None:
        truths = pd.Series(np.array(truths), index=labels)
    
    # Off-diagonal: Only 2D KDE contours
    for i in range(n_dim):
        for j in range(i):
            ax = g.axes[i, j]
            x = df_samples[labels[j]].values
            y = df_samples[labels[i]].values
            
            # Subsample for speed if necessary
            if len(x) > 10000:
                idx = np.random.choice(len(x), size=10000, replace=False)
                x_sub, y_sub = x[idx], y[idx]
            else:
                x_sub, y_sub = x, y
                
            kde = gaussian_kde(np.vstack([x_sub, y_sub]))
            
            # Use current limits or provide a slightly padded range
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            dx, dy = (xmax - xmin) * 0.1, (ymax - ymin) * 0.1
            xx, yy = np.mgrid[xmin-dx:xmax+dx:100j, ymin-dy:ymax+dy:100j]
            
            positions = np.vstack([xx.ravel(), yy.ravel()])
            zz = kde(positions).reshape(xx.shape)
            
            # Probability levels
            levels_frac = [0.989, 0.865, 0.393]
            zz_sorted = np.sort(zz.ravel())[::-1]
            cumsum = np.cumsum(zz_sorted) / np.sum(zz_sorted)
            contour_levels = [zz_sorted[np.searchsorted(cumsum, f)] for f in levels_frac]
            contour_levels = sorted(contour_levels)
            
            ax.contour(xx, yy, zz, levels=contour_levels, colors=['#C44E52', '#E8A838', '#4C72B0'],
                       linewidths=[1.0, 1.2, 1.5], alpha=0.8)
    
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(labelsize=12)
            ax.xaxis.label.set_size(1.2*BIGGER_SIZE)
            ax.yaxis.label.set_size(1.2*BIGGER_SIZE)
    
    # Add medians and quantiles on diagonals, crosshairs on off-diagonals
    for i, label in enumerate(labels):
        g.axes[i, i].axvline(recons[label], color='#C44E52', ls='--', lw=1.5, zorder=100)
        g.axes[i, i].axvspan(left_bound[label], right_bound[label], color='#C44E52', alpha=0.1, zorder=100)
        # if show_titles:
        #     plus = right_bound[label] - recons[label]
        #     minus = recons[label] - left_bound[label]
        #     t = f'{label} = {recons[label]:.2f}$^{{+{plus:.2f}}}_{{-{minus:.2f}}}$'
        #     default_title_kwargs = {'fontsize': BIGGER_SIZE}
        #     default_title_kwargs.update(title_kwargs)
        #     g.axes[i, i].set_title(t, **default_title_kwargs)
        if show_titles:
            plus = right_bound[label] - recons[label]
            minus = recons[label] - left_bound[label]
            # Extract the unit from the label if it exists
            if '[' in label and ']' in label:
                quantity, unit = label.split('[')
                unit = unit.strip(']')
                quantity = quantity.strip()
                t = f'{quantity} = {recons[label]:.2f}$^{{+{plus:.2f}}}_{{-{minus:.2f}}}$ {unit}'
            else:
                t = f'{label} = {recons[label]:.2f}$^{{+{plus:.2f}}}_{{-{minus:.2f}}}$'
            default_title_kwargs = {'fontsize': 1.2*BIGGER_SIZE}
            default_title_kwargs.update(title_kwargs)
            g.axes[i, i].set_title(t, **default_title_kwargs)

        for j in range(i):
            g.axes[i, j].scatter(recons[labels[j]], recons[label], color='#C44E52', marker='x', s=100, lw=5, zorder=100)
    
    # Mark truths if provided
    if truths is not None:
        for i, label in enumerate(labels):
            g.axes[i,i].axvline(truths[label], c='green', ls='-', lw=1.5)
            for j in range(i):
                g.axes[i, j].scatter(truths[labels[j]], truths[label], color='green', marker='o', s=100, edgecolor='none', lw=1.5)
    
    # Add legend
    handles = [
        Line2D([0], [0], color='#C44E52', lw=2, label='recons'),
        Line2D([0], [0], color='#C44E52', alpha=0.1, lw=24, label='68% CI')
    ]
    if truths is not None and any(t is not None for t in truths):
        handles.append(Line2D([0], [0], color='green', lw=1.5, label='truth'))
    g.figure.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.6, .85), frameon=False, fontsize=1.2*BIGGER_SIZE)
    
    suptitle = title if title is not None else 'MCMC Posterior Distributions'
    g.figure.suptitle(suptitle, y=1.02, fontsize=16)
    g.figure.tight_layout()
    if show:
        plt.show()
    g.figure.align_ylabels()
    
    return g