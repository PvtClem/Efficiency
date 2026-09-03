import matplotlib.pyplot as plt
from flask import ctx
import numpy as np

from pipeline.inference import predict_voltage
from .utils import sph2cart, R2D, c
from .emcee_utils import (min_error_trace, 
                          apply_shift,
                          get_amps)
from .inference import predict_voltage
from .input_formating import event_swf_time, make_input_array, ZHSEffectiveRefractionIndexvect
import emcee
from numpy.lib.stride_tricks import sliding_window_view


from scipy.signal import convolve, fftconvolve
from scipy.special import logsumexp
#######################################################
###### PRIOR FUNCTIONS
#######################################################

def log_prior_informative(x_batch, ctx):
    """
    Informative Gaussian log-prior for shift-based parametrization.

    Converts shifts (dx, dy, dz, dE, dtheta, dphi) to absolute quantities
    then applies Gaussian priors on direction, distance, Xmax position and energy.

    Works with both single walker (6,) and batched (nwalkers, 6) inputs.

    Requires ctx to have: base_values, k_guess, angle_conf, D_guess, D_conf,
    Xs_guess, Xs_conf, log_E_conf, mean_pos.
    """
    x_batch = np.atleast_2d(x_batch)
    xmax_pos = np.array([ctx.base_values['xmax_pos_x'][0],
                         ctx.base_values['xmax_pos_y'][0],
                         ctx.base_values['xmax_pos_z'][0]])
    Xmax_cand = x_batch[:, :3] + xmax_pos[None, :]

    log_E_cand = np.log10( 
        np.clip(ctx.base_values['energy_em'][0] * (1 + x_batch[:, 3]), 1e-1, 1e13)
    )

    theta_cand_rad = np.radians(x_batch[:, 4]) + ctx.base_values['zenith'][0]
    phi_cand_rad = np.radians(x_batch[:, 5]) + ctx.base_values['azimuth'][0]
    k_cand = -sph2cart(theta_cand_rad, phi_cand_rad)

    mean_pos = ctx.mean_pos
    D = np.sqrt(np.square(Xmax_cand - mean_pos[None, :]).sum(axis=1))

    cos_a0 = np.clip((k_cand * ctx.k_guess[None, :]).sum(axis=1), -1, 1)
    Angle_0 = np.arccos(cos_a0)
    cos_a1 = np.clip(((mean_pos[None, :] - Xmax_cand) * ctx.k_guess[None, :]).sum(axis=1) / D, -1, 1)
    Angle_1 = np.arccos(cos_a1)

    angle_prior_0 = -(Angle_0 * R2D) ** 2 / (2 * ctx.angle_conf ** 2)
    angle_prior_1 = -(Angle_1 * R2D) ** 2 / (2 * (2 * ctx.angle_conf) ** 2)
    D_prior = -(D - ctx.D_guess) ** 2 / (2 * ctx.D_conf ** 2)
    Xs_prob = -np.sum((Xmax_cand - ctx.Xs_guess[None, :]) ** 2, axis=1) / (2 * ctx.Xs_conf ** 2)
    energy_prior = -(log_E_cand - 8.5) ** 2 / (2 * ctx.log_E_conf ** 2)

    result = angle_prior_0 + angle_prior_1 + D_prior + energy_prior + Xs_prob
    # Squeeze back to scalar if single-walker input
    if result.shape[0] == 1:
        return result[0]
    return result

def log_prior_vectorized(x_batch):
    """
    Vectorized log-prior for a batch of walker positions.
    
    Parameters
    ----------
    x_batch : np.ndarray, shape (nwalkers, 6)
    
    Returns
    -------
    np.ndarray, shape (nwalkers,) — log-prior for each walker
    """
    dx, dy, dz, dE, dtheta, dphi = x_batch.T
    
    valid = (
        (dx > -30000) & (dx < 30000) &
        (dy > -30000) & (dy < 30000) &
        (dz > -30000) & (dz < 30000) &
        (dE > -0.4) & (dE < 1.5) &
        (dtheta > -7) & (dtheta < 7) &
        (dphi > -7) & (dphi < 7)
    )
    
    result = np.where(valid, 0.0, -np.inf)
    return result


#######################################################
###### LIKELIHOOD FUNCTIONS
#######################################################

def compute_shape_llh(preds_voltage, ctx):
    """
    Compute shape log-likelihood using precomputed quantities from EventContext.
    """
    pl = ctx.pad_left
    pr = ctx.pad_right
    if pr > 0:
        preds_trimmed = preds_voltage[:, :, pl:-pr]
    else:
        preds_trimmed = preds_voltage[:, :, pl:]
    
    measured_trimmed = ctx.measured_trimmed
    
    # Find optimal delay
    delays = min_error_trace(preds_trimmed, ctx.measured_delayed, ctx.n_2)
    
    # Apply delay (np.roll per antenna)
    rolled = np.empty_like(preds_trimmed)
    for i, d in enumerate(delays):
        rolled[i] = np.roll(preds_trimmed[i], d, axis=-1)
    
    # Compute error
    error = (measured_trimmed - rolled) ** 2
    
    if ctx.smearing > 0:
        if isinstance(ctx.noise_std, np.ndarray):
            denom = 2 * (ctx.smearing ** 2 * rolled ** 2 + ctx.noise_var_2)
        else:
            amp_2 = rolled[:, 0, :] ** 2 + rolled[:, 1, :] ** 2 + rolled[:, 2, :] ** 2
            denom = 2 * (ctx.smearing ** 2 * amp_2[:, None, :] + ctx.noise_var_2)
        tot_error = (error / denom).sum()
    else:
        tot_error = error.sum() / (2 * ctx.noise_var_2)
    
    return -tot_error

def compute_shape_llh_cov(preds_voltage, ctx):
    """
    Compute shape log-likelihood using covariance matrix from EventContext.
    """
    pl = ctx.pad_left
    pr = ctx.pad_right
    if pr > 0:
        preds_trimmed = preds_voltage[:, :, pl:-pr]
    else:
        preds_trimmed = preds_voltage[:, :, pl:]
        
    # Find optimal delay
    l = preds_trimmed.shape[-1]
    window_len = l - 2 * ctx.n_2
    preds_trimmed_delayed = sliding_window_view(preds_trimmed, window_len, axis=2)
    residual = preds_trimmed_delayed - ctx.measured_delayed[:, :, None, :]  # shape (n_ant, n_pol, n_delays, window_len)
    
    n_ant, n_pol, n_delays, w = residual.shape

    #cholesky-hybrid:
    # if not hasattr(ctx, 'chol_inv_Cov_traces'):
    #     ctx.chol_inv_Cov_traces = np.ascontiguousarray(np.linalg.cholesky(ctx.inv_Cov_traces))
    # L = ctx.chol_inv_Cov_traces  # (n_ant, w, w)
    # res_rs = np.ascontiguousarray(residual.reshape(n_ant, n_pol * n_delays, w))
    # transformed = np.matmul(res_rs, L)
    # errors_flat = np.sum(transformed * transformed, axis=-1)
    # errors = errors_flat.reshape(n_ant, n_pol, n_delays).sum(axis=1)

    # matmul variant:
    res_rs = np.ascontiguousarray(residual.reshape(n_ant, n_pol * n_delays, w))
    tmp = np.matmul(res_rs, ctx.inv_Cov_traces)
    errors_flat = np.sum(tmp * res_rs, axis=-1)
    errors = errors_flat.reshape(n_ant, n_pol, n_delays).sum(axis=1)

    # Only keep the minimum error across delays for each antenna
    # llh_shape = -0.5 * np.sum(np.min(errors, axis=1))

    # Keep the average error over delays
    llh_shape = -0.5 * np.sum(errors)

    # plt.show()
    # fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    # for i in range(3):
    #     ax[i].plot(ctx.measured_delayed[0, i], label='Measured Pol 0')
    #     ax[i].plot(preds_trimmed_delayed[0, i, ctx.n_2], label='Predicted Pol 0')
    #     ax[i].set_title(f'polar {i} - llh: {llh_shape:.2f}')
    #     ax[i].plot(np.diag(ctx.Cov_traces[0])*3, label='Cov diag x3', ls='--')
    #     ax[i].legend()

    # fig.suptitle(f'Shape log-likelihood: {llh_shape:.2f}')
    # plt.tight_layout()
    # plt.show()
    return llh_shape


def compute_swf_llh(xmax_shifted, ctx, n_effs=None):
    """
    Compute SWF time log-likelihood using pure numpy.
    """
    if ctx.times_noisy is None:
        return 0.0
    
    X_ants = ctx.antenna_pos  # shape (n_ant, 3)
    
    swf_times = event_swf_time(xmax_shifted, X_ants, n_effs=n_effs)
    swf_times -= swf_times.mean()
    
    error = (swf_times - ctx.times_noisy_centered) ** 2 / (2 * ctx.sigma_t ** 2)
    return -np.sum(error)



def log_likelihood_pulse(x, ctx):
    """
    Optimized log-likelihood that replaces log_likelihood_hybrid/log_likelihood_shape.
    
    Eliminates:
    - DataFrame copy/modify per call
    - Redundant .values extractions  
    - Repeated numpy↔torch conversions
    - Recomputation of invariant quantities
    """
    # 1. Apply shift (pure numpy, no DataFrame)
    values, antenna_pos, xmax_shifted = apply_shift(x, ctx)
    
    # 2. Compute input array (pure numpy, no DataFrame)
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    
    # 3. Model inference + E-field → Voltage (full pipeline)
    Xs_first = xmax_shifted[0]  # All antennas see same Xmax
    preds_voltage = predict_voltage(input_arr, kxB_loc, antenna_pos, Xs_first, ctx.model, ctx.fs_ds, ctx.t_SN, ctx.t_EW, ctx.t_Z, ctx.tf)
    # 4. Shape likelihood
    llh_shape = compute_shape_llh(preds_voltage, ctx)
    
    # 5. Time likelihood (if hybrid)
    if ctx.times_noisy is not None and ctx.sigma_t is not None:
        # Check for n_eff
        n_effs = None
        if ctx.has_n_eff:
            # n_effs = input_arr[:, ctx.n_eff_idx] + 1
            n_effs = input_arr[:, ctx.n_eff_idx] + 1
        llh_time = compute_swf_llh(Xs_first, ctx, n_effs=n_effs)
        return 2 * ctx.alpha * llh_shape + 2 * (1 - ctx.alpha) * llh_time
    else:
        return llh_shape
    

def log_likelihood_amplitude(params, ctx):
    """
    Amplitude-based log-likelihood.

    Compares predicted vs measured peak Hilbert-envelope amplitudes per antenna,
    using a heteroscedastic noise model (noise + smearing).

    Parameters
    ----------
    params : np.ndarray, shape (6,)
        [dx, dy, dz, dE, dtheta, dphi] — shifts relative to base event.
    ctx : EventContext

    Returns
    -------
    llh : float
        Log-likelihood value (includes SWF time term if times_noisy is set).
    """
    values, antenna_pos, xmax_shifted = apply_shift(params, ctx)
    Xs_first = xmax_shifted[0]
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    voltage_traces = predict_voltage(input_arr, kxB_loc, antenna_pos, Xs_first, ctx.model, ctx.fs_ds, ctx.t_SN, ctx.t_EW, ctx.t_Z, ctx.tf)
    pred_amps = get_amps(voltage_traces)

    # Heteroscedastic noise: sigma^2 = noise_var + (smearing * measured_amp)^2
    variance = np.sum(ctx.std_polar ** 2) + (ctx.smearing * ctx.measured_amps) ** 2
    shape_error = np.sum((pred_amps - ctx.measured_amps) ** 2 / (2 * variance))
    llh_shape = -shape_error

    if ctx.times_noisy is not None and ctx.sigma_t is not None:
        # Check for n_eff
        n_effs = None
        if ctx.has_n_eff:
            n_effs = input_arr[:, ctx.n_eff_idx] + 1
        llh_time = compute_swf_llh(Xs_first, ctx, n_effs=n_effs)
        return 2 * (ctx.alpha * llh_shape + (1 - ctx.alpha) * llh_time)
    else:
        return llh_shape



def log_likelihood_pulse_matched_filtering_high_snr(params, ctx, smear=None):
    """
    Log-likelihood using matched filtering for time alignment.
    
    For each antenna, computes the optimal delay that maximizes correlation
    between predicted and measured traces, then computes likelihood based on
    the aligned traces.
    
    Parameters
    ----------
    params : np.ndarray, shape (6,)
        [dx, dy, dz, dE, dtheta, dphi] — shifts relative to base event.
    ctx : EventContext
    smear : np.ndarray or None
        Optional smearing factors for heteroscedastic noise model.
    returns
    -------
    llh : float
        Log-likelihood value (includes SWF time term if times_noisy is set).
    """
    values, antenna_pos, xmax_shifted = apply_shift(params, ctx)
    Xs_first = xmax_shifted[0]
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    voltage_traces_f = predict_voltage(input_arr, 
                                       kxB_loc, 
                                       antenna_pos, 
                                       Xs_first, 
                                       ctx.model, 
                                       ctx.fs_ds, 
                                       ctx.t_SN, 
                                       ctx.t_EW, 
                                       ctx.t_Z,
                                       ctx.tf,
                                       compute_td=False)
    t_swf = event_swf_time(Xs_first, antenna_pos)
    t_swf -= t_swf.mean()

    
    matched_filter = np.sum(ctx.measured_fft_over_psd * np.conjugate(voltage_traces_f), axis=1)
    n_ant, n_freq = matched_filter.shape
    correlations = np.fft.irfft(matched_filter, axis=-1)
    correlations -= 1/2 * (np.abs(voltage_traces_f) ** 2 / ctx.psd).sum(axis=(1,2))[:,None]/(n_freq-1)
    
    log_g=np.log(ctx.jitter_kernel)
    max_idxs = np.argmax(correlations, axis=-1)
    peak_times = ctx.t_bin_0 + max_idxs / ctx.fs_ds
    peak_times -= peak_times.mean()
    
    maxes = np.max(correlations, axis=-1)
    smoothed_maxes = maxes[:,None] + log_g[None,:]
    result = np.empty(3*len(ctx.jitter_kernel))

    shifts = ((peak_times - t_swf)*ctx.fs_ds).astype(int)

    mid_point = len(result) // 2
    for i in range(n_ant):
        first_bin = mid_point + shifts[i] - len(ctx.jitter_kernel) // 2
        last_bin = first_bin + len(ctx.jitter_kernel)
        low_bound = max(0, first_bin)
        up_bound = min(len(result), last_bin)

        smooth_start = max(0, -first_bin)
        smooth_end = min(len(ctx.jitter_kernel), len(ctx.jitter_kernel) - (last_bin - len(result)))
        result[low_bound:up_bound] += smoothed_maxes[i, smooth_start:smooth_end]

    log_probas_total = np.max(result) # If high SNR, can use max instead of logsumexp for numerical stability
    return log_probas_total


def log_likelihood_pulse_matched_filtering_low_snr(params, ctx, smear=None):
    """
    Log-likelihood using matched filtering for time alignment.
    
    For each antenna, computes the optimal delay that maximizes correlation
    between predicted and measured traces, then computes likelihood based on
    the aligned traces.
    
    Parameters
    ----------
    params : np.ndarray, shape (6,)
        [dx, dy, dz, dE, dtheta, dphi] — shifts relative to base event.
    ctx : EventContext
    smear : np.ndarray or None
        Optional smearing factors for heteroscedastic noise model.
    returns
    -------
    llh : float
        Log-likelihood value (includes SWF time term if times_noisy is set).
    """
    values, antenna_pos, xmax_shifted = apply_shift(params, ctx)
    Xs_first = xmax_shifted[0]
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    voltage_traces_f = predict_voltage(input_arr, 
                                       kxB_loc, 
                                       antenna_pos, 
                                       Xs_first, 
                                       ctx.model, 
                                       ctx.fs_ds, 
                                       ctx.t_SN, 
                                       ctx.t_EW, 
                                       ctx.t_Z,
                                       ctx.tf,
                                       compute_td=False)
    t_swf = event_swf_time(Xs_first, antenna_pos)
    t_swf -= t_swf.mean()

    
    matched_filter = np.sum(ctx.measured_fft_over_psd * np.conjugate(voltage_traces_f), axis=1)
    n_ant, n_freq = matched_filter.shape
    correlations = np.fft.irfft(matched_filter, axis=-1)
    correlations -= 1/2 * (np.abs(voltage_traces_f) ** 2 / ctx.psd).sum(axis=(1,2))[:,None]/(n_freq-1)
    
    maxes = np.max(correlations, axis=-1)
    probas = np.exp(correlations - maxes[:, None])
    ## accounting for jitter:
    g=ctx.jitter_kernel
    v = np.maximum(convolve(probas, g[None,:], mode="same"), 1e-300)
    out = maxes[:,None] + np.log(v)

    delta_ts = t_swf - ctx.t_bin_0
    out = np.roll(out, int(delta_ts * ctx.fs_ds), axis=-1)
    aligned_trace = np.sum(out, axis=0)
    # log_probas_total = logsumexp(aligned_trace) If low SNR
    log_probas_total = np.max(aligned_trace) # If high SNR, can use max instead of logsumexp for numerical stability
    return log_probas_total


def log_likelihood_pulse_matched_filtering_high_snr(params, ctx, smear=None):
    """
    Log-likelihood using matched filtering for time alignment.
    
    For each antenna, computes the optimal delay that maximizes correlation
    between predicted and measured traces, then computes likelihood based on
    the aligned traces.
    
    Parameters
    ----------
    params : np.ndarray, shape (6,)
        [dx, dy, dz, dE, dtheta, dphi] — shifts relative to base event.
    ctx : EventContext
    smear : np.ndarray or None
        Optional smearing factors for heteroscedastic noise model.
    returns
    -------
    llh : float
        Log-likelihood value (includes SWF time term if times_noisy is set).
    """
    values, antenna_pos, xmax_shifted = apply_shift(params, ctx)
    Xs_first = xmax_shifted[0]
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    voltage_traces_f = predict_voltage(input_arr, 
                                       kxB_loc, 
                                       antenna_pos, 
                                       Xs_first, 
                                       ctx.model, 
                                       ctx.fs_ds, 
                                       ctx.t_SN, 
                                       ctx.t_EW, 
                                       ctx.t_Z,
                                       ctx.tf,
                                       compute_td=False)
    t_swf = event_swf_time(Xs_first, antenna_pos)
    t_swf -= t_swf.mean()

    
    matched_filter = np.sum(ctx.measured_fft_over_psd * np.conjugate(voltage_traces_f), axis=1)
    n_ant, n_freq = matched_filter.shape
    correlations = np.fft.irfft(matched_filter, axis=-1)
    correlations -= 1/2 * (np.abs(voltage_traces_f) ** 2 / ctx.psd).sum(axis=(1,2))[:,None]/(n_freq-1)
    
    maxes = np.max(correlations, axis=-1)
    peak_idx = np.argmax(correlations, axis=-1)

    probas = np.exp(correlations - maxes[:, None])
    ## accounting for jitter:
    g=ctx.jitter_kernel
    v = np.maximum(convolve(probas, g[None,:], mode="same"), 1e-300)
    out = maxes[:,None] + np.log(v)

    delta_ts = t_swf - ctx.t_bin_0
    out = np.roll(out, int(delta_ts * ctx.fs_ds), axis=-1)
    aligned_trace = np.sum(out, axis=0)
    # log_probas_total = logsumexp(aligned_trace) If low SNR
    log_probas_total = np.max(aligned_trace) # If high SNR, can use max instead of logsumexp for numerical stability
    return log_probas_total



###### Vectorized versions
def compute_swf_llh_vect(X_s, ctx, n_effs=None):
    """
    Compute SWF time log-likelihood using pure numpy.
    """
    # X_s = X_s.reshape(-1, 3)
    llh_swf = np.zeros(X_s.shape[0])
    D_mat = np.sqrt(np.sum( (X_s[:, None, :] - ctx.antenna_pos[None, :, :]) ** 2, axis=-1))
    if n_effs is None:
        n_effs = np.zeros((len(X_s), len(ctx.antenna_pos))) 
        for i in range(len(X_s)):
            n_effs[i] =   ZHSEffectiveRefractionIndexvect(X_s[i], ctx.antenna_pos)

    T_mat = D_mat * n_effs / c
    T_mat -= T_mat.mean(axis=1, keepdims=True)  # Center per candidate
    llh_swf = -np.sum( (T_mat - ctx.times_noisy_centered[None, :]) ** 2 / (2 * ctx.sigma_t ** 2), axis=1)
    return llh_swf

def log_likelihood_pulse_vect(params, ctx, smear=None):
    params = params.reshape(-1, params.shape[-1])
    n_batch = len(params)
    n_dus = len(ctx.antenna_pos)        
    full_input_arrays = np.zeros((n_dus * n_batch, len(ctx.config_inputs)))
    full_kxB_loc = np.zeros((n_batch*n_dus, 3))
    for i in range(n_batch):
        values, _, _ = apply_shift(params[i], ctx)
        input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
        full_input_arrays[i*n_dus:(i+1)*n_dus] = input_arr
        full_kxB_loc[i*n_dus:(i+1)*n_dus] = kxB_loc
    full_du_pos = np.tile(ctx.antenna_pos, (n_batch, 1))
    Xmax_cand = params[:, :3] + np.array([ctx.base_values['xmax_pos_x'][0],
                                         ctx.base_values['xmax_pos_y'][0],
                                         ctx.base_values['xmax_pos_z'][0]])[None,:]
    full_Xmax_cand = np.repeat(Xmax_cand, n_dus, axis=0)    #shape (n_batch*n_dus, 3)
    voltage_traces_predicted = predict_voltage(full_input_arrays, full_kxB_loc, full_du_pos, full_Xmax_cand, ctx.model, ctx.fs_ds, ctx.t_SN, ctx.t_EW, ctx.t_Z, ctx.tf) #shape (n_batch*n_dus, n_pol, n_time)
    if smear is not None:
        voltage_traces_predicted *= (1 + smear.flatten()[:, None, None])
    llh_shape = np.zeros(n_batch)
    for i in range(n_batch):
        llh_shape[i] = compute_shape_llh(voltage_traces_predicted[i*n_dus:(i+1)*n_dus], ctx)
    # Time likelihood (if hybrid)
    full_neffs = full_input_arrays[:, ctx.n_eff_idx] + 1 if ctx.has_n_eff else None
    if ctx.times_noisy is not None and ctx.sigma_t is not None:
        llh_time = compute_swf_llh_vect(Xmax_cand, ctx, n_effs=full_neffs.reshape(n_batch, n_dus))
        return 2 * ctx.alpha * llh_shape + 2 * (1 - ctx.alpha) * llh_time
    else:
        return llh_shape
        


def log_prob_amplitude(params, ctx):
    """
    Full log-posterior for amplitude-based reconstruction (single walker).

    Parameters
    ----------
    params : np.ndarray, shape (6,)
        [dx, dy, dz, dE, dtheta, dphi] — shifts relative to base event.
    ctx : EventContext

    Returns
    -------
    float : log-posterior value
    """
    # Prior (log_prior_informative handles both single and batched inputs)
    if ctx.k_guess is not None and ctx.Xs_guess is not None:
        lp = log_prior_informative(params, ctx)
    else:
        lp = log_prior_vectorized(params)[0]
    if not np.isfinite(lp):
        return -np.inf

    try:
        # Amplitude likelihood (includes SWF time if times_noisy is set)
        llh = log_likelihood_amplitude(params, ctx)
        return lp + llh
    except Exception:
        return -np.inf

def log_prob_pulse(x_batch, ctx, smear=None):
    """
    Vectorized log-probability for emcee.
    
    Evaluates all walkers in a single call.
    The model inference for all walkers could be batched in principle,
    but since each walker has different Xmax shifts (and therefore different 
    response matrices), we still loop over walkers for to_voltage.
    
    However, we eliminate all DataFrame overhead and minimize torch conversions.
    
    Parameters
    ----------
    x_batch : np.ndarray, shape (nwalkers, 6)
    ctx : EventContext
    
    Returns
    -------
    np.ndarray, shape (nwalkers,) — log-probability for each walker
    """
    nwalkers = x_batch.shape[0]
    
    # Vectorized prior check
    if ctx.k_guess is not None and ctx.Xs_guess is not None:
        lp = log_prior_informative(x_batch, ctx)
    else:
        lp = log_prior_vectorized(x_batch)

    # Compute likelihood only for valid walkers
    result = np.full(nwalkers, -np.inf)
    valid_mask = np.isfinite(lp)
    
    ll = log_likelihood_pulse_vect(x_batch[valid_mask], ctx, smear=smear)
    
    result = lp + ll
    
    return result


def log_prob_pulse_smearing(x_batch, ctx):
    recons_x = x_batch[:, :6]
    smearing_x = x_batch[:, 6:]

    ctx_no_smearing = ctx.copy()
    ctx_no_smearing.smearing = 0.0
    log_prob = log_prob_pulse(recons_x, ctx_no_smearing, smear=smearing_x)
    
    smearing_prior = -0.5 * (smearing_x / ctx.smearing) ** 2
    return log_prob + np.sum(smearing_prior, axis=1)


def mcmc_emcee_optimized(x0, ctx, n_steps=10000, seed=None, progress=True, amplitude_based=False):
    """
    Run MCMC using emcee with vectorized log_prob.
    
    Uses vectorize=True in emcee.EnsembleSampler for batch evaluation.
    """
    
    nwalkers = x0.shape[0]
    ndim = x0.shape[-1]
    
    if amplitude_based:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, 
            log_prob_amplitude,
            args=(ctx,),
            vectorize=False  # Key optimization: emcee passes all walkers at once
        )
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, 
            log_prob_pulse,
            args=(ctx,),
            vectorize=True  # Key optimization: emcee passes all walkers at once
        )
    
    sampler.run_mcmc(x0, n_steps, progress=progress)
    samples = sampler.get_chain()
    log_probas = sampler.get_log_prob()
    return samples, log_probas

def mcmc_emcee_smearing(x0, ctx, n_steps=10000, seed=None, progress=True, amplitude_based=False):
    """
    Run MCMC using emcee with vectorized log_prob.
    
    Uses vectorize=True in emcee.EnsembleSampler for batch evaluation.
    """
    
    nwalkers = x0.shape[0]
    ndim = x0.shape[-1]
    
    if amplitude_based:
        raise NotImplementedError("Amplitude-based smearing MCMC not implemented yet.")
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, 
            log_prob_pulse_smearing,
            args=(ctx,),
            vectorize=True  # Key optimization: emcee passes all walkers at once
        )
    
    sampler.run_mcmc(x0, n_steps, progress=progress)
    samples = sampler.get_chain()
    log_probas = sampler.get_log_prob()
    return samples, log_probas



