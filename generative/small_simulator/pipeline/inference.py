import numpy as np
import torch
from apply_rfchain import percieved_theta_phi, efield_2_voltage, make_full_response_matrix, voltage_to_adc

from .utils import trace_strong_filtering 

class filters_tf():
    def __init__(self, filters, duration=2.048, fs=2000):
        self._tf = None
        self.filters = filters
        self.duration = duration
        self.fs = fs

    @property
    def tf(self):
        if self._tf is None:
            self._tf = self.create_tf()
        return self._tf

    @tf.setter
    def tf(self, value):
        self._tf = value

    def create_tf(self):
        dirac = torch.zeros(int(self.fs*self.duration))
        dirac[0] = 1.0
        for f in self.filters:
            dirac = f(dirac, fs=self.fs)
        fft_dirac = np.fft.rfft(dirac)
        return fft_dirac

def make_filter(fs_f: float, duration: float, low: float, high: float):
    """Create frequency domain filter."""
    f = filters_tf([lambda x, fs: trace_strong_filtering(x, low, high, fs)], fs=fs_f, duration=duration).tf
    return f
    
@torch.no_grad()
def get_preds(model, params):
    """
    Optimized inference-only version of get_preds.
    
    Assumes model.eval() and requires_grad=False have already been set.
    Wraps everything in torch.no_grad() for speed.
    """
    if hasattr(model, "predict_gated"):
        return model.predict_gated(params)

    raw_params = params
    if isinstance(params, np.ndarray):
        params = torch.tensor(params, dtype=torch.float32, device=model.device)
    elif params.device != model.device:
        params = params.to(model.device)
    preds = model(params)
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = model.normalizer.inverse(preds, outputs=True)
    preds = preds.cpu().numpy()
    if hasattr(model, "postprocess_outputs"):
        preds = model.postprocess_outputs(raw_params, preds)
    return preds

def abc_to_fourrier(a, b, c, freqs, f_0=30):
    assert type(freqs) is np.ndarray
    if isinstance(a, (np.floating, float)):
        a = np.array([a])
        b = np.array([b])
        c = np.array([c])
    clipped = 250.
    minimums = - b/(2 * c + 1e-15) + f_0
    minimums[c<0] = clipped
    freqs_clip = np.minimum(freqs[None,:], minimums[:, None])
    exponant = a[:, None] + b[:, None]*(freqs_clip - f_0) + c[:, None]*(freqs_clip - f_0)**2
    exponant = np.clip(exponant, -700, 700)  # Avoid overflow in exp
    amp = np.exp(exponant)
    return amp


def params_to_phase(phase_q, phase_p, phase_offset, freqs, f_1=30):
    assert type(freqs) is np.ndarray
    if isinstance(phase_q, (np.floating, float)):
        phase_q = np.array([phase_q])
        phase_p = np.array([phase_p])
        phase_offset = np.array([phase_offset])
    phase = phase_p[:, None]*(freqs[None,:] - f_1) + phase_q[:, None]*(freqs[None,:] - f_1)**2
    phase = phase + (phase_offset[:, None] - phase[:, [0]])
    return phase


def pred_trace_kxB(model, params, fs=2e3, duration=2.048, f_0=30, fourier_filter=1.0, phase=None):
    """
    Predict the time-domain signal from the model parameters.
    This function is optimized for inference speed. It avoids unnecessary conversions and function calls.
    It assumes that the model is already in eval mode and that requires_grad=False has been set.
    inputs:
        - model: the trained model to use for prediction
        - params: the input parameters for the model (numpy array/torch tensor)
        - fs: sampling frequency in MHz (default 2000 MHz)
        - duration: duration of the output signal in ms (default 2.048 ms)
        - f_0: reference frequency in MHz for the model's predictions (default 30 MHz)
        - fourier_filter: optional frequency domain filter to apply to the predicted Fourier coefficients (default 1.0, i.e. no filtering)
        - phase: optional pre-computed phase array to use instead of computing it from the model's predictions. 
                 If None, the phase will be computed from the model's predictions. (default None)
    Outputs:
        - signal: the predicted time-domain signal (numpy array)
        - fourrier_signal: the predicted Fourier signal (numpy array)
    """
    n_freq = int(fs * duration / 2) + 1
    n_time = int(fs * duration)
    freqs = np.linspace(0, fs / 2, n_freq)
    
    preds = get_preds(model, params)
    a, b, c = preds[:, 0], preds[:, 1], preds[:, 2]
    
    if preds.shape[1] <= 3:
        raise ValueError("Phase must be provided or model must predict phase parameters.")
    if phase is None:
        phase_q, phase_p = preds[:, 3], preds[:, 4]
        phase_offset = preds[:, 5] if preds.shape[1] > 5 else np.zeros_like(phase_q) + np.pi
        
        # Inline params_to_phase (avoid function call overhead)
        phase = phase_p[:, None] * (freqs[None, :] - f_0) + phase_q[:, None] * (freqs[None, :] - f_0) ** 2
        phase = phase + (phase_offset[:, None] - phase[:, [0]])
    
    # Inline abc_to_fourrier (avoid function call overhead)
    clipped = 250.0
    minimums = -b / (2 * c + 1e-15) + f_0
    minimums[c < 0] = clipped
    freqs_clip = np.minimum(freqs[None, :], minimums[:, None])
    exponant = a[:, None] + b[:, None] * (freqs_clip - f_0) + c[:, None] * (freqs_clip - f_0) ** 2
    exponant = np.clip(exponant, -700, 700)  # Avoid overflow in exp
    amps = np.exp(exponant)
    
    fourrier_signal = amps * np.exp(1j * phase) * fourier_filter * (fs / 2e3)
    signal = np.fft.irfft(fourrier_signal, n=n_time)
    return signal, fourrier_signal
    

def predict_voltage(input_arr, kxB, antenna_pos, Xs, model, fs_ds, t_SN, t_EW, t_Z, tf, muV=False, compute_td=True):
    """
    Optimized voltage prediction: model inference → E-field → voltage.
    
    This is the full pipeline equivalent of predict_voltage() in emcee_utils.py:
      1. Run the model to get the kxB E-field trace
      2. Expand to 3D via kxB polarization
      3. Apply the RF chain (antenna response) to convert E-field → voltage
    
    Key optimizations:
    - Uses pred_trace_kxB (torch.no_grad, single np↔torch conversion)
    - torch.compile on model forward pass (if available)
    - float32 throughout the torch path
    
    Parameters
    ----------
    input_arr : np.ndarray, shape (n_ant, n_features)
        Input features for the model.
    kxB : np.ndarray, shape (3,) or (n_ant, 3)
        Polarization direction(s) for the kxB component.
    antenna_pos : np.ndarray, shape (n_ant, 3)
        Antenna positions.
    Xs : np.ndarray, shape (3,)
        Shifted Xmax position (single source seen by all antennas).
    ctx : EventContext
        Precomputed event context.
    
    Returns
    -------
    preds_voltage : np.ndarray, shape (n_ant, 3, n_samples)
        Predicted voltage traces after RF chain application.
    """
    # 1. Model inference: input features → kxB E-field trace
    pred_kxB, pred_kxB_fft = pred_trace_kxB(model, input_arr, fs=fs_ds)
    

    # modified here to filter the E-field traces :
    pred_kxB_filtered = trace_strong_filtering(pred_kxB, 50, 200, fs = fs_ds)
    pred_kxB_fft_filtered = np.fft.rfft(pred_kxB_filtered, axis=-1)

    #plot the filtured trace as a function of frequency
    """
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(np.fft.rfftfreq(pred_kxB.shape[-1], 1/fs_ds), np.abs(pred_kxB_fft_filtered[0]), label='Filtered E-field trace (kxB)')
    plt.plot(np.fft.rfftfreq(pred_kxB.shape[-1], 1/fs_ds), np.abs(pred_kxB_fft[0]), label='Unfiltered E-field trace (kxB)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Amplitude')
    plt.title('Filtered E-field trace in frequency domain')
    plt.legend()
    plt.figure()
    """

    pred_kxB_fft_filtered = pred_kxB_fft
    
    # 2. Expand to 3D E-field via kxB polarization
    if kxB.ndim == 1:
        # preds_3d = pred_kxB[:, None, :] * kxB[None, :, None]
        preds_3d_fft = pred_kxB_fft_filtered[:, None, :] * kxB[None, :, None]
    else:
        # preds_3d = pred_kxB[:, None, :] * kxB[:, :, None]
        preds_3d_fft = pred_kxB_fft_filtered[:, None, :] * kxB[:, :, None]

    # 3. E-field → Voltage via RF chain (response matrix depends on viewing angles)
    vout, vout_f = to_voltage(preds_3d_fft, 
                              antenna_pos, Xs, 
                              fs_ds, fs_ds, 
                              t_SN, t_EW, t_Z, tf, 
                              duration=2.048, to_adc=False, is_fourier=True, compute_td=compute_td)
    
    
    if compute_td:
        if not muV:
            return vout/1e3
        return vout
    else:
        if not muV:
            return vout_f/1e3
        return vout_f
    

def to_voltage(efield, 
               antenna_pos, Xs, 
               fs_input, fs_output, 
               t_SN, t_EW, t_Z, tf, duration=2.048, 
               to_adc=False, is_fourier=False, compute_td=True):
    theta_du, phi_du = percieved_theta_phi(antenna_pos, Xs)
    if is_fourier:
        efield_fft = efield
    else:
        efield_fft = np.fft.rfft(efield, axis=-1)
    full_response_matrix = make_full_response_matrix(t_SN, t_EW, t_Z, theta_du, phi_du, tf, input_sampling_freq=fs_input*1e6, duration=duration*1e-6)
    vout, vout_f = efield_2_voltage(efield_fft, 
                                    full_response_matrix, 
                                    current_rate=fs_input*1e6, target_rate=fs_output*1e6,
                                    compute_td=compute_td)
    
    if to_adc and compute_td:
        vout = voltage_to_adc(vout)
        vout_f = np.fft.rfft(vout)
        return vout, vout_f
    if not compute_td:
        return None, vout_f
    return vout, vout_f
    