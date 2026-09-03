# code from Arsène Ferrière


from pathlib import Path
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path as MplPath
from matplotlib.colors import LogNorm
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
import sys

sys.path.append("/sps/grand/cprevotat/grand/DU_response_computation") #to modify
from make_input import load_input_params_from_dict
from noise import compute_noise

sys.path.append("/sps/grand/cprevotat/grand/generative") #to modify
from pipeline.utils import altitude, R2D, sph2cart, Bvec
from pipeline.input_formating import make_input_array
from pipeline.models import MLP_metamodel, MLPClassifierGated
from pipeline.inference import predict_voltage, pred_trace_kxB
from pipeline.emcee_utils import sample_galactic_noise



import joblib

CONFIG = {
    "paths": {
        "base_model_path": "/sps/grand/aferrier/MODELS/generative_models/", #"/volatile/home/af274537/Documents/MODELS/generative_models"
        "model_subdir": "/sps/grand/aferrier/MODELS/generative_models/moriond/moriond_final",
        "params_file": "/sps/grand/aferrier/MODELS/antenna/RF_params_new_leffs.json", #"recons/RF_params_new_leffs.json", #
        "antenna_pos": "/sps/grand/cprevotat/grand/efficiency/gp65_rtksort.txt",
    },
}
#"model_subdir": "moriond/moriond_final",

def load_antenna_positions(path_to_file):
    #antenna_pos_path = Path(cfg["paths"]["antenna_pos"])
    #if antenna_pos_path.exists():
        #return np.load(antenna_pos_path)
    #if Path(path_to_file).exists():

    return np.loadtxt(path_to_file["paths"]["antenna_pos"])[:, 1:4]
    #else:
        #raise RuntimeError("Antenna positions file not found.")

def load_model_and_config(cfg):
    model_dir = Path(cfg["paths"]["base_model_path"]) / cfg["paths"]["model_subdir"]
    state_dict = torch.load(model_dir / "model_last.pth", map_location=torch.device("cpu"))
    with open(model_dir / "config.json", "r") as f:
        model_cfg = json.load(f)

    model = MLP_metamodel(
        inputs=model_cfg["data"]["inputs"],
        n_layers=model_cfg["model"]["n_layers"],
        skip_connection=model_cfg["model"]["skip_connection"],
        hidden_size=model_cfg["model"]["hidden_size"],
        activation=model_cfg["model"]["activation"],
        log_out=model_cfg["model"]["use_log"],
        output_size=len(model_cfg["data"]["outputs"]),
    )
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.device = "cpu"

    clsf = joblib.load(model_dir / "hist_gbdt.joblib")
    with open(model_dir / "hist_gbdt.pkl", "rb") as f:
        clsf = pickle.load(f)
    gated_model = MLPClassifierGated(model, clsf)
    #gated_model = model


    return gated_model, model_cfg

def generate_shower(entries, model, model_cfg, antenna_pos, t_SN, t_EW, t_Z, tf_ds, voltage=False, ADC=True):
    n_ants = antenna_pos.shape[0]
    xmax_pos = np.array([entries["xmax_pos_x"], entries["xmax_pos_y"], entries["xmax_pos_z"]])
    eem = entries["eem"]
    theta = entries["theta"]
    phi = entries["phi"]
    values = {
                "event_number": np.zeros(n_ants),
                "xmax_pos_x": np.ones(n_ants) * xmax_pos[0],
                "xmax_pos_y": np.ones(n_ants) * xmax_pos[1],
                "xmax_pos_z": np.ones(n_ants) * xmax_pos[2],
                "energy_em": np.ones(n_ants) * eem,
                "zenith": np.ones(n_ants) * theta,
                "azimuth": np.ones(n_ants) * phi,
                "du_pos_x": antenna_pos[:, 0],
                "du_pos_y": antenna_pos[:, 1],
                "du_pos_z": antenna_pos[:, 2],
            }
    input_arr, (k, kxB, kxkxB) = make_input_array(values, model_cfg['data']['inputs'], ) 

    k_r = core_pos - xmax_pos
    k_r = k_r/np.linalg.norm(k_r)
    print("k : ", k_r[2], k[2])

    omegas = input_arr[:, model_cfg['data']['inputs'].index("omega")]
    
    valid = model.classifier.predict_proba(input_arr)
    mask = valid[:, 1] > 0.2
    #mask = np.ones(len(input_arr), dtype=bool)




    if voltage or ADC:
        voltage_traces_valid = predict_voltage(input_arr[mask], kxB, antenna_pos[mask], xmax_pos, model.mlp_model, 500, t_SN, t_EW, t_Z, tf_ds, True) #in µV
        voltage_traces = np.zeros((n_ants, 3, voltage_traces_valid.shape[2]))
        voltage_traces[mask] = voltage_traces_valid
        if ADC:
            adc_traces = np.round( voltage_traces / (0.9*1e6) * 8192 ).astype(int)
            return adc_traces
        return voltage_traces
    else:
        pred_kxB_valid, pred_kxB_fft = pred_trace_kxB(model, input_arr, fs=500)  
        if kxB.ndim == 1:
            preds_3d_valid = pred_kxB_valid[:, None, :] * kxB[None, :, None]
        else:
            preds_3d_valid = pred_kxB_valid[:, None, :] * kxB[:, :, None]
        preds_3d = np.zeros((n_ants, 3, pred_kxB_valid.shape[1]), dtype=complex)
        preds_3d[mask] = preds_3d_valid
        return preds_3d
    
def main(thetas, phis, eems, xmax_poss, antenna_pos=None):
    model, model_cfg = load_model_and_config(CONFIG)
    if antenna_pos is None:
        antenna_pos = load_antenna_positions(CONFIG)


    with open(CONFIG["paths"]["params_file"], "r") as f:
        params_rf = json.load(f)
    (_,
     latitude,
     _,
     _input_sampling_freq,
     _out_sampling_freq,
     _n_samples,_sampling_period,_freqs,
     _out_n_samples,_out_sampling_period,_out_freqs,
     _lst_radians,
     tf,t_sn,t_ew,t_z,
    ) = load_input_params_from_dict(params_rf)
    ratio_fs = 4
    tf_ds = tf[..., : int(tf.shape[-1] / ratio_fs) + 1]
    all_traces = []

    for xmax_pos, theta, phi, eem in zip(xmax_poss, thetas, phis, eems):
        inputs = {
            "xmax_pos_x": xmax_pos[0],
            "xmax_pos_y": xmax_pos[1],
            "xmax_pos_z": xmax_pos[2],
            "eem": eem,
            "theta": theta,
            "phi": phi,
        }
        trace = generate_shower(inputs, model, model_cfg, antenna_pos, t_sn, t_ew, t_z, tf_ds, voltage=True, ADC=True)
        all_traces.append(trace)
    return all_traces

if __name__ == "__main__":
    from pipeline.opening_root import _get_all_event_numbers, get_event_properties, get_event_traces, open_event_root
    #BASE_ROOT = "/sps/grand/DC2_Coreas/Coreas_nonoise"
    #root_dir = f"{BASE_ROOT}/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_0000"
    #all_event_numbers = _get_all_event_numbers(root_dir)
    BASE_ROOT = "/sps/grand/DC2.1rc4/GP300ZHAireS-NJ"
    root_dir = f"{BASE_ROOT}/sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-NJ_0011"
    all_event_numbers = _get_all_event_numbers(root_dir)

    """
    for event_number in all_event_numbers[1800:]:
        antenna_pos, properties = get_event_properties(root_dir, event_number)
        print(f"Event number {event_number}, the energy is : ", properties["energy_primary"], np.log10(properties["energy_em"]), properties["zenith"])
    """

    #print(all_event_numbers)
    idx = 501 # 2 is not working fine
    event_number = all_event_numbers[idx] #3728 for Zhaires, with file number 11
    antenna_pos, properties = get_event_properties(root_dir, event_number)
    while properties["energy_primary"] < 1e9:
        idx += 1
        event_number = all_event_numbers[idx] #3728 for Zhaires, with file number 11
        antenna_pos, properties = get_event_properties(root_dir, event_number)
    print("our event number is : ", event_number, "idx is : ", idx)
    all_traces, all_trace_voltage, du_ids, theta, azimuth = get_event_traces(root_dir, event_number, voltage=True)
    print(properties.keys())
    print("we are dealing with a particle which is a ", properties["p_type"])
    print("The energy is : ", np.log10(properties["energy_primary"]), np.log10(properties["energy_em"]), properties["zenith"])
    core_pos = properties["core_pos"] + np.array([0, 0, 1264])

    #print(antenna_pos + np.array([0, 0, 1264]))
    #antenna_pos = None

    generated_trace = main(
        thetas= [properties["zenith"]],
        phis= [properties["azimuth"]],
        eems= [properties["energy_em"]],
        xmax_poss= [properties["xmax_pos"] + np.array([0, 0, 1264])],
        antenna_pos=antenna_pos + np.array([0, 0, 1264])
    )
    # +np.array([0, 0, 1264])
    #print(np.shape(antenna_pos))
    #print(antenna_pos)
    print("The parameters are : ", properties["zenith"], properties["azimuth"], properties["energy_em"], properties["xmax_pos"] + np.array([0, 0, 1264]), properties["du_id"])

    t = np.arange(generated_trace[0].shape[-1]) * (1.0 / 500) * 1e3
    t_coreas = t #- 250
    print("times : ", t)

    # let’s compare the Zhaires and Coreas simulations : 
    # first let’s have a look at X_max as a function of energy and zenith angle for protons
    # then let’s also look at the ratio E_primary / E_em

    """
    # plot the fft of the true trace :
    plt.figure()
    plt.plot(np.fft.rfftfreq(np.shape(generated_trace[0][0, 0, :])[0], 1/500), np.abs(np.fft.rfft(generated_trace[0][0, 0, :])) , label='FFT of generated trace (kxB)')
    plt.plot(np.fft.rfftfreq(np.shape(all_trace_voltage[0, 0, :])[0], 1/500), np.abs(np.fft.rfft(all_trace_voltage[0, 0, :])) , label='FFT of true trace (kxB)')
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Amplitude")
    plt.title("FFT of generated and true traces in frequency domain")
    plt.legend()
    
    """
    for i in range(0, np.shape(generated_trace[0])[0]):
        print(i)
        if generated_trace[0][i, 0, :].max() < 50:
            continue
        plt.figure()
        plt.plot(t[:], generated_trace[0][i, 0, :], label='interp X') 
        plt.plot(t_coreas, all_trace_voltage[i, 0, :], label='True X', ls = "--") 
        plt.plot(t[:], generated_trace[0][i, 1, :], label='interp Y')
        plt.plot(t_coreas, all_trace_voltage[i, 1, :], label='True Y', ls = "--")
        plt.plot(t[:], generated_trace[0][i, 2, :], label='interp Z')
        plt.plot(t_coreas, all_trace_voltage[i, 2, :], label='True Z', ls = "--")
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude (µV)")
        plt.legend()
        plt.show()

    
    

    fig, axs = plt.subplots(3,3, figsize=(12, 8))
    axs = axs.flatten()
    print(np.shape(generated_trace))
    print(generated_trace[0].shape) # shape is N_ants * 3 * N_samples
    for i,ax in enumerate(axs):
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (µV)")
        ax.plot(t[:], generated_trace[0][i, 0, :], label='voltage ADC') #i+4
        ax.plot(t[:], all_trace_voltage[i, 0, :], label='True voltage ADC') #i+4
        ax.legend()
    plt.show()

    fig, axs = plt.subplots(2,2, figsize=(12, 8))
    axs = axs.flatten()
    for i,ax in enumerate(axs):
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (µV)")
        ax.plot(t[:], np.abs((generated_trace[0][i, 0, :] - all_trace_voltage[i, 0, :])), label='Difference') #i+4
        ax.legend()
    plt.show()
    
