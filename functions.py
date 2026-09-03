# code for evaluating the efficiency of GP80 (or 65 for all I know)

import grand.dataio.data_handling as dh
import grand.dataio.data_tree as dt
import grand.dataio.run_trees as rt
import grand.dataio.event_trees as et

import sys
import os
sys.path.append(os.path.abspath("/sps/grand/cprevotat/grand/grand/grand/exposure/")) # this is not very clean, but it was not to mess with Sei’s code, and not copy everything here
from utils import calculate_PAO_spectrum, calculate_relative_trace_start_time, get_FLT0_trigger_parameters_du_level, get_FLT0_trigger_parameters, get_FLT0_trigger_time, notch_filter, filter_traces_bandpass, running_DUs, integrate_PAO_spectrum

sys.path.append(os.path.abspath("/sps/grand/cprevotat/grand/efficiency/")) # this is not very clean, but it was not to mess with Sei’s code, and not copy everything here
from dict_th1_th2 import data_dict

import offline_FLT0_trigger as FLT0

#from grand.grandlib_classes.grandlib_classes import * # it’s not in the version of my tag


import numpy as np
import pandas as pd
import random
from scipy.integrate import trapezoid as trpz
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from glob import glob
import os.path
import matplotlib.pyplot as plt
import warnings
from scipy.signal import hilbert
# % matplotlib ipypml

from grand import (
    Coordinates,
    CartesianRepresentation,
    SphericalRepresentation,
    GeodeticRepresentation,
)
from grand import ECEF, Geodetic, GRANDCS, LTP
from grand import Geomagnet
from grand import Topography, Reference, geoid_undulation
from grand import topography

from pathlib import Path
import ROOT
import gc
import json
import cProfile

import matplotlib as mpl

import struct
from datetime import datetime, timezone

from _event_pair import process_signals, calculate_pair_differences

import adapted_CausalityCut_Kwen

import grand.analysis.fitting.adf as adf
print(adf)
import grand.analysis.signals.extraction as ext
import grand.analysis.constants as cons
import grand.analysis.fitting.spherical as swf
import grand.analysis.geom.angles as an


np.infty = np.inf

def get_nice_plots():

    plt.rcParams.update({
        # General
        "figure.figsize": (8,6),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 17,            # Base font size for labels
        "axes.titlesize": 17,       # Plot title
        "axes.labelsize": 17,       # Axis labels
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "font.family": "serif",
        
        # Lines
        "lines.linewidth": 2,
        "lines.markersize": 6,
        
        # Grid
        "axes.grid": False,
        "grid.linestyle": "--",
        "grid.color": "gray",
        "grid.alpha": 0.4,
        
        # Tick settings
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        
        # Legends
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        
        # Axes
        "axes.spines.top": True,
        "axes.spines.right": True,
        
        # Mathtext (for units in labels)
        "mathtext.fontset": "dejavuserif",
    })

    return

get_nice_plots()

def set_plot_style():
    mpl.rcParams.update({
        # --- Figure ---
        "figure.figsize": (6.5, 4.5),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "figure.autolayout": True,

        # --- Fonts ---
        "font.size": 14,
        "font.family": "serif",
        "font.serif": [ "DejaVu Serif"],

        # --- Axes ---
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "axes.axisbelow": True,

        # --- Ticks ---
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        # --- Lines ---
        "lines.linewidth": 2,
        "lines.markersize": 6,

        # --- Legend ---
        "legend.fontsize": 12,
        "legend.frameon": False,

        # --- Grid ---
        "grid.linestyle": "--",
        "grid.alpha": 0.5,

        # --- Math text ---
        "mathtext.fontset": "cm",
        "text.usetex": False,  # set True if you want LaTeX rendering

        # --- Errorbar caps ---
        "errorbar.capsize": 3,
    })
set_plot_style()



# it seems 2212 is used to represent protons, I guess the rest should be Iron




def look_simulation_trace():
    sim_number = 84

    data_directory = f"/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_{sim_number:04d}" # ensure correct formating 

    with open(f"./efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_{sim_number:04d}.json", "r") as f:
        first_judge_data = json.load(f)


    run_numbers = [row["fixed"][0] for row in first_judge_data] # select the run numbers of the events that passed the first trigger
    event_numbers = [row["fixed"][1] for row in first_judge_data] 


    ### Read GRAND root data
    d_input = dh.DataDirectory(data_directory)
    trun_l1, tadc_l1, tshower_l0 = d_input.trun_l1, d_input.tadc_l1, d_input.tshower_l0

    previous_run = None


    f_sample = 500e6 # Hz, ADC sampling rate
    t_res = int(1. / f_sample * 1.e9) # ns, ADC time resolution

    iteration = 0

    event_list = tadc_l1.get_list_of_events()
    nb_events  = len(event_list)


    for event_number,run_number in zip(event_numbers, run_numbers):
        if iteration < 24:
            iteration += 1
            continue
        print("Iteration ", iteration, "out of ", len(event_numbers))

        tadc_l1.get_event(event_number, run_number)
        tshower_l0.get_event(event_number, run_number)

        du_id = np.array(tadc_l1.du_id)

        if previous_run != run_number:
            trun_l1.get_run(run_number)
            previous_run = run_number
                
        for du_id_n in range(du_id.shape[0]):
            print(du_id_n)

            #if T1_idx[du_id_n] > 524: #we are supposed to have at least 500ns (ie Tperiod) after the first T1 crossign # I need to investigate that (why do I have a first T1 crossign at such high value ?)
                #continue

            #print("I came here")

            tadc_trace = np.array(tadc_l1.trace_ch) 
            #print(channels_number, du_id_n, du_ids)
            tadc_trace = tadc_trace[du_id_n][1]

        iteration += 1


    return


def look_simulation_trace_fft():
    sim_number = 84

    data_directory = f"/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_{sim_number:04d}" # ensure correct formating 

    with open(f"./efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_{sim_number:04d}.json", "r") as f:
        first_judge_data = json.load(f)


    run_numbers = [row["fixed"][0] for row in first_judge_data] # select the run numbers of the events that passed the first trigger
    event_numbers = [row["fixed"][1] for row in first_judge_data] 


    ### Read GRAND root data
    d_input = dh.DataDirectory(data_directory)
    trun_l1, tadc_l1, tshower_l0 = d_input.trun_l1, d_input.tadc_l1, d_input.tshower_l0

    previous_run = None


    f_sample = 500e6 # Hz, ADC sampling rate
    t_res = int(1. / f_sample * 1.e9) # ns, ADC time resolution

    iteration = 0

    event_list = tadc_l1.get_list_of_events()
    nb_events  = len(event_list)


    for event_number,run_number in zip(event_numbers, run_numbers):
        if iteration < 24:
            iteration += 1
            continue
        print("Iteration ", iteration, "out of ", len(event_numbers))

        tadc_l1.get_event(event_number, run_number)
        tshower_l0.get_event(event_number, run_number)

        du_id = np.array(tadc_l1.du_id)

        if previous_run != run_number:
            trun_l1.get_run(run_number)
            previous_run = run_number
                
        for du_id_n in range(du_id.shape[0]):
            print(du_id_n)

            #if T1_idx[du_id_n] > 524: #we are supposed to have at least 500ns (ie Tperiod) after the first T1 crossign # I need to investigate that (why do I have a first T1 crossign at such high value ?)
                #continue

            #print("I came here")

            tadc_trace = np.array(tadc_l1.trace_ch) 
            #print(channels_number, du_id_n, du_ids)
            tadc_trace = tadc_trace[du_id_n][1]

        iteration += 1


    return


#look_simulation_trace()
#sys.exit()
                        




def compare_antenna_numbers():
    sim_name = "/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_0000"

    path = "/sps/grand/data/gp80/GrandRoot/2026/03/"
    list_files = [str(f) for f in Path(path).rglob("*CD*.root") if f.is_file()]
    list_files = list_files[:1]
    print(list_files[0])

    d_input = dh.DataDirectory(sim_name)
    trun_l1, tadc_l1, tshower_l0 = d_input.trun_l1, d_input.tadc_l1, d_input.tshower_l0
    print(trun_l1)

    event_list = tadc_l1.get_list_of_events()
    print(event_list, np.shape(event_list))
    nb_events = len(event_list)

    # for not sim data now
    feb_id = []
    rtk_positions = []

    
    for data_name in list_files:
        file_root = dh.DataFile(data_name)
        trun = file_root.trun
        tadc = file_root.tadc
        run_number = tadc.get_list_of_events()[0][1]
        trun.get_run(run_number)
        du_id = trun.du_id
        for i, du in enumerate(du_id):
            if du not in feb_id:
                feb_id.append(du)
                rtk_positions.append(trun.du_xyz[i])
    print("trun")
    print(len(feb_id), len(rtk_positions))
    print(feb_id)
    print(trun.du_id)
    print(trun.du_feb)
    print(tadc.du_id)
    print(rtk_positions)
    

    
    #rtk_positions = np.genfromtxt("./efficiency/gp65_rtksort_Marion.txt").T

    #feb_id = rtk_positions[0]


    for event_number, run_number in event_list:

        trun_l1.get_run(run_number)
        print(trun_l1.du_id)

        liste_DUs = trun_l1.du_id
        position_DUs = trun_l1.du_xyz
        break


    print(liste_DUs)
    print(position_DUs)
    #array_dus = [124, 94, 98, 85, 57, 39, 43, 56, 47, 31, 19, 27, 36, 46, 37, 23, 15, 7, 14, 30, 42, 74, 48, 24, 11, 1, 6, 10, 22, 54, 
    #32, 16, 2, 0, 4, 18, 34, 82, 52, 20, 8, 5, 3, 12, 26, 50, 40, 28, 13, 9, 17, 25, 38, 44, 35, 29, 21, 33, 49, 45, 41, 53, 120, 106, 88, 78] # selected to match the layout
    array_dus = [i for i in range(0, 300)]
    plt.figure()
    for i in range(len(liste_DUs)):
        if liste_DUs[i] in array_dus:
            plt.scatter( -position_DUs[i][1], position_DUs[i][0], s = 5, color = "blue")
            plt.annotate(liste_DUs[i], (-position_DUs[i][1], position_DUs[i][0]), fontsize = 14)

    plt.scatter(-(rtk_positions[2]+266), (rtk_positions[1]+523), s = 5, color = "red", alpha = 0.6)
    #for i in range(len(feb_id)):
        #plt.annotate(feb_id[i], (rtk_positions[1][i]+523, rtk_positions[2][i]), fontsize = 14, alpha = 0.6)

    plt.xlabel("x [m]", fontsize = 16)
    plt.ylabel("y [m]", fontsize = 16)
    #plt.ylim((-2200, 5000)) #2200, 5000
    #plt.xlim(-4400, 2400)
    #plt.ylim((1.1*np.min(rtk_positions[2])), 1.1*(np.max(rtk_positions[2])))
    #plt.xlim((1.1*np.min(rtk_positions[1])), 1.1*(np.max(rtk_positions[1])))
    plt.show()

    return

def generate_correspondance_febID_to_duID():
    data = np.loadtxt("/sps/grand/cprevotat/grand/efficiency/gp65_correspondance_ID_febID.txt").T
    data[0] -= 65000 # so that it starts at 1
    dict = {}
    for du_id, feb_id in zip(data[0], data[1]):
        dict[int(feb_id)] = int(convert_onsite_to_sim_du_numbers(du_id))

    return dict

def generate_correspondance_duID_to_febID():
    data = np.loadtxt("/sps/grand/cprevotat/grand/efficiency/gp65_correspondance_ID_febID.txt").T
    data[0] -= 65000 # so that it starts at 1
    dict = {}
    for du_id, feb_id in zip(data[0], data[1]):
        dict[int(convert_onsite_to_sim_du_numbers(du_id))] = int(feb_id) # we use du numbers from the sim, not the ones on site (since we use those functions for simulations (Correas))

    #print("dict duID to febID : ", dict)
    return dict
    


def convert_onsite_to_sim_du_numbers(sim_numbers):
    return sim_numbers - 1 # maybe there is an exception, but in general it should be that 

def convert_sim_du_numbers_to_onsite(sim_numbers):
    return sim_numbers + 1 # maybe there is an exception, but in general it should be that

dict_duID_to_febID = generate_correspondance_duID_to_febID()
dict_febID_to_duID = generate_correspondance_febID_to_duID()


def extract_infos(path_to_file):

    """Read a root file and extract the info we need, such as ID of the triggering units, energy, position

    """

    d_input = dh.DataDirectory(path_to_file) # read the root file ?
    #d_input.print()

    tshower_l0=d_input.tshower_l0
    tEfield = d_input.tefield_l1
    #tEfield.print()

    tEfield_l1 = d_input.tefield_l1
    #tEfield_l1.print()

    tadc_l1 = d_input.tadc_l1
    events_list = tadc_l1.get_list_of_events()
    #print("shape event list : ", np.shape(events_list))
    nb_events = len(events_list)


    tsimshower_l0 = d_input.tshowersim_l0
    #tshower_l0.print()
    #events_list = tshowersim_l0.get_list_of_events()


    if nb_events == 0:
        sys.exit("There are no events in the file! Exiting.")

    output = open("./grand/efficiency/out_extract_infos/out_"+path_to_file[path_to_file.find('sim'):] + ".txt",'w')

    output.write("#event_number primary_type du_ids energy_primary em_energy Xmax_pos(x, y, z), Xmax_grams zenith azimuth shower_core_pos_x shower_core_pos_y shower_core_pos_z, N_tested_cores, units are degrees, GeV and meters\n") # header of the output file

    for event_number,run_number in events_list:

        assert isinstance(event_number, int)
        assert isinstance(run_number, int)
            
        tadc_l1.get_event(event_number, run_number)
        tshower_l0.get_event(event_number, run_number)
        tEfield_l1.get_event(event_number, run_number)    
        tsimshower_l0.get_event(event_number, run_number) 

        #print(tEfield_l1.du_seconds)
        #print(tEfield_l1.t_pre)
        #print(tEfield_l1.t_post)
        #print(tEfield_l1.du_nanoseconds)   
        
        N_tested_cores = tsimshower_l0.tested_cores # tested : rejected, doesn’t include the one that passed the cuts
        #print(N_tested_cores)
        #if np.shape(N_tested_cores)[0] != 0:
            #print(np.shape(N_tested_cores)[0])
            #print("N_tested_cores : ", N_tested_cores)


        du_id = np.array(tadc_l1.du_id).astype(str)
        du_id = "_".join(du_id)


        #particle_ID = tshower_l0.primary_type
        #if particle_ID != 2212:
            #particle_ID = 1000260560 # ie it’s iron 

        #print(tsimshower_l0.event_weight, np.shape(N_tested_cores)[0])
        
        output.write(str(event_number)+' ')
        output.write('{:9s}'.format(tshower_l0.primary_type)+' ')
        output.write(du_id+' ')
        output.write('{:9.2e}'.format(tshower_l0.energy_primary)+' ')
        output.write('{:9.2e}'.format(tshower_l0.energy_em)+' ')
        output.write('{:9.2f}'.format(tshower_l0.xmax_pos_shc[0])+' ')
        output.write('{:9.2f}'.format(tshower_l0.xmax_pos_shc[1])+' ')
        output.write('{:9.2f}'.format(tshower_l0.xmax_pos_shc[2])+' ')
        output.write('{:9.2f}'.format(tshower_l0.xmax_grams)+' ')
        output.write('{:9.2f}'.format(tshower_l0.zenith)+' ')
        output.write('{:9.2f}'.format(tshower_l0.azimuth)+' ')
        output.write('{:9.2f}'.format(tshower_l0.shower_core_pos[0])+' ')
        output.write('{:9.2f}'.format(tshower_l0.shower_core_pos[1])+' ')
        output.write('{:9.2f}'.format(tshower_l0.shower_core_pos[2])+" ")
        output.write(str(np.shape(N_tested_cores)[0]) + "\n")
        

    output.close()

    d_input.close()
    tadc_l1.close_file()
    tshower_l0.close_file()
    tEfield_l1.close_file()
    tEfield.close_file()

    return


def compare_antenna_positions():

    #sim_name = "/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_0000"
    sim_name = "/sps/grand/DC2.1rc4/GP300ZHAireS-AN/sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-AN_0001"
    rtk_positions = np.genfromtxt("./efficiency/gp65_rtksort.txt").T
    for i in range(len(rtk_positions[0])):
        if rtk_positions[0][i] in dict_febID_to_duID:
            rtk_positions[0][i] = dict_febID_to_duID[rtk_positions[0][i]] # so that it starts at 0 as in the simulations
        else:
            print("Warning: feb_id ", rtk_positions[0][i], " not found in the dictionary. It will be ignored.")
            rtk_positions[0][i] = -1  # Mark as invalid

    print("rtk_positions : ", rtk_positions[0])

    d_input = dh.DataDirectory(sim_name)
    d_input.print()
    trun_l1, tadc_l1, tshower_l0 = d_input.trun_l1, d_input.tadc_l1, d_input.tshower_l0
    print(trun_l1)

    event_list = tadc_l1.get_list_of_events()
    #print(event_list, np.shape(event_list))
    nb_events = len(event_list)

    for event_number, run_number in event_list:

        trun_l1.get_run(run_number)
        print(trun_l1.du_id)

        liste_DUs = trun_l1.du_id
        position_DUs = trun_l1.du_xyz
        break

    print("liste DUs : ", liste_DUs)
    #print(position_DUs)
    #array_dus = [124, 94, 98, 85, 57, 39, 43, 56, 47, 31, 19, 27, 36, 46, 37, 23, 15, 7, 14, 30, 42, 74, 48, 24, 11, 1, 6, 10, 22, 54, 
    #32, 16, 2, 0, 4, 18, 34, 82, 52, 20, 8, 5, 3, 12, 26, 50, 40, 28, 13, 9, 17, 25, 38, 44, 35, 29, 21, 33, 49, 45, 41, 53, 120, 106, 88, 78] # selected to match the layout
    array_dus = [i for i in range(0, 300)]
    plt.figure()
    idx = 0
    for i in range(len(liste_DUs)):
        if liste_DUs[i] in array_dus:
            plt.scatter( -position_DUs[i][1], position_DUs[i][0], s = 5, color = "blue")
            plt.annotate(liste_DUs[i], (-position_DUs[i][1], position_DUs[i][0]), fontsize = 14)

            
        if liste_DUs[i] in rtk_positions[0]:
            plt.scatter((rtk_positions[1][idx]+266), (rtk_positions[2][idx]+523), s = 5, color = "red", alpha = 0.6)
            plt.annotate(liste_DUs[i], ((rtk_positions[1][idx]+216), (rtk_positions[2][idx]+523)), fontsize = 14, alpha = 0.6) # this code works like that 
            idx += 1
    print("plotted")
    plt.xlabel("x [m]", fontsize = 16)
    plt.ylabel("y [m]", fontsize = 16)
    plt.ylim((-2200, 5000)) #2200, 5000
    plt.xlim(-4400, 2400)
    plt.show()

    return
        


def define_allowed_range_t0():

    """
    save somewhere the times at which the detector triggered something, ie was operational, so that then we can draw t0 among those allowed times
    I don’t  know why, but memory used by this function keeps increasing slowly (order 0.2 Mb per file, and it starts around 700 Mb), so that it’s probably better to run it several
    times, once for each month... ?
    """


    directory = "/sps/grand/data/gp80/GrandRoot/"
    years = ["2024/", "2025/", "2026/"]
    years = ["2026/"] # to keep things short
    months = {"2024/": ["11/", "12/"],
            "2027/": ["01/", "02/", "03/", "04/", "05/", "06/", "07/", "08/", "09/", "10/", "11/", "12/"],
            "2026/": ["05/"], "2025/": ["07/"]}

    max_size = 0

    for year in years:
        for month in months[year]:

            out = open("/sps/grand/cprevotat/grand/efficiency/out_extract_infos/allowed_times_t0gps_" + year[:-1] + "_" + month[:-1] + ".txt", 'w')
            out.write("# allowed times for t0 (gps time), in seconds in unix time, 1st column : name of the file, then min time and max time \n")

            path = directory + year + month
            print("path : ", path)
            list_files = [str(f) for f in Path(path).rglob("*.root") if f.is_file()]
            print(f"We have {len(list_files)} files")
            list_files = [f for f in list_files if "MD" in f] # select only the MD data
            print(f"We have {len(list_files)} files after selecting only the MD data")
            list_files = [f for f in list_files if "TEST" not in f]
            print(f"We have {len(list_files)} files after removing the TEST data")
            #list_files = [f for f in list_ftimeiles if (int(f.split("/")[-1].split("_")[1]) < 20260519)] # select files before the 19th of May

            print(list_files)
            print(f"number of files in {month}: ", len(list_files))

            # should probably select only the MD files 
            snapshot_before = None


            for count, file in enumerate(list_files):

                if count % 100 == 0:
                    print("file number : ", count, "out of ", len(list_files))
                        
                """
                mem_mb = process.memory_info().rss / 1024**2
                print(f"[{count}] RSS: {mem_mb:.1f} MB")

                if count % 100 == 0:
                    print("file number : ", count, "out of ", len(list_files))

                

    
                # Take snapshots every 100 files
                if count % 100 == 0 and count > 0:
                    snapshot_after = tracemalloc.take_snapshot()
                    
                    if snapshot_before is not None:
                        stats = snapshot_after.compare_to(snapshot_before, 'lineno')
                        print(f"\n--- Top memory changes at file {count} ---")
                        for stat in stats[:5]:  # top 5 growers
                            print(stat)
                    
                    snapshot_before = snapshot_after
                """
    
                #print(file)
                file_root = dh.DataFile(file) # read the root file ?
                try:
                    rawvoltage_tree = file_root.trawvoltage
                except AttributeError:
                    print(f"Missing trawvoltage in file: {file}")
                    file_root.close()
                    del file_root
                    continue


                N_events = rawvoltage_tree.get_entries() # get the number of events of the tree (for any tree)
                #print("Number of events : ", N_events)

                """
                #loop : very bad but should work 
                times = []
                tmin = np.inf
                tmax = -np.inf
                for i in range(0, N_events):
                    adc_tree.get_entry(i)
                    event_seconds = np.array(adc_tree.du_seconds) # mostly not moving, same for many many events
                    event_nano = np.array(adc_tree.du_nanoseconds) # this one is moving / changing  ; we take the min of the events that have min seconds

                    time = event_seconds + event_nano*1e-9
                    if len(time) > max_size:
                        max_size = len(time)

                    min_time = np.min(time)
                    max_time = np.max(time)

                    tmin = min(tmin, min_time)
                    tmax = max(tmax, max_time)

                    times.extend(adc_tree.du_seconds) # mostly not moving, same for many many events

                print(tmin, tmax)

                print(N_events, len(times))

                      
                """
                """ # with the adc tree here
                #Without loop : I do get all the entries of the column, the array is flattened, so that I have t_ev0_du0, t_ev0_du1, t_ev1_du0 etc...
                draw_count = adc_tree.draw("du_seconds[]:du_nanoseconds[]", "", "goff") # extract the complete column from the tree ; "goff" : we don’t want to draw a histogram 
                ll_events_seconds = adc_tree.get_v1() # ll : low level
                events_seconds = np.frombuffer(ll_events_seconds, dtype=np.float64, count = draw_count).copy() # copy can save some memmory according to Claude
                #print(N_events, draw_count, np.shape(events_seconds))
                ll_events_nanoseconds = adc_tree.get_v2()
                events_nanoseconds = np.frombuffer(ll_events_nanoseconds, dtype = np.float64, count = draw_count).copy()
                """

                # with the raw voltage tree here, (ie we use gps time)
                draw_count = rawvoltage_tree.draw("gps_time[]", "", "goff") # extract the complete column from the tree ; "goff" : we don’t want to draw a histogram
                ll_events_seconds = rawvoltage_tree.get_v1() # ll : low level
                times = np.frombuffer(ll_events_seconds, dtype=np.float64, count = draw_count).copy()
                #print(events_seconds)
                #plt.figure()
                #plt.scatter([i for i in range(0, len(events_seconds))], events_seconds, s = 2)
                #plt.xlabel("Event number")
                #plt.ylabel("Seconds")
                #plt.show()


                #print(np.shape(events_nanoseconds), N_events, count)

                #times = events_seconds + events_nanoseconds*1e-9
                times = times[(times < 1.8e9) & (times > 1.7e9)] # january 2027 and November 2023 (ie it’s a very shallow cut)
                if len(times) < 2:
                    print("weird file with only ", len(times), " times remaining : ", file)
                    continue
                tmin = np.min(times)
                tmax = np.max(times)

                #print(tmin, tmax)

                out.write(file + " " + str(tmin) + " " + str(tmax) + "\n")

                rawvoltage_tree.close_file()
                file_root.close()
                del rawvoltage_tree, file_root, times
                gc.collect()
                
                                

                # to plot the times
                """

                times = []

                

                for i in range(0, N_events):
                    adc_tree.get_entry(i)
                    event_seconds = np.array(adc_tree.du_seconds) # mostly not moving, same for many many events
                    event_nano = np.array(adc_tree.du_nanoseconds) # this one is moving / changing  ; we take the min of the events that have min seconds

                    time = event_seconds + event_nano*1e-9
                    times.extend(time)

                print(np.min(times), np.max(times))
                plt.figure()
                plt.scatter([i for i in range(0, len(times))], times, s = 2)
                plt.xlabel("Event number")
                plt.ylabel("Time (s)")
                plt.show()
                """
        out.close()



def define_allowed_range_t0_test():

    """
    save somewhere the times at which the detector triggered something, ie was operational, so that then we can draw t0 among those allowed times
    I don’t  know why, but memory used by this function keeps increasing slowly (order 0.2 Mb per file, and it starts around 700 Mb), so that it’s probably better to run it several
    times, once for each month... ?
    """


    directory = "/sps/grand/data/gp80/GrandRoot/"
    years = ["2024/", "2025/", "2026/"]
    years = ["2026/"] # to keep things short
    months = {"2024/": ["11/", "12/"],
            "2027/": ["01/", "02/", "03/", "04/", "05/", "06/", "07/", "08/", "09/", "10/", "11/", "12/"],
            "2026/": ["01/"], "2025/": ["07/"]}

    max_size = 0

    for year in years:
        for month in months[year]:

            out = open("/sps/grand/cprevotat/grand/efficiency/out_extract_infos/test_allowed_times_t0_" + year[:-1] + "_" + month[:-1] + ".txt", 'w')
            out.write("# allowed times for t0, in seconds since I don’t know when, 1st column : name of the file, then min time and max time \n")

            path = directory + year + month
            print("path : ", path)
            list_files = [str(f) for f in Path(path).rglob("*.root") if f.is_file()]
            print(f"number of files in {month}: ", len(list_files))



            for count, file in enumerate(list_files):

                if count % 100 == 0:
                    print("file number : ", count, "out of ", len(list_files))
                        

    
                #print(file)
                file_root = dh.DataFile(file) # read the root file ?
                try:
                    adc_tree, run_tree = file_root.tadc, file_root.trun
                except AttributeError:
                    print(f"Missing tadc in file: {file}")
                    file_root.close()
                    del file_root
                    continue


                event_array = np.array(adc_tree.get_list_of_events())
                run_list = np.unique(event_array[:, 1])  # Get unique run numbers
                N_events = adc_tree.get_entries() # get the number of events of the tree (for any tree)
                #print("Number of events : ", N_events)

                min_times, max_times = [], []
                #print(run_list)
                for run_number in run_list:
                    run_tree.get_run(run_number)

                    min_time = run_tree.first_event_time
                    max_time = run_tree.last_event_time

                    first_event_id = run_tree.first_event
                    last_event_id = run_tree.last_event


                    print(min_time, max_time, first_event_id, last_event_id)
               
                    out.write(file + "  " + str(run_number) + " " + str(min_time) + " " + str(max_time) + "\n")

                adc_tree.close_file()
                file_root.close()
                del adc_tree, run_tree, file_root
                gc.collect()

                                


        out.close()



    
def plot_times_for_file(path):

    """
    plot times for a given file, it seems there are some weirds things sometimes
    """


    max_size = 0


    snapshot_before = None


    file_root = dh.DataFile(path) # read the root file ?

    print("the root file is : ", path)
    try:
        adc_tree = file_root.tadc
    except AttributeError:
        print(f"Missing tadc in file: {path}")
        file_root.close()
        del file_root

    rawvoltage_tree = file_root.trawvoltage


    #print(adc_tree)
    #print(rawvoltage_tree)

    N_events = adc_tree.get_entries() # get the number of events of the tree (for any tree)
    print("Number of events : ", N_events)


    """
    #loop : very bad but should work 
    times = []
    tmin = np.inf
    tmax = -np.inf
    for i in range(0, N_events):
        adc_tree.get_entry(i)
        event_seconds = np.array(adc_tree.du_seconds) # mostly not moving, same for many many events
        event_nano = np.array(adc_tree.du_nanoseconds) # this one is moving / changing  ; we take the min of the events that have min seconds

        time = event_seconds + event_nano*1e-9
        if len(time) > max_size:
            max_size = len(time)

        min_time = np.min(time)
        max_time = np.max(time)

        tmin = min(tmin, min_time)
        tmax = max(tmax, max_time)
    print(tmin, tmax)
    min_times, max_times = tmin, tmax
            
    """   

    times = []
    times_gps = []
    diff_gps_adc = []

    du_ids = []

    

    for i in range(0, N_events):
        
        adc_tree.get_entry(i)
        du_ids.extend(adc_tree.du_id)
    
        event_seconds = np.array(adc_tree.du_seconds) # mostly not moving, same for many many events
        event_nano = np.array(adc_tree.du_nanoseconds) # this one is moving / changing  ; we take the min of the events that have min seconds
        times.extend(event_nano)

        #time = event_seconds + event_nano*1e-9
        #print(len(time))
        #times.extend(time)
        
        #for time_du in time:
            #if time_du > 2e9:
                #print("weird time : ", time)

                #break
        
        rawvoltage_tree.get_entry(i)
        seconds_gps = np.array(rawvoltage_tree.gps_time)
        times_gps.extend(seconds_gps)
        diff_gps_adc.extend(seconds_gps - (event_seconds ))
        
    
    times = np.array(times)
    #times = times[(times < 1.8e9) & (times > 1.5e9)]
    #times = times[(times < 1.8e10) & (times > 1.5e9)]
    times_gps = np.array(times_gps)
    times_gps = times_gps[(times_gps < 1.8e9) & (times_gps > 1.5e9)]
    min_times = np.min(times)
    max_times = np.max(times)
    
    ### getting the times as Bohao does : ###

    du_ids = np.unique(du_ids)
    print("du_ids : ", du_ids)




    filename = path.replace("GrandRoot", "raw").replace(".root", ".bin")
    print("the raw data file is : ", filename)
    #filename = "/sps/grand/data/gp80/raw/2026/04/GP80_20260414_143613_RUN343_MD_20dB-GP65-58DUs-10s-512trace-FY2Float-newdataformat-noeventbuilder-0009.bin"
    """
    times_B = []  # [(timestamp, file, offset)]
    target_duid = du_ids[0]

    headerLength = 256
    DAQHeaderLength = 12

    DUid_in_DAQHeader_Pos = 8
    ChannelLengthPos = 64
    GPSTimePos = 96
    ADC1Pos = 516

    with open(filename, 'rb') as f:
        loopLength = 0

        while True:
            f.seek(headerLength + loopLength)
            a = f.read(4)
            if a == b"":
                break

            dataLength, = struct.unpack('I', a)

            f.seek(headerLength + DUid_in_DAQHeader_Pos + loopLength)
            du = f.read(4)
            duID, = struct.unpack('I', du)
            #print("duID : ", duID)
            
            #if duID == target_duid:
            if duID > 0:

                f.seek(headerLength + DAQHeaderLength + GPSTimePos + loopLength)
                c = f.read(7)
                year, month, day, hour, minute, second = struct.unpack('1H5B', c)

                if year == 0 or month == 0 or day == 0:
                    loopLength += dataLength
                    continue

                dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
                ts = int(dt.timestamp())

                times_B.append(ts)

            loopLength += dataLength

    print("N events in Bohao file : ", len(times_B))
    times_B = np.array(times_B)


    print("max diff between gps and adc is : ", np.max(diff_gps_adc), "s, min is : ", np.min(diff_gps_adc), "s")
    print("min and max times for the root file : ", min_times - min_times , max_times - min_times)
    print("min and max times for the Bohao file : ", np.min(times_B) - min_times, np.max(times_B) - min_times)
    print("min and max times for the gps times : ", np.min(times_gps) - min_times, np.max(times_gps) - min_times)
    #print("max diff gps-bohao, adc-bohao :", np.max(times_gps - times_B), np.max(times - times_B))
    print("Bin files : ",np.min(times_B), np.max(times_B))
    print("Root files : ", np.min(times),  np.max(times))"""
    plt.figure()
    plt.scatter([i for i in range(0, len(times))], np.array(times)-1.77e9, s = 2, color = "green", label = "ADC")
    #plt.plot([i for i in range(0, len(times_B))], np.array(times_B)-1.77e9, color = "red", label = "Bohao")
    plt.scatter([i for i in range(0, len(times_gps))], np.array(times_gps)-1.77e9, s = 2, color = "blue", label = "GPS")
    plt.xlabel("Event number")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.legend()
    plt.show()



def plot_allowed_range_t0(filename = "./efficiency/out_extract_infos/allowed_times_t0gps_2026_05.txt"):

    """Plot the allowed range for t0, to check that it’s not too bad

    """

    data = np.loadtxt(filename, dtype = str).T # load the output file of the previous function, to check the results
    print(np.shape(data), np.shape(data[0]))

    mask = np.char.find(data[0], "MD") != -1 # select only our MD data
    print(np.shape(data), data[0], data[1], data[2])
    filenames = data[0][mask]
    min_times = data[1][mask].astype(float)
    max_times = data[2][mask].astype(float)

    mask = (max_times > 1.8e9) | (min_times > 1.8e9) | (max_times < 1e9) | (min_times < 1e9) # weird values, probably due to some problem in the file, we remove them
    print("number of files in which I have weird values : ", np.shape(mask[mask == True]), "out of ", np.shape(min_times))
    min_times = min_times[~mask]
    max_times = max_times[~mask]
    print(min_times, max_times)

    min_min_times = np.min(min_times)
    max_max_times = np.max(max_times)

    min_to_plot = min_times - min_min_times
    max_to_plot = max_times - min_min_times

    idx = np.argsort(min_to_plot)
    min_to_plot, max_to_plot = min_to_plot[idx], max_to_plot[idx]

    print("min to plot : ", min_to_plot)
    print("max to plot : ", max_to_plot)
    print(np.where(max_to_plot < min_to_plot))

    print(np.min(min_to_plot))

    print("total time range is : ", max_max_times - min_min_times, "s, ie ", (max_max_times - min_min_times)/3600, "h")
    print("the observation time here is (beware some intervals seem to overlap) ", np.sum(max_to_plot - min_to_plot), "s, ie ", np.sum(max_to_plot - min_to_plot)/3600, "h")
    print("the ratio of the observation time over the total time range is : ", np.sum(max_to_plot - min_to_plot)/(max_max_times - min_min_times))

    merged_tmin, merged_tmax, liste_idx_in_intervals = merge_intervals(min_times, max_times)
    print("before and after merging : ", len(min_times), len(merged_tmin))

    print("the observation time here is ", np.sum(merged_tmax - merged_tmin), "s, ie ", np.sum(merged_tmax - merged_tmin)/3600, "h")

    idx = np.argsort(min_times)
    min_times, max_times, ordered_filenames = min_times[idx], max_times[idx], filenames[idx] # sort the intervals by their min time, as we do with merge_intervals

    for i in range(0, len(liste_idx_in_intervals)):
        if len(liste_idx_in_intervals[i]) > 1:
            print("interval ", i, " has ", len(liste_idx_in_intervals[i]), " files that contribute to it, with values : ", min_times[liste_idx_in_intervals[i]], max_times[liste_idx_in_intervals[i]], ordered_filenames[liste_idx_in_intervals[i]])

    plt.figure()
    for i in range(0, len(min_times)):
        #print(min_to_plot[i])
        plt.plot([min_to_plot[i] , max_to_plot[i] ], [1 - (i+1)*1e-4, 1 - (i+1)*1e-4], color = "blue")

    plt.plot(0, 0, color = "blue", label = "original intervals")
    plt.plot(0, 0, color = "red", label = "merged intervals")

    for i in range(0, len(merged_tmin)):
        plt.plot([merged_tmin[i] - min_min_times, merged_tmax[i] - min_min_times], [1 - (i+1)*1e-4, 1 - (i+1)*1e-4], color = "red", alpha = 0.5)
    plt.xlabel("Time interval (s)")
    plt.ylabel("")
    plt.title("Allowed range for t0")
    #plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()



def merge_intervals(t_min, t_max): # this code is not super efficient I guess... but it’s quick enough (order s for 1e4 intervals) ; the thing is that for July it returns only 1 interval (that’s before considering only MDs ?)
    idx = np.argsort(t_min)
    t_min, t_max = t_min[idx], t_max[idx] # sort the intervals by their min time, so that we can merge them more easily

    merged_min = [t_min[0]]
    merged_max = [t_max[0]]

    liste_idx_in_intervals = [[0]] # list of the idx of the files that contributes to each interval ; should have shape of merged_min

    for i in range(1, len(t_min)):
        if i % 1000 == 0:
            print("merging intervals : ", i, "out of ", len(t_min))
        if merged_max[-1] - t_min[i] > 1e-6: # it seems more numerical than anything else ?
            merged_max[-1] = max(merged_max[-1], t_max[i]) # if our t_min is smaller than our p
            liste_idx_in_intervals[-1].append(i)
        else:
            merged_min.append(t_min[i])
            merged_max.append(t_max[i])
            liste_idx_in_intervals.append([i])

        
    print("result of merge_intervals", liste_idx_in_intervals) # okay it seems to be working
    return np.array(merged_min), np.array(merged_max), liste_idx_in_intervals


def prepare_for_drawing_t0(filename = "./efficiency/out_extract_infos/allowed_times_t0_2026_04.txt"):
    # returns the ordered list of data files, the merged t_min and the merged tmax, and the list of idx files that contribute to each interval
    data = np.loadtxt(filename, dtype = str).T # load the output file of the previous function, to check the results

    mask = np.char.find(data[0], "MD") != -1 # select only our MD data
    min_times = data[1][mask].astype(float)
    max_times = data[2][mask].astype(float)

  
    min_min_times = np.min(min_times)
    max_max_times = np.max(max_times)

    min_to_plot = min_times - min_min_times
    max_to_plot = max_times - min_min_times

    idx = np.argsort(min_to_plot)
    min_to_plot, max_to_plot = min_to_plot[idx], max_to_plot[idx]

    merged_tmin, merged_tmax, liste_idx_in_intervals = merge_intervals(min_times, max_times)

    return data[0][idx], merged_tmin, merged_tmax, liste_idx_in_intervals





def draw_t0(N_draws, t_min, t_max): # N_times : number of times to draw

    """ Draw a time ; the implementation here directly draw a time in the allowed intervals, but for that we need to have intervals that don’t overlap : we first need to merge ;
    but then it’s still degenerated : to one t0 we can have several different files, so we’ll need to open all those files and then extract the closest events
    """

    lengths = t_max - t_min # ideally t_min and t_max are ordered and merged # 

    cum = np.cumsum(lengths)
    total = cum[-1]

    u = np.random.uniform(0, total, size=N_draws) # draw N_draws times uniformly between 0 and total

    idx = np.searchsorted(cum, u) # find the interval to which u belong (it returns the index)

    #  compute offsets inside intervals
    prev_cum = np.zeros_like(u)
    mask = idx > 0 
    prev_cum[mask] = cum[idx[mask] - 1] # if the idx was 0, then we still have prev_cum = 0, so we don’t have any issue

    # final sample
    t0 = t_min[idx] + (u - prev_cum) # start from t_min, then add u - prev_cum, ie the time we drew - the time from cum sum up to the previous interval

    return t0, idx


def plot_t_Coreas_simulations(filename = "/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_0044"):
    d_input = dh.DataDirectory(filename)
    trun_l1, tadc_l1, tshower_l0 = d_input.trun_l1, d_input.tadc_l1, d_input.tshower_l0

    N_events = tadc_l1.get_entries()

    for i in range(N_events):

        tadc_l1.get_entry(i)
        #print(tadc_l1.du_seconds)
        du_seconds = np.array(tadc_l1.du_seconds)
        du_nanoseconds = np.array(tadc_l1.du_nanoseconds)
        times = du_seconds + du_nanoseconds*1e-9

        #print(du_nanoseconds/1e9)
        print(du_seconds)
        #print(times)
        




#path_to_file = "/sps/grand/DC2Training/ZHAireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0001/" #     # may find the COREAS simulations in this directory : /sps/grand/DC2_Coreas/old_sims/COREAS, but should ask someone for that (to know what they are etc) ; we don’t care for now as the structure should be similar to the Zhaires one
#extract_infos(path_to_file) # commands to extract the infos we need from the simulations

def plot_MD_trace():
    files = ['/sps/grand/data/gp80/GrandRoot/2026/05/GP80_20260523_031943_RUN491_MD_20dB-58DUs-512trace-FY2Float-Normal-Wonlinefilter-new-cs-daq-v1p0-cw-CT-60-40ADC-0119.root','/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260409_220716_RUN340_MD_20dB-GP65-58DUs-10s-512trace-FY2Float-0018.root', '/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260411_015433_RUN341_MD_20dB-GP65-58DUs-10s-512trace-FY2Float-newdataformat-0015.root', '/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260406_210924_RUN338_MD_20dB-GP65-42DUs-10s-512trace-FY2Float-dunhuangsiteTestGPU-0124.root', '/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260406_221454_RUN338_MD_20dB-GP65-42DUs-10s-512trace-FY2Float-dunhuangsiteTestGPU-0125.root', '/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260417_083248_RUN381_MD_20dB-58DUs-512trace-FY2Float-Beacon-10Hz-pulse-new-cs-daq-chengwei-0001.root', '/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260408_095323_RUN340_MD_20dB-GP65-58DUs-10s-512trace-FY2Float-0024.root', '/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260411_161233_RUN342_MD_20dB-GP65-58DUs-10s-512trace-FY2Float-newdataformat-noeventbuilder-0008.root', '/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260416_202503_RUN359_MD_20dB-58DUs-10s-1024trace-FY2Float-noeventbuilder-0011.root', '/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260417_064101_RUN378_MD_20dB-58DUs-512trace-FY2Float-Beacon-10Hz-pulse-new-cs-daq-chengwei-0001.root', '/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260419_033225_RUN426_MD_20dB-58DUs-512trace-FY2Float-Normal-Wonlinefilter-checkFEB66-new-cs-daq-chengwei-0001.root', '/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260419_044503_RUN426_MD_20dB-58DUs-512trace-FY2Float-Normal-Wonlinefilter-checkFEB66-new-cs-daq-chengwei-0002.root', '/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260405_083204_RUN338_MD_20dB-GP65-42DUs-10s-512trace-FY2Float-dunhuangsiteTestGPU-0089.root', '/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260412_113823_RUN342_MD_20dB-GP65-58DUs-10s-512trace-FY2Float-newdataformat-noeventbuilder-0033.root']
    for file in files:
        print("file : ", file)
        file_root = dh.DataFile(file)
        adc_tree = file_root.tadc
        event_list = adc_tree.get_list_of_events()
        #print("our event list is : ", event_list)

        #print(adc_tree)
        plt.figure()

        for i in range(0, 100):
            adc_tree.get_event(event_list[i][0], event_list[i][1])
            #print(adc_tree)

            du_id = np.array(adc_tree.du_id)
            print("du_id : ", du_id)
            tadc_trace = np.array(adc_tree.trace_ch) # we focus only on the 2 (3) first columns : x, y and z
            print(np.shape(tadc_trace))
            tadc_trace_X = tadc_trace[0][0]
            tadc_trace_Y = tadc_trace[0][1]

            
            print(np.shape(tadc_trace_X), np.shape(tadc_trace_Y))
            x = np.arange(len(tadc_trace_X))
            plt.plot(x+(i+1)*len(x), tadc_trace_Y, alpha = 0.5)
            plt.plot(x+(i+1)*len(x), tadc_trace_X, alpha = 1)
        plt.legend()
        plt.show()


def plot_trace_sims():
    #directory = "/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_0084"
    directory = "/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_0000"
    file_root = dh.DataDirectory(directory)
    adc_tree = file_root.tadc
    event_list = adc_tree.get_list_of_events()


    adc_tree.get_event(event_list[0][0], event_list[0][1])

    du_id = np.array(adc_tree.du_id)
    tadc_trace = np.array(adc_tree.trace_ch) # we focus only on the 2 (3) first columns : x, y and z
    tadc_trace_X = tadc_trace[0][0] # trace of the first du
    tadc_trace_Y = tadc_trace[0][1]

    
    print(np.shape(tadc_trace_X), np.shape(tadc_trace_Y))
    x = np.arange(len(tadc_trace_X))
    plt.figure()
    plt.plot(x, tadc_trace_Y, alpha = 0.5, label = "Y")
    plt.plot(x, tadc_trace_X, alpha = 1, label = "X")
    plt.legend()
    plt.show()


def plot_number_of_triggering_units():

    """Plot the number of triggering units for each event

    """

    data = np.loadtxt("./efficiency/out_extract_infos/out_sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0001.txt", dtype = str).T # load the output file of the previous function, to check the results
    id_units = data[0]
    count = np.array([id_units[i].count("_")-1 for i in range(0, len(id_units))])
    liste = [i for i in range(0, len(id_units))]

    plt.figure()
    plt.plot(liste, count, "x")
    plt.axhline(y = 5, color = "red")
    plt.ylabel("Number of detecting units")
    plt.show()

    return

def plot_Auger_spectrum():

    e_eV = np.logspace(17, 20, 100) # energy in eV

    spectrum = calculate_PAO_spectrum(e_eV)

    plt.figure()
    plt.plot(e_eV, spectrum * e_eV**3, label = "PAO spectrum")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (eV)")
    plt.ylabel("E^3 * dN/dE")
    plt.title("CR spectrum (PAO)")
    plt.legend()
    plt.show()



def get_trigger_position(filename = "2026/04/GP80_20260420_081103_RUN10303_CD_20dB-GP65-58DUs-512trace-FY2Float-Normal-Wonlinefilter-new-cs-daq-chengwei-CD-100000-195.root"):
    # should be a constant for CD files from what Olivier said ?

    path = "/sps/grand/data/gp80/GrandRoot/"
    filename = path + filename
    print("filename : ", filename)

    file_root = dh.DataFile(filename) # read the root file ?
    try:
        adc_tree = file_root.tadc_l0
    except AttributeError:
        print(f"Missing tadc in file: {filename}")
        file_root.close()
        del file_root


    N_events = adc_tree.get_entries()
    positions = np.zeros(N_events)

    for i in range(0, N_events):
        adc_tree.get_entry(i)
        position = adc_tree.trigger_position # that’s a vector, one trigger position for each DU
        print(position)
        positions[i] = position[0]
    
    plt.figure()
    plt.plot([i for i in range(0, N_events)], positions, "x")
    plt.xlabel("Event number")
    plt.ylabel("Trigger position")
    plt.show()


def plot_Nevents_per_DU_in_file(filename):

    if "bin" in filename:
        filename = filename.replace("raw", "GrandRoot").replace(".bin", ".root")  

    print("we are looking at this file : ", filename)


    file_root = dh.DataFile(filename) # read the root file ?
    try:
        raw_voltage_tree = file_root.trawvoltage
    except AttributeError:
        print(f"Missing trawvoltage in file: {filename}")
        file_root.close()
        del file_root


    N_events = raw_voltage_tree.get_entries()
    counts = {}

    for i in range(0, N_events):
        raw_voltage_tree.get_entry(i)
        du_ids = raw_voltage_tree.du_id # that’s a vector, one du_id for each DU that triggered

        for du_id in du_ids:
            if du_id not in counts:
                counts[du_id] = 1
            else:
                counts[du_id] += 1 
    

    plt.figure()
    plt.bar(counts.keys(), counts.values())
    # I want to write the DU ID on the x axis above the bar
    plt.xticks(list(counts.keys()), rotation=45, ha="right")
    plt.xlabel("DU ID")
    plt.ylabel("Number of events")
    plt.title("Number of events per DU in file " + filename.split("/")[-1])
    plt.show()



def sim_distribution_E_angles():

    # should be uniform in zenith and azimuth

    energies = []
    zeniths = []
    azimuths = []

    for i in range(0, 150):

        data_directory = f"/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_{i:04d}" # ensure correct formating 


        print('out_judge_trigger/'+data_directory[data_directory.find('sim'):], data_directory)

        ### Read GRAND root data
        d_input = dh.DataDirectory(data_directory)
        tshower_l0, tadc_l1 =  d_input.tshower_l0, d_input.tadc_l1

        event_list = tadc_l1.get_list_of_events()
        nb_events  = len(event_list)
        if nb_events == 0: sys.exit("No events in the file. Exiting.")

        ### Start event loop
        previous_run = None


        for event_number,run_number in event_list:

            assert isinstance(event_number, int)
            assert isinstance(run_number, int)

            tshower_l0.get_event(event_number, run_number)
            
                
            energy = tshower_l0.energy_primary
            zenith = tshower_l0.zenith
            azimuth = tshower_l0.azimuth

            energies.append(energy)
            zeniths.append(zenith)
            azimuths.append(azimuth)

        tshower_l0.close_file()
        tadc_l1.close_file()
        d_input.close()
        del tshower_l0, tadc_l1, d_input
        gc.collect()

    


    print("min and max energies : ", np.min(energies), np.max(energies))
    print("min and max zeniths : ", np.min(zeniths), np.max(zeniths))
    print("min and max azimuths : ", np.min(azimuths), np.max(azimuths))  

    bins_zenith = np.linspace(64, 90, 15)
    bins_cos_zenith = np.linspace(0, 0.5, 15)
    bins_azimuth = np.linspace(0, 360, 30)
    bins_energy = np.logspace(8, 11, 15)

    mid_bins_zenith = (bins_zenith[:-1] + bins_zenith[1:]) / 2
    mid_bins_cos_zenith = (bins_cos_zenith[:-1] + bins_cos_zenith[1:]) / 2
    mid_bins_azimuth = (bins_azimuth[:-1] + bins_azimuth[1:]) / 2
    mid_bins_energy = np.sqrt(bins_energy[:-1] * bins_energy[1:])



    histogram_zenith = np.histogram(zeniths, bins = bins_zenith)[0]
    histogram_cos_zenith = np.histogram(np.cos(np.radians(zeniths)), bins = bins_cos_zenith)[0]
    histogram_azimuth = np.histogram(azimuths, bins = bins_azimuth)[0]
    histogram_energy = np.histogram(energies, bins = bins_energy)[0]


    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(bins_zenith[:-1], histogram_zenith, label = "Zenith distribution")
    plt.xlabel("Zenith angle (degrees)")
    plt.ylabel("Number of events")
    plt.title("Zenith distribution")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(bins_cos_zenith[:-1], histogram_cos_zenith, label = "Cosine of zenith distribution")
    plt.xlabel("Cosine zenith angle")
    plt.ylabel("Number of events")
    plt.title("Cosine of zenith distribution")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(bins_azimuth[:-1], histogram_azimuth, label = "Azimuth distribution")
    plt.xlabel("Azimuth angle (degrees)")
    plt.ylabel("Number of events")
    plt.title("Azimuth distribution")
    plt.legend()   

    plt.subplot(2, 2, 4)
    plt.plot(bins_energy[:-1], histogram_energy, label = "Energy distribution")
    plt.xscale("log")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Number of events")
    plt.title("Energy distribution")
    plt.legend()    


    plt.tight_layout()
    plt.savefig("simulated_distribution_E_angles.pdf", bbox_inches = "tight")
    plt.show()


def compute_normalisation_factor(max_zenith_degrees = 88, N_bins = 50):
    # function to weight the simulations in zenith and energy

    years_to_s = 31557600

    energies = np.array([])
    zenith = np.array([])

    simulation_numbers = [i for i in range(0, 150)] # ensure correct formating

    for n_sim in simulation_numbers:
        with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{n_sim:04d}.json", "r") as f:
            data = json.load(f)


        energies = np.concatenate([energies, np.array([10**row["fixed"][3] for row in data])])
        zenith = np.concatenate([zenith, np.array([row["fixed"][4] for row in data])]) 

    to_delete = np.where(zenith > max_zenith_degrees)[0]
    energies = np.delete(energies, to_delete)
    zenith = np.delete(zenith, to_delete)


    print("max zenith is now : ", np.max(zenith))

    with open("/sps/grand/cprevotat/grand/efficiency/out_extract_infos/time_period_2026_05.txt", "r") as f:
        total_time = float(f.read())
        total_time = 15*24*3600 # in s  #from the 4th of May including the 18th but not the 19th


    E_min = np.min(energies *1e9) # in eV
    E_max = np.max(energies *1e9) # in eV

  
    weights = np.ones_like(energies) # start with weights = 1
    weights *= np.sin(np.radians(zenith)) # to have a distribution uniform in cos(zenith)

    spectrum_auger = calculate_PAO_spectrum(energies*1e9) / years_to_s * 1e-6 # switch energies from GeV to eV # unit of calculate_PAO_spectrum are years km^2 eV and sr : convert yr to seconds and km^2 to m^2
    weights *= spectrum_auger  # now the shape should be correct, we only need the normalization factor

    bins_energy = np.logspace(np.log10(E_min), np.log10(E_max), N_bins)
    mid_bins = np.sqrt(bins_energy[:-1] * bins_energy[1:])
    mid_bins = (bins_energy[:-1] + bins_energy[1:]) / 2
    histogram_energy = np.histogram(energies*1e9, bins = bins_energy, weights = weights)[0] # / np.diff(bins_energy) * mid_bins # should plot the spectrum at some point to check wether it is consistent or not


    integrate_PAO_spectrum()
    energies_auger = np.logspace(np.log10(mid_bins[0]), np.log10(mid_bins[-1]), 1000)
    auger_integrated = trpz(calculate_PAO_spectrum(energies_auger), energies_auger)
    auger_integrated  = auger_integrated / years_to_s * 1e-6 # put it in s and m
    print("number of cosmic rays according to Auger is : ", auger_integrated * np.pi * (-np.cos(np.radians(88))**2. + np.cos(np.radians(65))**2.) * (11.3 - (-5.2)) * (6.5 - (-4.5)) * 1e6 * total_time)

    integrate_events = trpz(histogram_energy, mid_bins)

    print(auger_integrated, integrate_events)

    scaling_factor = auger_integrated / integrate_events


    bins_zenith = np.linspace(64, max_zenith_degrees, 20)
    mid_bins_zenith = (bins_zenith[:-1] + bins_zenith[1:]) / 2
    histogram_zenith = np.histogram(zenith, bins = bins_zenith, weights = weights)[0]
    
    p = 3.
    plt.figure()
    plt.plot(mid_bins, histogram_energy * mid_bins**(p), label = "Simulated spectrum after weighting")
    plt.plot(bins_energy, 1e-9*calculate_PAO_spectrum(bins_energy) * bins_energy**p, label = "PAO spectrum")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (eV)")
    plt.ylabel("dN/dE")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(mid_bins_zenith, histogram_zenith, label = "Zenith distribution after weighting")
    plt.xlabel("Zenith angle (degrees)")
    plt.ylabel("Number of events")
    plt.title("Zenith distribution after weighting")
    plt.legend()
    plt.show()



    plt.figure()
    plt.plot(mid_bins, scaling_factor * histogram_energy * mid_bins**p, label = "Simulated spectrum after scaling")
    plt.plot(bins_energy, calculate_PAO_spectrum(bins_energy) * bins_energy**p / years_to_s * 1e-6, label = "PAO spectrum")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (eV)")
    plt.ylabel("dN/dE")
    plt.legend()
    plt.show()

    print(scaling_factor)
    print("comparison : ", auger_integrated, trpz(histogram_energy, mid_bins) * scaling_factor)
    weights *= scaling_factor # multiply the intensity in s-1 per the observation time in s


    np.savetxt("./efficiency/out_extract_infos/simulation_normalisation_factor.txt", np.array([scaling_factor])) # it doesn’t take time into account for now ; that’s the only thing we should care about for now, the rest should be computed on the fly
    # for the time it depends on whether we compute the efficiency (ie look at data only when the detector is on) or look at the result in the given time period, including times where the detector is off
    # okay if we want to compute the number of cosmic rays we wish we had detected in a given month (then I just need to multiply by the detection surface, and integrate the intensity over energies, and multiply by the solid angle that is being used in the simulations)

    return 

def compute_NCRs():
    years_to_s = 31557600
    min_zenith = np.radians(65)
    max_zenith_degrees = 83
    max_zenith = np.radians(max_zenith_degrees) # 88 if we follow simulations, 83 for Nathan
    N_bins = 20

    compute_normalisation_factor(max_zenith_degrees, N_bins)
    # compute the number of cosmic rays we wish we had detected in a given month, by integrating the intensity over energies, and multiplying by the solid angle that is being used in the simulations (which is 2pi(1-cos(64)) for zeniths between 64 and 90)

    with open("/sps/grand/cprevotat/grand/efficiency/out_extract_infos/simulation_normalisation_factor.txt", "r") as f:
        scaling_factor = np.loadtxt(f)

    with open("/sps/grand/cprevotat/grand/efficiency/out_extract_infos/time_period_2026_05.txt", "r") as f:
        total_time = float(f.read())
        total_time = 15*24*3600 # in s  #from the 4th of May including the 18th but not the 19th

    energies = np.array([])
    zenith = np.array([])
    azimuth = np.array([])

    simulation_numbers = [i for i in range(0, 150)] # ensure correct formating

    for n_sim in simulation_numbers:
        with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{n_sim:04d}_th30.json", "r") as f:
            data = json.load(f)

        data_fixed = [row["fixed"] for row in data]
        data_triggered2 = [row["triggering_events_2"] for row in data]
        #print(data_triggered2)
        #print(len(data_triggered2), len(data_triggered2[0]))
        for i in range(0, len(data_triggered2)):
            if data_fixed[i][4] > max_zenith_degrees: # if the zenith is too large, we don’t consider the event, even if it triggered, because it’s not in the range of our simulations
                continue

            N_triggered_dus = 0
            du_ids = []

            if len(data_triggered2[i]) < 5:
                continue

            for j in range(len(data_triggered2[i])):
                if (np.abs(data_triggered2[i][j][2]) < 10) & (np.abs(data_triggered2[i][j][3]) < 10): # distance in time to closest MD event
                    du_ids.append(data_triggered2[i][j][0])


            if len(np.unique(du_ids)) >= 5: 
                energies = np.concatenate([energies, np.array([10**data_fixed[i][3]])])
                zenith = np.concatenate([zenith, np.array([data_fixed[i][4]])])
                azimuth = np.concatenate([azimuth, np.array([data_fixed[i][5]])]) # don’t consider the cluster part for now
                
                


    spectrum_auger = calculate_PAO_spectrum(energies*1e9) / years_to_s * 1e-6 # putting it in m
    weights = np.ones_like(energies) # start with weights = 1
    weights *= np.sin(np.radians(zenith)) * spectrum_auger * scaling_factor * np.cos(np.radians(zenith)) # last cos for geometry

    E_min = 10**17. # in eV
    E_max = 10**20. # in eV

    bins_energy = np.logspace(np.log10(E_min), np.log10(E_max), N_bins)
    #mid_bins = np.sqrt(bins_energy[:-1] * bins_energy[1:])
    mid_bins = (bins_energy[:-1] + bins_energy[1:]) / 2

    histogram = np.histogram(energies*1e9, bins = bins_energy, weights = weights)[0] #/ np.diff(bins_energy) * mid_bins # should plot the spectrum at some point to check wether it is consistent or not
    print("histogram : ", histogram)

    print("number of detected events : ", np.shape(energies))
    plt.figure()
    plt.scatter([i for i in range(0, len(energies))], energies*1e9, s = 2)
    plt.yscale("log")
    plt.show()


    plt.figure()
    plt.plot(mid_bins, histogram * mid_bins**3., label = "Simulated spectrum after weighting")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (eV)")
    plt.ylabel("dN/dE")
    plt.legend()
    plt.show()  


    integrated_intensity = trpz(histogram, mid_bins) # integrate the intensity over energies

    solid_angle = 2 * np.pi * (-np.cos(max_zenith) + np.cos(min_zenith)) # solid angle for zeniths between min and 90
    surface = (11.3 - (-5.2)) * (6.5 - (-4.5)) * 1e6 # in m^2

    N_CR = integrated_intensity * solid_angle * surface * total_time # number of cosmic rays we wish we had detected in a given month

    print("Number of cosmic rays we wish we had detected in a our time period : ", N_CR)

    histogram = np.histogram(energies*1e9, bins = bins_energy, weights = weights)[0] 
    N_CR2 = np.sum(histogram * np.diff(bins_energy)) * solid_angle * surface * total_time
    print("N_CR2 : ", N_CR2)

    histogram = np.histogram(energies*1e9, bins = bins_energy, weights = weights)[0]
    d_log_E = np.diff(np.log(bins_energy))   # = const for log-uniform bins
    integrated_intensity = np.sum(histogram * mid_bins * d_log_E)
    N_CR3 = integrated_intensity * solid_angle * surface * total_time
    print("N_CR3 : ", N_CR3)


def compute_Nevents():
    # more up to date than compute_N_CRs.
    years_to_s = 31557600

    min_zenith = np.radians(65)
    max_zenith_degrees = 83 #83
    max_zenith = np.radians(max_zenith_degrees) # 88 if we follow simulations, 83 for Nathan
    N_bins = 20

    bins_energy = np.logspace(17, 20, N_bins) # in eV
    mid_bins_energy = np.sqrt(bins_energy[:-1] * bins_energy[1:]) # in eV

    bins_zenith = np.linspace(min_zenith, max_zenith, N_bins+1) # in radians # just to check that everything works fine
    mid_bins_zenith = (bins_zenith[:-1] + bins_zenith[1:]) / 2 # in radians

    total_time = 31*24*3600#15*24*3600 # in s  #from the 4th of May including the 18th but not the 19th

    energies_triggered = np.array([])
    zenith_triggered = np.array([])
    azimuth_triggered = np.array([])

    energies_all = np.array([])
    zenith_all = np.array([])

    simulation_numbers = [i for i in range(0, 150)] # ensure correct formating

    for n_sim in simulation_numbers:
        if n_sim == 350: # it seems simulation 35 didn’t run
            continue
        print(n_sim)
        with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{n_sim:04d}_th30.json", "r") as f:
            data = json.load(f)

        data_fixed = [row["fixed"] for row in data]
        data_triggered2 = [row["triggering_events_2"] for row in data]
        #print(data_triggered2)
        #print(len(data_triggered2), len(data_triggered2[0]))
        for i in range(0, len(data_fixed)):
            if data_fixed[i][4] > max_zenith_degrees: # if the zenith is too large, we don’t consider the event, even if it triggered, because it’s not in the range of our simulations
                continue # should uncomment that, commented it to compare with plot_2d_efficiency
            energies_all = np.concatenate([energies_all, np.array([10**data_fixed[i][3]])])
            zenith_all = np.concatenate([zenith_all, np.array([data_fixed[i][4]])])

        for i in range(0, len(data_triggered2)):
            if data_fixed[i][4] > max_zenith_degrees: # if the zenith is too large, we don’t consider the event, even if it triggered, because it’s not in the range of our simulations
                continue # should also be uncommented

            N_triggered_dus = 0
            du_ids = []

            if len(data_triggered2[i]) < 5:
                continue

            for j in range(len(data_triggered2[i])):
                if (np.abs(data_triggered2[i][j][2]) < 10) & (np.abs(data_triggered2[i][j][3]) < 10): # distance in time to closest MD event
                    du_ids.append(data_triggered2[i][j][0])


            if len(np.unique(du_ids)) >= 5: 
                energies_triggered = np.concatenate([energies_triggered, np.array([10**data_fixed[i][3]])])
                zenith_triggered = np.concatenate([zenith_triggered, np.array([data_fixed[i][4]])])
                azimuth_triggered = np.concatenate([azimuth_triggered, np.array([data_fixed[i][5]])]) # don’t consider the cluster part for now

    weights_triggered = calculate_PAO_spectrum(energies_triggered*1e9)  # last cos for geometry
    weights_all = calculate_PAO_spectrum(energies_all*1e9) 


    weights_triggered = weights_triggered * np.sin(np.radians(zenith_triggered)) * np.cos(np.radians(zenith_triggered)) # sin : to get uniform distribution in cos(zenith)
    weights_all = weights_all  * np.sin(np.radians(zenith_all)) 


    hist_triggered = np.histogram(energies_triggered*1e9, bins = bins_energy, weights = weights_triggered)[0]
    hist_all = np.histogram(energies_all*1e9, bins = bins_energy, weights = weights_all)[0]

    total_surface = (11.3 - (-5.2)) * (6.5 - (-4.5)) * 1e6 # in m^2


    hist_triggered_2d = np.histogram2d(energies_triggered*1e9, np.radians(zenith_triggered), bins = [bins_energy, bins_zenith], weights = weights_triggered)[0]  # 
    hist_all_2d = np.histogram2d(energies_all*1e9, np.radians(zenith_all), bins = [bins_energy, bins_zenith], weights = weights_all)[0] # 

    print("summ of the histograms : ", np.sum(hist_triggered_2d, axis = (0, 1)), np.sum(hist_all_2d, axis = (0, 1)))

    print("number of events : ", len(energies_all))
    print("number of triggered events : ", len(energies_triggered))
    print(np.min(hist_triggered_2d), np.max(hist_triggered_2d))

    efficiency_2d = np.nan_to_num(hist_triggered_2d / hist_all_2d) # what about when hist_all_2d is 0 : set the value to 0
    print(np.sum(efficiency_2d), np.sum(efficiency_2d, axis = 0))

    exposure_E = 2 * np.pi * total_time * total_surface * trpz(np.sin(mid_bins_zenith)[None, :] * efficiency_2d, mid_bins_zenith)
    exposure_theta = 2 * np.pi * total_time * total_surface * trpz(efficiency_2d, mid_bins_energy, axis = 0)

    exposure = 2 * np.pi * total_time * total_surface * hist_triggered / hist_all * (-np.cos(max_zenith) + np.cos(min_zenith)) # in m^2 s sr

    print("ratio exposures : ", exposure_E / exposure)

    

    #auger_integrated = quad(calculate_PAO_spectrum, 10**17, 10**20)[0] * 1e-6 / years_to_s # in s and m^2
    auger_integrated = trpz(calculate_PAO_spectrum(mid_bins_energy), mid_bins_energy) * 1e-6 / years_to_s # in s and m^2
    print("exposure : ", exposure, "auger integrated : ", auger_integrated)
    print(hist_triggered / hist_all)
    print(auger_integrated * 2 * np.pi * (-np.cos(max_zenith)**2. + np.cos(min_zenith)**2.) * total_surface * total_time)
    print("N_CR:",  trpz(calculate_PAO_spectrum(mid_bins_energy) * exposure, mid_bins_energy) * 1e-6 / years_to_s )
    print("N_CR2:",  trpz(calculate_PAO_spectrum(mid_bins_energy) * exposure_E, mid_bins_energy) * 1e-6 / years_to_s ) # this one is more to be trusted I think


                
    plt.figure()
    plt.plot(mid_bins_energy, exposure_E, label = "Exposure from 2D efficiency")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy [eV]")
    plt.ylabel("Exposure [m^2 s sr]")
    plt.legend()
    plt.title("Exposure from 2D efficiency")
    plt.show()

    plt.figure()
    plt.plot(np.cos(mid_bins_zenith), exposure_theta, label = "Exposure from 1D efficiency")
    plt.ylabel("Exposure [m^2 s sr]")
    plt.xlabel("cos(zenith)") # is this correct ?
    plt.yscale("log")
    plt.legend()
    plt.show()

    plt.figure()
    plt.pcolormesh(mid_bins_energy, np.cos(mid_bins_zenith), efficiency_2d.T, shading = "auto")
    plt.xscale("log")
    plt.xlabel("Energy [eV]")
    plt.ylabel("cos(zenith)")
    plt.colorbar(label = "Efficiency")
    plt.title("Efficiency as a function of energy and zenith angle")
    plt.show()

    N_events_per_energy = [quad(calculate_PAO_spectrum, bins_energy[i], bins_energy[i+1])[0] * exposure_E[i] * 1e-6 / years_to_s for i in range(len(bins_energy)-1)]
    print("N_CR3:", np.sum(N_events_per_energy))

    plt.figure()
    plt.scatter(mid_bins_energy, N_events_per_energy, label = "Number of events per energy bin")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy [eV]")
    plt.ylabel("Number of events per bin")
    plt.legend()
    plt.show()

    return




def get_idx_1st_T1crossing(filename, th_parameters):
    
    path = "/sps/grand/data/gp80/GrandRoot/"
    filename = path + filename
    print("filename : ", filename)

    file_root = dh.DataFile(filename) # read the root file ?
    try:
        adc_tree = file_root.tadc
    except AttributeError:
        print(f"Missing tadc in file: {filename}")
        file_root.close()
        del file_root


    N_events = adc_tree.get_entries()
    idx_1st_crossing_x = []
    idx_1st_crossing_y = []


    for i in range(0, N_events):
        adc_tree.get_entry(i)
        du_ids = adc_tree.du_id
        for (n, ids) in enumerate(du_ids):


            #print(adc_tree.trace_ch)
            traces = adc_tree.trace_ch # we focus only on the 2 (3) first columns : x, y and z
            #print(len(traces))

            """
            for i, event in enumerate(traces):
                for j, ch in enumerate(event):
                    print(f"du {i}, channel {j}, length = {len(ch)}")

            continue
            """
            traces = np.array(adc_tree.trace_ch)
            trace_x = traces[n][1]  # select the du and then the channel
            trace_y = traces[n][2]
            #print(np.shape(traces), np.shape(trace_x), np.shape(trace_y))
            """
            idx, _, _ = FLT0.trigger_FLT0(trace_x, {"th1":th_parameters[ids][1], "th2":th_parameters[ids][0], "t_sepmax": 50, "nc_max": 7, "nc_min": 2, "t_quiet": 500, "t_period": 500})
            if len(idx) > 0:
                idx_1st_crossing_x.append(idx[0])

            idx, _, _ = FLT0.trigger_FLT0(trace_y, {"th1":th_parameters[ids][3], "th2":th_parameters[ids][2], "t_sepmax": 50, "nc_max": 7, "nc_min": 2, "t_quiet": 500, "t_period": 500})
            if len(idx) > 0:
                idx_1st_crossing_y.append(idx[0])
            """
            
            idx = np.where(trace_x > th_parameters[ids][1])[0] # th1 on X 
            if i == 0 and n == 0:
                print(np.shape(trace_x), np.shape(trace_y))
            #if len(idx) > 0: 
                #idx = idx[idx > 100] # I’d say that’s for the case where we add noise on top of simulated data, and in this case the notch filter is not well made and create a peak (from what Marion said)
            if len(idx) > 0:
                idx_1st_crossing_x.append(int(idx[0]))
            idx = np.where(trace_y > th_parameters[ids][3])[0] # th1 on Y
            #if len(idx) > 0:
                #idx = idx[idx > 100]
            if len(idx) > 0:
                idx_1st_crossing_y.append(int(idx[0]))

            """
            print(idx_1st_crossing_x, idx_1st_crossing_y)
            plt.figure()
            plt.plot([i for i in range(0, len(trace_x))], trace_x, label = "X")
            plt.plot([i for i in range(0, len(trace_y))], trace_y, label = "Y")
            plt.axhline(y = th_parameters[ids][1], color = "red", label = "th1 X")
            plt.axhline(y = th_parameters[ids][3], color = "green", label = "th1 Y")
            plt.title(f"DU {ids} trace, event {i}")
            plt.legend()
            plt.show()
            """



    file_root.close()
    adc_tree.close_file()
    
    plt.figure()
    plt.scatter([i for i in range(0, len(idx_1st_crossing_x))], idx_1st_crossing_x, marker = "x", label="X-axis crossings", s = 2)
    plt.scatter([i for i in range(0, len(idx_1st_crossing_y))], idx_1st_crossing_y, marker = "o", label="Y-axis crossings", alpha = 0.5, s = 2)
    plt.xlabel("Event number")
    plt.ylabel("Trigger position")
    plt.legend()
    plt.show()
    return


def extract_noise_from_t0(InputDataFiles, target_timestamp, target_duid): # seems to be the location of the data files, timestamp in second, duid that we are looking for (should be feb)

    #print("started looking for the file")
    # paras
    headerLength = 256
    DAQHeaderLength = 12

    DUid_in_DAQHeader_Pos = 8
    ChannelLengthPos = 64
    GPSTimePos = 96
    ADC1Pos = 516

    #print(target_duid)

    #subfiles = os.listdir(InputDataFile)

    InputDataFiles.sort()
    #print("number of files in the directory : ", len(InputDataFiles))

    # *******************************************************
    # Step 1: collect all the timestamps' positions of the DU 
    # *******************************************************
    records = []  # [(timestamp, file, offset)]

    for i, filename in enumerate(InputDataFiles):

        #with open(os.path.join(InputDataFile, filename), 'rb') as f:
        with open(filename, 'rb') as f:
            loopLength = 0

            while True:
                f.seek(headerLength + loopLength)
                a = f.read(4)
                if a == b"":
                    break

                dataLength, = struct.unpack('I', a)

                # 读取 DU id
                f.seek(headerLength + DUid_in_DAQHeader_Pos + loopLength)
                du = f.read(4)
                duID, = struct.unpack('I', du)
                #print("DU ID : ", duID)

                

                if duID == target_duid:

                    # 读取时间
                    f.seek(headerLength + DAQHeaderLength + GPSTimePos + loopLength)
                    c = f.read(7)
                    year, month, day, hour, minute, second = struct.unpack('1H5B', c)

                    if year == 0 or month == 0 or day == 0:
                        loopLength += dataLength
                        continue

                    dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
                    ts = int(dt.timestamp())

                    # get the filename
                    records.append((ts, filename, loopLength))

                loopLength += dataLength

    # *****************************************
    # Step 2: find the most 2 closet timestamps
    # *****************************************
    if len(records) < 2:
        #print("No enough records found for this DU : ", target_duid, InputDataFiles)
        return None, 0.0, 0.0, "failed", "failed"

    records.sort(key=lambda x: abs(x[0] - target_timestamp))
    closest_two = records[:2]

    #print("Closest timestamps:")
    #for r in closest_two:
        #print(r[0])

    # ============================================
    # Step 3: read the taces' info correspondingly
    # ============================================
    result = {target_duid: {}}

    close_enough = True

    for ts, filename, offset in closest_two:

        with open(filename, 'rb') as f:

            # channel length
            f.seek(headerLength + DAQHeaderLength + ChannelLengthPos + offset)
            b = f.read(8)
            ch1_len, ch2_len, ch3_len, ch4_len = struct.unpack('4H', b)

            channelLength = ch1_len * 2

            # ===== Channel X =====
            f.seek(headerLength + DAQHeaderLength + ADC1Pos + offset + channelLength)
            ch1_data = []
            for _ in range(ch1_len):
                m = f.read(2)
                if len(m) < 2:
                    print("Unexpected end of file while reading channel data.", filename)
                    if len(ch1_data) < 512:
                        #fill it with 0
                        ch1_data.extend([0] * (512 - len(ch1_data)))
                    break
                adc, = struct.unpack('h', m)
                ch1_data.append(adc)

            channelLength += ch1_len * 2

            # ===== Channel Y =====
            f.seek(headerLength + DAQHeaderLength + ADC1Pos + offset + channelLength)
            ch2_data = []
            for _ in range(ch2_len):
                m = f.read(2)
                if len(m) < 2:
                    print("Unexpected end of file while reading channel data.", filename)
                    if len(ch2_data) < 512:
                        #fill it with 0
                        ch2_data.extend([0] * (512 - len(ch2_data)))
                    break
                adc, = struct.unpack('h', m)
                ch2_data.append(adc)

            # ====== Save data ======
            result[target_duid][ts] = {
                "channel1": ch1_data,
                "channel2": ch2_data
            }


    # *******************************
    # Output of the results
    # *******************************
    #print("\nFinal result:\n")

    output = []

    for du, data in result.items():
        #print("DU:", du)
        for ts, ch in data.items():
            #print("  Timestamp:", ts)
            #print("    Channel1 length:", len(ch["channel1"]))
            # print("         Channel 1 traces: ", ch["channel1"])
            #print("    Channel2 length:", len(ch["channel2"]))
            # print("         Channel 1 traces: ", ch["channel2"])
            output.append(ch["channel1"])
            output.append(ch["channel2"])

    return output, target_timestamp - closest_two[0][0], target_timestamp - closest_two[1][0], closest_two[0][1], closest_two[1][1] # list of channel1, channel2 for the 2 closest timestamps, and time difference between closest MD trace and the timestamp we drew ; return the filename so that we know in which one to find the best time


def extract_noise_from_t0_duIdlist(InputDataFiles, target_timestamp, target_duid): # seems to be the location of the data files, timestamp in second, duid that we are looking for (should be feb)
    #now target_duid is a list of the target_duid, so that we read the file only once per t0 time
    #print("started looking for the file")
    # paras
    headerLength = 256
    DAQHeaderLength = 12

    DUid_in_DAQHeader_Pos = 8
    ChannelLengthPos = 64
    GPSTimePos = 96
    ADC1Pos = 516

    #print(target_duid)

    #subfiles = os.listdir(InputDataFile)

    InputDataFiles.sort()
    #print("number of files in the directory : ", len(InputDataFiles))

    # *******************************************************
    # Step 1: collect all the timestamps' positions of the DU 
    # *******************************************************
    records_dict = {du: [] for du in target_duid}  # Dictionary to hold records for each DU
    out = []

    for i, filename in enumerate(InputDataFiles):

        #with open(os.path.join(InputDataFile, filename), 'rb') as f:
        with open(filename, 'rb') as f:
            loopLength = 0

            while True:
                f.seek(headerLength + loopLength)
                a = f.read(4)
                if a == b"":
                    break

                dataLength, = struct.unpack('I', a)

                # 读取 DU id
                f.seek(headerLength + DUid_in_DAQHeader_Pos + loopLength)
                du = f.read(4)
                duID, = struct.unpack('I', du)
                #print("DU ID : ", duID)

                if duID in records_dict:

                    # 读取时间
                    f.seek(headerLength + DAQHeaderLength + GPSTimePos + loopLength)
                    c = f.read(7)
                    year, month, day, hour, minute, second = struct.unpack('1H5B', c)

                    if year == 0 or month == 0 or day == 0:
                        loopLength += dataLength
                        continue

                    dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
                    ts = int(dt.timestamp())

                    # get the filename
                    records_dict[duID].append((ts, filename, loopLength))

                loopLength += dataLength

    # *****************************************
    # Step 2: find the most 2 closet timestamps
    # *****************************************
    for du in target_duid:
        records = records_dict[du]
        if len(records) < 2:
            #print("No enough records found for this DU : ", target_duid, InputDataFiles)
            out.append((None, 0.0, 0.0, "failed", "failed"))
            continue

        records.sort(key=lambda x: abs(x[0] - target_timestamp))
        closest_two = records[:2]

        #print("Closest timestamps:")
        #for r in closest_two:
            #print(r[0])

        # ============================================
        # Step 3: read the taces' info correspondingly
        # ============================================
        result = {du: {}}

        close_enough = True

        for ts, filename, offset in closest_two:

            with open(filename, 'rb') as f:

                # channel length
                f.seek(headerLength + DAQHeaderLength + ChannelLengthPos + offset)
                b = f.read(8)
                ch1_len, ch2_len, ch3_len, ch4_len = struct.unpack('4H', b)

                channelLength = ch1_len * 2

                # ===== Channel X =====
                f.seek(headerLength + DAQHeaderLength + ADC1Pos + offset + channelLength)
                ch1_data = []
                for _ in range(ch1_len):
                    m = f.read(2)
                    if len(m) < 2:
                        print("Unexpected end of file while reading channel data.", filename)
                        if len(ch1_data) < 512:
                            #fill it with 0
                            ch1_data.extend([0] * (512 - len(ch1_data)))
                        break
                    adc, = struct.unpack('h', m)
                    ch1_data.append(adc)

                channelLength += ch1_len * 2

                # ===== Channel Y =====
                f.seek(headerLength + DAQHeaderLength + ADC1Pos + offset + channelLength)
                ch2_data = []
                for _ in range(ch2_len):
                    m = f.read(2)
                    if len(m) < 2:
                        print("Unexpected end of file while reading channel data.", filename)
                        if len(ch2_data) < 512:
                            #fill it with 0
                            ch2_data.extend([0] * (512 - len(ch2_data)))
                        break
                    adc, = struct.unpack('h', m)
                    ch2_data.append(adc)

                # ====== Save data ======
                result[du][ts] = {
                    "channel1": ch1_data,
                    "channel2": ch2_data
                }


        # *******************************
        # Output of the results
        # *******************************
        #print("\nFinal result:\n")

        output = []

        for du, data in result.items():
            #print("DU:", du)
            for ts, ch in data.items():
                #print("  Timestamp:", ts)
                #print("    Channel1 length:", len(ch["channel1"]))
                # print("         Channel 1 traces: ", ch["channel1"])
                #print("    Channel2 length:", len(ch["channel2"]))
                # print("         Channel 1 traces: ", ch["channel2"])
                output.append(ch["channel1"])
                output.append(ch["channel2"])

        out.append((output, target_timestamp - closest_two[0][0], target_timestamp - closest_two[1][0], closest_two[0][1], closest_two[1][1])) # list of channel1, channel2 for the 2 closest timestamps, and time difference between closest MD trace and the timestamp we drew ; return the filename so that we know in which one to find the best time

    return out

"""
def add_noise_to_simulations(MD_data_name, event_number_md, run_md, tadc_trace, T1_idx, channel, du_id):
    # from a MD data file, and simulated event, given the channel and the DU that triggered, I cut the trace of the simulation, and add noise on top of it, 
    # I cut the traces of the simulations (from 1024 to 512, it’s not the sampling that changes but the duriation of the trace),
    # so that the 1st T1 crossing is at the same position as the one we find in the CD data ; I don’t know this position for now, say it’s 126 for all of them
    # in order to load the trace of the simulation I need to know the simulation number oand the event number in the simulation file
    #T1 idx is the first crossing of the th1
    #channel is 0 for x and 1 for y
    # let’s ay that du_id is the one from the data files, not from the simulation files


    #print("in add noise to simulations : ", np.shape(tadc_trace), T1_idx)
    tadc_trace = tadc_trace[T1_idx - 126 : T1_idx + 386] # 386 = 512-126 # should check the shape of the trace, it should be 512 ; cut the trace to go back to 512 points

    dict_duID_to_febID = generate_correspondance_duID_to_febID() # I could build this dictionary only onec in the main function
    feb_du_id = dict_duID_to_febID[du_id] 

    md_data = dh.DataFile("/sps/grand/data/gp80/GrandRoot/" + MD_data_name)
    md_adc_tree = md_data.tadc
    md_adc_tree.get_event(event_number_md, run_md)
    du_ids = np.array(md_adc_tree.du_id)

    #md_trace = np.array(md_adc_tree.trace_ch)[np.where(du_ids == feb_du_id)[0][0]][channel] # id from the data files, command I should use, assuming I was given the correct file
    md_trace = np.array(md_adc_tree.trace_ch)[0][channel] # assume this is the correct du, but this is going to change

    #print(np.shape(tadc_trace), np.shape(md_trace))
    tadc_trace = tadc_trace + md_trace*0 # hopefully they have the same shape

    md_data.close()
    md_adc_tree.close_file()

    return tadc_trace
"""



def judge_trigger_true_events(list_list_MD_files, sim_number, t0): # list_list_MD_files : a list of list of MD files, each sublist correspond to a t0 ; sim_number : number of the simulation we are processing


    data_directory = f"/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}" # ensure correct formating 
    with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/test_sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", "r") as f:
        first_judge_data = json.load(f)


    run_numbers = [row["fixed"][0] for row in first_judge_data] # select the run numbers of the events that passed the first trigger
    event_numbers = [row["fixed"][1] for row in first_judge_data] # select the event numbers of the events that passed the first trigger

    triggering_events = [row["triggering_events"] for row in first_judge_data] 
    #print(triggering_events)


    ### Read GRAND root data
    d_input = dh.DataDirectory(data_directory)
    tadc_l1, tshower_l0 = d_input.tadc_l1, d_input.tshower_l0

    dict_duID_to_febID = generate_correspondance_duID_to_febID() # this is to be able to use the FLT0 parameters for each du, which are given in febID, but we want from duIDs
    dict_febID_to_duID = generate_correspondance_febID_to_duID() # this is to be able to use the FLT0 parameters for each du, which are given in febID, but we want from duIDs

    f_sample = 500e6 # Hz, ADC sampling rate
    t_res = int(1. / f_sample * 1.e9) # ns, ADC time resolution

    iteration = 0


    for row, event_number,run_number in zip(first_judge_data, event_numbers, run_numbers):


        #print("our event number is : ", event_number)


        if row["fixed"][-1] in [0, 1]: # never ran it
            row["fixed"].append(t0[iteration]) # add t0 to the event
        else: # we already run it
            row["fixed"][-1] = t0[iteration] # we already ran the code for this simulation, we just need to update the value of t0
        #print("Iteration ", iteration, "out of ", len(event_numbers))
        trig_chnl_list = []
        killed_md_list = []
        killed_Tquiet_list = []

        tadc_l1.get_event(event_number, run_number)
        tshower_l0.get_event(event_number, run_number)
        
            
        du_ids = [event[0] for event in triggering_events[iteration]] # du_ids are the dus involved in the triggering in the simulation data
        du_ids_sims = tadc_l1.du_id # all the due involved in the simulation file
        #print(triggering_events[iteration], iteration)
        if len(du_ids) == 0:
            row["triggering_events_2"] = trig_chnl_list 
            row["killed_md"] = killed_md_list
            row["killed_Tquiet"] = killed_Tquiet_list
            iteration += 1
            continue


        #T1_idx = [event[3] for event in triggering_events[iteration]]
        channels = [event[1] for event in triggering_events[iteration]]
        channels_number = [0 if ch == "X" else 1 for ch in channels] # switch from "X" and "Y" to 0 and 1, to be able to use them as indices

                    
        rel_trace_start_time = calculate_relative_trace_start_time(tshower_l0, tadc_l1, t_res) # relative to the t_shower_core, and we get a different time for each DU (not each channel)


        previous_du = -1 # thing is because I did by channel, many DUs are here twice, so that when I iterate over them I can overshoot the shape of the trace
        #idx_adc_trace = -1
        feb_du_ids = [dict_duID_to_febID[du_id_n] for du_id_n in du_ids] 
        out_extract_noise = extract_noise_from_t0_duIdlist(list_list_MD_files[iteration], target_timestamp = t0[iteration], target_duid = feb_du_ids)
    
        # Start DU loop

        for du_idx, du_id_n in enumerate(du_ids): #assume du_ids is ordered, which is the case here

            #if du_id_n > previous_du: # we update the idx of the trace only if we see a new du
                #idx_adc_trace += 1
                #previous_du = du_id_n

            idx_adc_trace = du_ids_sims.index(du_id_n) # we need to find the index of the du in the simulation file, in order to be able to get the correct trace

            #if T1_idx[du_id_n] > 524: #we are supposed to have at least 500ns (ie Tperiod) after the first T1 crossign # I need to investigate that (why do I have a first T1 crossign at such high value ?)
                #continue

            #print("I came here")
            #tadc_trace = tadc_l1.trace_ch
            #print("shape of the tadc trace and type : ", np.shape(tadc_trace), type(tadc_trace), type(tadc_trace[0][0])) # output is shape of the tadc trace and type :  (196, 3, 1024) <class 'grand.dataio.descriptors.StdVectorList'> <class 'list'>

            tadc_trace = tadc_l1.trace_ch
            #print(channels_number, du_id_n, du_ids)
            #print("du_id_n : ", du_id_n, "du_ids : ", du_ids, "du_idx : ", du_idx, "event number : ", event_number, "run number : ", run_number)
            tadc_trace = np.array(tadc_trace[idx_adc_trace][channels_number[du_idx]]) # select the channel that triggered for this du # should ensure that things are in the same order 
            #print("shape of the adc trace : ", np.shape(tadc_trace))
            MD_traces, delta_t1, delta_t2, file1, file2 = out_extract_noise[du_idx]


            if type(row["triggering_events"][du_idx][-1]) == str: #ie we already ran this code
                row["triggering_events"][du_idx][-2] = file1.replace("raw", "GrandRoot").replace(".bin", ".root") # update the filenames from which we extracted the noise
                row["triggering_events"][du_idx][-1] = file2.replace("raw", "GrandRoot").replace(".bin", ".root")
            else:
                row["triggering_events"][du_idx].append(file1.replace("raw", "GrandRoot").replace(".bin", ".root")) # add the filenames from which we extracted the noise
                row["triggering_events"][du_idx].append(file2.replace("raw", "GrandRoot").replace(".bin", ".root"))



            if type(MD_traces) is list: # if not list then we just skip the rest and write an empty list

                #real_tadc_trace = add_noise_to_simulations(MD_data_name, event_number_md, run_md, tadc_trace, T1_idx[du_id_n], channels_number[du_id_n]+1, du_ids[du_id_n])

                first_part_real_tadc_trace = tadc_trace[:len(MD_traces[channels_number[du_idx]])] + MD_traces[channels_number[du_idx]] # the MD noise (trace) is twice smaller than the traces in the simulations
                if len(tadc_trace) == len(MD_traces[channels_number[du_idx]]):
                    real_tadc_trace = first_part_real_tadc_trace
                    print("We have a 1024 trace here, index is :", iteration)
                else:
                    #print(du_id_n, channels_number[du_idx], len(MD_traces))
                    real_tadc_trace = np.concatenate((first_part_real_tadc_trace, tadc_trace[len(MD_traces[channels_number[du_idx]]):] + MD_traces[channels_number[du_idx]+2])) 

                rel_start_time = rel_trace_start_time[idx_adc_trace] # select the time for our DU of interest

                ### Filtering of the traces
                ### The following lines applies
                ### 1. notch filter with a notch frequency of 39 MHz, &
                ### 2. FIR filter only passing the signals below 115 MHz
                ### the order does matter, and this one is the best according to Lech (26/08/2026)
                tadc_trace_filt = notch_filter(real_tadc_trace, 39e6, 0.9, f_sample)

                tadc_trace_filt = filter_traces_bandpass(tadc_trace_filt, coeff_file='/sps/grand/cprevotat/grand/grand/grand/exposure/lowpass115MHz.txt')

                """
                if du_id_n == 33:

                    plt.figure()
                    plt.plot([i for i in range(0, len(tadc_trace))], tadc_trace, label = "Original trace")
                    plt.plot([i for i in range(0, len(tadc_trace_filt))], tadc_trace_filt, label = "Filtered trace")
                    plt.xlabel("Time (ns)")
                    plt.ylabel("ADC counts")
                    plt.title(f"DU {du_id_n} channel {channels[du_idx]} trace, event {event_number}")
                    plt.legend()
                    plt.show()
                """

                ### Discuss the FLT0 trigger in the channel level
                ### The function "trigger_FLT0" is used (developed by M. Guelfand),
                ### but here we use a wrap function which gives a relative trigger time(s) of the DU of interest.
                feb_id_n = feb_du_ids[du_idx] # this is to be able to use the FLT0 parameters for each du, which are given in febID, but we want from duIDs

                FLT0_trig_params = get_FLT0_trigger_parameters_du_level(FLT0_trig_params_file, feb_id_n, channels[du_idx])

                #print("FLT0 trigger parameters for du ", du_id_n, " channel ", channels[du_idx], " : ", FLT0_trig_params)

                #print(feb_id_n)
                #FLT0_trig_time, first_idx_T1 = get_FLT0_trigger_time(tadc_trace_filt, FLT0_trig_params, rel_start_time, t_res, for_efficiency_T1_idx=True) # t_res : ADC time resolution
                FLT0_trig_time, killer = get_FLT0_trigger_time(tadc_trace_filt, FLT0_trig_params, rel_start_time, t_res) # killer : to know if the trace was killed by Tquiet or not ; FLT0_trig_time is in ns
                #print("FLT0 trigger time(s) for du ", du_id_n, " channel ", channels[du_idx], " : ", FLT0_trig_time)
                #print(FLT0_trig_time, first_idx_T1)
                if len(FLT0_trig_time) == 0:
                    killed_Tquiet_list.append([int(du_id_n), str(channels[du_idx]), str(killer)]) # list the du_ids and channels for which we had an issue in the Tquiet so we couldn’t build the traces with noise
                    continue
                elif len(FLT0_trig_time) > 1:
                    print("Warning : more than one trigger time found for this DU and this channel, we take the first one") # this didn’t trigger for file 84
                    print(FLT0_trig_time)
                    FLT0_trig_time = [FLT0_trig_time[0]] # take the first one, but we should investigate why we have more than one trigger time

                trig_chnl_list.append([int(du_id_n), str(channels[du_idx]), round(delta_t1, 4), round(delta_t2, 4), int(FLT0_trig_time[0])]) # list the du_ids that triggered, delta_t1 is distance in time between t0 and closest MD_trace

            else:
                killed_md_list.append([int(du_id_n), str(channels[du_idx])]) # list the du_ids and channels for which we had an issue in the MD data so we couldn’t build the traces with noise


        row["triggering_events_2"] = trig_chnl_list
        row["killed_md"] = killed_md_list
        row["killed_Tquiet"] = killed_Tquiet_list

        iteration += 1

    d_input.close()
                
    with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/test_sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", "w") as f:
        json.dump(first_judge_data, f, indent = 2) # I’d say that indent is useless, only for readability


    return 


def look_at_results(Oma = False):
    print("Running look_at_results()")

    simulation_numbers = [i for i in range(0, 150)] 
    count_all = 0
    if Oma == False:

        with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_0084_th30.json", "r") as f:
            data = json.load(f)

        possibly_triggering_dus = []
        triggering_dus = []
        energies = []
        zeniths = []
        killed_md = []
        md_cut = []
        for row in data:
            energy = row["fixed"][3]
            energies.append(10**energy)
            zenith = row["fixed"][4]
            zeniths.append(zenith)
            #print(row["triggering_events"], row["triggering_events_2"])
            if len(row["triggering_events"]) > 0:
                antennas = [liste[0] for liste in row["triggering_events"]] 
                unique_antennas = np.unique(antennas)
                possibly_triggering_dus.append(len(unique_antennas)) # consider the du level, not the channel level
            else:
                possibly_triggering_dus.append(0)

            if len(row["triggering_events_2"]) > 0:
                antennas = []
                cut_antennas_md = []
                for liste in row["triggering_events_2"]:
                    if np.abs(liste[2]) > 10 or np.abs(liste[3]) > 10: # if the closest MD trace is more than 10 seconds away from the t0, we consider that the event should be discarded
                        cut_antennas_md.append(liste[0])
                    else:
                        antennas.append(liste[0])
                
                unique_antennas = np.unique(antennas)
                cut_antennas_md = np.unique(cut_antennas_md)

                triggering_dus.append(max(0, len(unique_antennas))) # consider the du level, not the channel level
                md_cut.append(max(0, len(cut_antennas_md)))
            else:
                triggering_dus.append(0)
                md_cut.append(0)


            if len(row["killed_md"]) > 0:
                antennas = [liste[0] for liste in row["killed_md"]] 
                unique_antennas = np.unique(antennas)
                killed_md.append(max(0, len(unique_antennas))) # consider the du level, not the channel level
            else:
                killed_md.append(0)

        energies = np.array(energies)
        zeniths = np.array(zeniths)
        print(np.shape(energies), np.shape(zeniths), np.shape(possibly_triggering_dus), np.shape(triggering_dus), np.shape(killed_md), np.shape(md_cut))
        print(possibly_triggering_dus)

        plt.figure()
        plt.plot(energies, possibly_triggering_dus, "o", label = "Possibly triggering DUs", markersize = 5)
        plt.plot(energies, triggering_dus, "x", label = "Triggering DUs", markersize = 5)
        plt.plot(energies, killed_md, "s", label = "Killed MD", markersize = 5)
        plt.plot(energies, md_cut, "d", label = "Cut by MD", markersize = 5)
        plt.axhline(y = 5, color = "red", label = "5 DUs")
        plt.xlabel("Energy (GeV)")
        plt.ylabel("Number of triggering DUs")
        plt.xscale("log")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(zeniths, possibly_triggering_dus, "o", label = "Possibly triggering DUs", markersize = 5)
        plt.plot(zeniths, triggering_dus, "x", label = "Triggering DUs", markersize = 5)
        plt.plot(zeniths, killed_md, "s", label = "Killed_MD", markersize = 5)
        plt.plot(zeniths, md_cut, "d", label = "Cut by MD", markersize = 5)
        plt.axhline(y = 5, color = "red", label = "5 DUs")
        plt.xlabel("Zenith angle (degrees)")
        plt.ylabel("Number of triggering DUs")
        plt.legend()
        plt.show()

    else:

        simulated_DUs = []
        alive_DUs = []
        triggered_DUs = []
        saved_DUs = []
        energies = []
        zeniths = []

        dict_duID_to_febID = generate_correspondance_duID_to_febID()
        DUs_ids = running_DUs()

        #print("DUs_ids before switching to febIDs : ", DUs_ids)
        #DUs_ids = [dict_duID_to_febID[du_id] for du_id in DUs_ids] # switch from duIDs to febIDs, which are the ones used in the MD data files
        DUs_ids = DUs_ids.tolist()
        print("DUs_ids : ", DUs_ids)
        complete_DUs_id = [i for i in range(int(np.max(DUs_ids))+1)]
        DUs_ids = complete_DUs_id 
        #du_status = np.zeros((len(data), len(DUs_ids))) # then 0 if we didn’t check the status, -1 if off, 1 if on

        du_status = np.zeros((100*len(simulation_numbers), int(np.max(DUs_ids))+1))

        for sim_number in simulation_numbers:
            #print(f"Processing simulation number: {sim_number}")
            with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", "r") as f:
                data = json.load(f)

            for count, row in enumerate(data):
                energies.append(10**row["fixed"][3])
                zeniths.append(row["fixed"][4])

                if len(row["triggering_events"]) > 0:
                    antennas = [liste[0] for liste in row["triggering_events"]] 
                    unique_antennas = np.unique(antennas)
                    simulated_DUs.append(len(unique_antennas))
                else:
                    simulated_DUs.append(0)
                    unique_antennas = [] 
                if len(row["killed_md"]) > 0:
                    antennas = [liste[0] for liste in row["killed_md"]] 
                    unique_killed_antennas = np.unique(antennas)
                    alive_DUs.append(len(unique_antennas) - len(unique_killed_antennas)) # consider the du level, not the channel level

                else:
                    alive_DUs.append(len(unique_antennas))
                    unique_killed_antennas = []

                for du in unique_killed_antennas:
                    #print(" I came here for du ", du)
                    if du in DUs_ids:
                        #print(" I found du ", du)
                        du_status[count_all][DUs_ids.index(du)] = -1 # we mark the du as off for this event
                for du in unique_antennas:
                    if du in DUs_ids and du not in unique_killed_antennas:
                        du_status[count_all][DUs_ids.index(du)] = 1 # we mark the du as on for this 
                        
                #print(du_status[count])

                if len(row["triggering_events_2"]) > 0:
                    antennas = []
                    saved_dus = []
                    for liste in row["triggering_events_2"]:
                        if (np.abs(liste[2]) < 10) and (np.abs(liste[3]) < 10): # if the closest MD trace is more than 10 seconds away from the t0, we consider that the event should be discarded
                            saved_dus.append(liste[0])
                        antennas.append(liste[0])
                    
                    unique_antennas = np.unique(antennas)
                    unique_saved_dus = np.unique(saved_dus)

                    triggered_DUs.append(len(unique_antennas)) # consider the du level, not the channel level
                    saved_DUs.append(len(unique_saved_dus))
                else:
                    triggered_DUs.append(0)
                    saved_DUs.append(0)

                count_all += 1

        energies = np.array(energies)        
        plt.figure()
        plt.imshow(du_status, aspect = "auto", cmap = "bwr", vmin = -1, vmax = 1, origin = "lower")
        plt.colorbar(label = "DU status (-1: off, 0: not checked, 1: on)")
        plt.xlabel("DU ID")
        plt.ylabel("Event number")
        plt.title("Status of the DUs for each event")
        plt.show()

        
        plt.figure()
        #for i in range(len(energies)):
            #plt.plot([energies[i], energies[i]], [saved_DUs[i], simulated_DUs[i]], color = "gray", alpha = 0.5)
        plt.plot(energies*1e9, simulated_DUs, "o", label = "Simulated DUs", markersize = 5)
        plt.plot(energies*1e9, alive_DUs, "x", label = "Alive DUs", markersize = 5)
        plt.plot(energies*1e9, triggered_DUs, "s", label = "Triggered DUs", markersize = 5)
        plt.plot(energies*1e9, saved_DUs, "d", label = "Saved DUs", markersize = 5)
        plt.axhline(y = 5, color = "red", label = "5 DUs")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Number of triggering DUs")
        plt.xscale("log")
        plt.legend()
        plt.show()

        plt.figure()
        #plt.plot(zeniths, simulated_DUs, "o", label = "Simulated DUs", markersize = 5)
        #plt.plot(zeniths, alive_DUs, "x", label = "Alive DUs", markersize = 5)
        #plt.plot(zeniths, triggered_DUs, "s", label = "Triggered DUs", markersize = 5)
        plt.plot(zeniths, saved_DUs, "d", label = "Saved DUs", markersize = 5)
        plt.axhline(y = 5, color = "red", label = "5 DUs")
        plt.xlabel("Zenith angle (degrees)")
        plt.ylabel("Number of triggering DUs")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()


        energy_bins = np.logspace(8, 11, 12)
        mid_energy_bins = np.sqrt(energy_bins[:-1] * energy_bins[1:])
        zenith_bins = np.linspace(65, 88, 12)
        mid_zenith_bins = (zenith_bins[:-1] + zenith_bins[1:]) / 2


        mean_simulated_DUs_energy = []
        mean_alive_DUs_energy = []
        mean_triggered_DUs_energy = []
        mean_saved_DUs_energy = []

        print("len", len(energies), len(simulated_DUs), len(alive_DUs), len(triggered_DUs), len(saved_DUs))
        idx = np.where(np.array(simulated_DUs) > 0)[0]
        simulated_DUs = np.array(simulated_DUs)[idx]
        alive_DUs = np.array(alive_DUs)[idx]
        triggered_DUs = np.array(triggered_DUs)[idx]
        saved_DUs = np.array(saved_DUs)[idx]
        energies = np.array(energies)[idx]
        zeniths = np.array(zeniths)[idx]

        for i in range(len(energy_bins)-1):
            mean_simulated_DUs_energy.append(np.mean([simulated_DUs[j] for j in range(len(energies)) if energies[j] >= energy_bins[i] and energies[j] < energy_bins[i+1]]))
            mean_alive_DUs_energy.append(np.mean([alive_DUs[j] for j in range(len(energies)) if energies[j] >= energy_bins[i] and energies[j] < energy_bins[i+1]]))
            mean_triggered_DUs_energy.append(np.mean([triggered_DUs[j] for j in range(len(energies)) if energies[j] >= energy_bins[i] and energies[j] < energy_bins[i+1]]))
            mean_saved_DUs_energy.append(np.mean([saved_DUs[j] for j in range(len(energies)) if energies[j] >= energy_bins[i] and energies[j] < energy_bins[i+1]]))

        mean_simulated_DUs_energy = np.array(mean_simulated_DUs_energy)
        mean_alive_DUs_energy = np.array(mean_alive_DUs_energy)
        mean_triggered_DUs_energy = np.array(mean_triggered_DUs_energy)
        mean_saved_DUs_energy = np.array(mean_saved_DUs_energy)

        plt.figure()
        width = np.diff(energy_bins)*1e9*0.8
        plt.bar(mid_energy_bins*1e9, mean_simulated_DUs_energy / mean_simulated_DUs_energy, width=width, label = "Simulated DUs")
        plt.bar(mid_energy_bins*1e9, mean_alive_DUs_energy / mean_simulated_DUs_energy, width=width, label = "Alive DUs")
        plt.bar(mid_energy_bins*1e9, mean_triggered_DUs_energy / mean_simulated_DUs_energy , width=width, label = "Triggered DUs")
        plt.bar(mid_energy_bins*1e9, mean_saved_DUs_energy / mean_simulated_DUs_energy , width=width, label = "Saved DUs")
        #plt.axhline(y = 5, color = "red", label = "5 DUs")
        plt.xlabel("Energy [eV]")
        plt.ylabel("Mean number of DUs per bin")
        plt.xscale("log")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title("Threshold at 30 ADC counts")
        plt.show()

        mean_simulated_DUs_zenith = []
        mean_alive_DUs_zenith = []
        mean_triggered_DUs_zenith = []
        mean_saved_DUs_zenith = []
        for i in range(len(zenith_bins)-1):
            mean_simulated_DUs_zenith.append(np.mean([simulated_DUs[j] for j in range(len(zeniths)) if zeniths[j] >= zenith_bins[i] and zeniths[j] < zenith_bins[i+1]]))
            mean_alive_DUs_zenith.append(np.mean([alive_DUs[j] for j in range(len(zeniths)) if zeniths[j] >= zenith_bins[i] and zeniths[j] < zenith_bins[i+1]]))
            mean_triggered_DUs_zenith.append(np.mean([triggered_DUs[j] for j in range(len(zeniths)) if zeniths[j] >= zenith_bins[i] and zeniths[j] < zenith_bins[i+1]]))
            mean_saved_DUs_zenith.append(np.mean([saved_DUs[j] for j in range(len(zeniths)) if zeniths[j] >= zenith_bins[i] and zeniths[j] < zenith_bins[i+1]]))

        mean_simulated_DUs_zenith = np.array(mean_simulated_DUs_zenith)
        mean_alive_DUs_zenith = np.array(mean_alive_DUs_zenith)
        mean_triggered_DUs_zenith = np.array(mean_triggered_DUs_zenith)
        mean_saved_DUs_zenith = np.array(mean_saved_DUs_zenith)


        plt.figure()
        width = np.diff(zenith_bins)*0.8
        plt.bar(mid_zenith_bins, mean_simulated_DUs_zenith / mean_simulated_DUs_zenith, label = "Simulated DUs", width = width)
        plt.bar(mid_zenith_bins, mean_alive_DUs_zenith / mean_simulated_DUs_zenith, label = "Alive DUs", width = width)
        plt.bar(mid_zenith_bins, mean_triggered_DUs_zenith / mean_simulated_DUs_zenith, label = "Triggered DUs", width = width)
        plt.bar(mid_zenith_bins, mean_saved_DUs_zenith / mean_simulated_DUs_zenith, label = "Saved DUs", width = width)
        #plt.axhline(y = 5, color = "red", label = "5 DUs")
        plt.xlabel("Zenith angle (degrees)")
        plt.ylabel("Mean number of DUs per bin")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    


def plot_triggering_event(list_list_MD_files, sim_number, t0):


    data_directory = f"/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_{sim_number:04d}" # ensure correct formating 
    with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_{sim_number:04d}.json", "r") as f:
        first_judge_data = json.load(f)

    triggering_events = [row["triggering_events_2"] for row in first_judge_data] 

    run_numbers = [row["fixed"][0] for row in first_judge_data] # select the run numbers of the events that passed the first trigger
    event_numbers = [row["fixed"][1] for row in first_judge_data] # select the event numbers of the events that passed the first trigger


    ### Read GRAND root data
    data_directory = f"/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_{sim_number:04d}" # ensure correct formating 
    d_input = dh.DataDirectory(data_directory)
    trun_l1, tadc_l1, tshower_l0 = d_input.trun_l1, d_input.tadc_l1, d_input.tshower_l0

    previous_run = None

    dict_duID_to_febID = generate_correspondance_duID_to_febID() # this is to be able to use the FLT0 parameters for each du, which are given in febID, but we want from duIDs
    dict_febID_to_duID = generate_correspondance_febID_to_duID() # this is to be able to use the FLT0 parameters for each du, which are given in febID, but we want from duIDs


    iteration = 0


    for row, event_number,run_number in zip(first_judge_data, event_numbers, run_numbers):
        print("Iteration ", iteration, "out of ", len(event_numbers))


        tadc_l1.get_event(event_number, run_number)
        tshower_l0.get_event(event_number, run_number)
        
            
        du_ids = [event[0] for event in triggering_events[iteration]] # du_ids are the dus involved in the triggering in the simulation data
        #print(triggering_events[iteration], iteration)
        if len(du_ids) == 0:

            iteration += 1
            continue

        channels = [event[1] for event in triggering_events[iteration]]
        channels_number = [0 if ch == "X" else 1 for ch in channels] # switch from "X" and "Y" to 0 and 1, to be able to use them as indices

        if previous_run != run_number:
            trun_l1.get_run(run_number)
            previous_run = run_number
                    
        rel_trace_start_time = calculate_relative_trace_start_time(tshower_l0, tadc_l1, t_res) # this is not used to determine the trigger, only to get the time in the output


        previous_du = -1 # thing is because I did by channel, many DUs are here twice, so that when I iterate over them I can overshoot the shape of the trace
        idx_adc_trace = -1
    
        # Start DU loop
        for du_idx, du_id_n in enumerate(du_ids):

            if du_id_n > previous_du: # we update the idx of the trace only if we see a new du
                idx_adc_trace += 1
                previous_du = du_id_n


            tadc_trace = np.array(tadc_l1.trace_ch) 

            tadc_trace = tadc_trace[idx_adc_trace][channels_number[du_idx]] # select the channel that triggered for this du # should ensure that things are in the same order 
            MD_traces = extract_noise_from_t0(list_list_MD_files[iteration], target_timestamp = t0[iteration], target_duid = dict_duID_to_febID[du_id_n])

            if type(MD_traces) is list: # if not list then we just skip the rest and write an empty list


                first_part_real_tadc_trace = tadc_trace[:len(MD_traces[channels_number[du_idx]])] + MD_traces[channels_number[du_idx]] # the MD noise (trace) is twice smaller than the traces in the simulations
                if len(tadc_trace) == len(MD_traces[channels_number[du_idx]]):
                    real_tadc_trace = first_part_real_tadc_trace
                    print("We have a 1024 trace here, index is :", iteration)
                else:
                    real_tadc_trace = np.concatenate((first_part_real_tadc_trace, tadc_trace[len(MD_traces[channels_number[du_idx]]):] + MD_traces[channels_number[du_idx]+2])) 

    

    return 


def plot_antennas_event():

    dus_gp65 = running_DUs()
    print("We have ", len(dus_gp65), " DUs")

    sim_number = 105
    with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", "r") as f:
        first_judge_data = json.load(f)


    triggering_events = [row["triggering_events"] for row in first_judge_data] 
    event_numbers = np.array([row["fixed"][1] for row in first_judge_data])

    data_directory = f"/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}" # ensure correct formating 
    d_input = dh.DataDirectory(data_directory)
    trun, tadc_l1, tshower_l0 = d_input.trun, d_input.tadc_l1, d_input.tshower_l0


    idx_event = np.where(np.array([len(triggering_events[i]) for i in range(len(triggering_events))]) > 20)[0][0] 
    idx_event = np.where(event_numbers == 103696)[0][0] # this is to ensure that we are looking at the same event in the json file and in the root files, because some events have the same event number but different run numbers, so we need to make sure we are looking at the right one

    du_ids = [event[0] for event in triggering_events[idx_event]] # du_ids are the dus involved in the triggering in the simulation data
    unique_ids = np.unique(du_ids)
    print("du_ids : ", du_ids)
    print("unique_ids : ", unique_ids)

    energy = 10**(first_judge_data[idx_event]["fixed"][3])*1e-9
    zenith = first_judge_data[idx_event]["fixed"][4]

    killed_md = np.unique([liste[0] for liste in first_judge_data[idx_event]["killed_md"]])
    print("killed_md : ", killed_md)
    print("triggering_events_2 : ", first_judge_data[idx_event]["triggering_events_2"])

    du_issue_md = []
    triggering_dus_before_md = []
    for du_set in first_judge_data[idx_event]["triggering_events_2"]:
        triggering_dus_before_md.append(du_set[0])
        if np.abs(du_set[2]) > 10 or np.abs(du_set[3]) > 10: # if the closest MD trace is more than 10 seconds away from the t0, we consider that the event should be discarded
            du_issue_md.append(du_set[0])

    du_issue_md = np.unique(du_issue_md)
    triggering_dus_before_md = np.unique(triggering_dus_before_md)
    print("du_issue_md : ", du_issue_md)
    print("triggering_dus_before_md : ", triggering_dus_before_md)

    event_number = event_numbers[idx_event]
    tadc_l1.get_event(event_number, 1)
    trun.get_run(1)

    du_ids_tadc = tadc_l1.du_id
    times = np.array(tadc_l1.du_seconds) + np.array(tadc_l1.du_nanoseconds) * 1e-9
    times = times[np.isin(du_ids_tadc, unique_ids)] # select only the times of the DUs that triggered
    dict_id_time = {unique_ids[i] : times[i] for i in range(len(unique_ids))} # build a dictionary to associate the du_id to the time of the trace, for the DUs that triggered

    list_dus_run_gp300 = trun.du_id
    position_DUs = trun.du_xyz
    list_dus_run = np.array(list_dus_run_gp300)[np.isin(list_dus_run_gp300, dus_gp65)] # select only the DUs that are in gp65
    position_DUs = np.array(position_DUs)[np.isin(list_dus_run_gp300, dus_gp65)] /1000
    dict_id_position = {list_dus_run[i] : position_DUs[i] for i in range(len(list_dus_run))} # build a dictionary to associate the du_id to the position of the DU, for all the DUs in the run

    print("list_dus_run: ", list_dus_run)
    plt.figure()

    x_triggered = []
    y_triggered = []
    time_triggered = []
    for i in range(len(list_dus_run)):
        if list_dus_run[i] in unique_ids:
            x_triggered.append(position_DUs[i][0])
            y_triggered.append(position_DUs[i][1])
            time_triggered.append(dict_id_time[list_dus_run[i]])

    x_triggered = np.array(x_triggered)
    y_triggered = np.array(y_triggered)
    time_triggered = (np.array(time_triggered) - np.min(time_triggered))*1e6 
    x = np.array([p[0] for p in position_DUs])
    y = np.array([p[1] for p in position_DUs])
    plt.scatter(-y, x, color = "k", s = 10, label = "All DUs")
    sc = plt.scatter(-y_triggered, x_triggered, c = time_triggered, cmap = "viridis", s = 500, label = "Simulated DUs", marker = ".")
    plt.colorbar(sc, label = "Trigger time (μs)")

    plt.xlabel("X [km]")
    plt.ylabel("Y [km]")
    plt.axis("equal")
    plt.legend()
    plt.title(f"E = {energy:.2e} EeV, Zenith = {zenith:.2f} °")
    plt.show()


    plt.figure()

    x = np.array([dict_id_position[list_dus_run[i]][0] for i in range(len(list_dus_run))])
    y = np.array([dict_id_position[list_dus_run[i]][1] for i in range(len(list_dus_run))])
    plt.scatter(-y, x, color = "k", s = 10, label = "All DUs")
    sc = plt.scatter(-y_triggered, x_triggered, c = time_triggered, cmap = "viridis", s = 500, label = "Simulated DUs", marker = ".")
    x_killed_md = np.array([dict_id_position[du][0] for du in killed_md])
    y_killed_md = np.array([dict_id_position[du][1] for du in killed_md])
    plt.scatter(-y_killed_md, x_killed_md, color = "red", marker = "x", s = 100, label = "Killed MD")

    plt.colorbar(sc, label = "Trigger time (μs)")

    plt.xlabel("X [km]")
    plt.ylabel("Y [km]")
    plt.axis("equal")
    plt.legend()
    plt.title(f"E = {energy:.2e} EeV, Zenith = {zenith:.2f} °")
    plt.show()


    plt.figure()

    x_trigger_before_md = np.array([dict_id_position[list_dus_run[i]][0] for i in range(len(list_dus_run)) if list_dus_run[i] in triggering_dus_before_md])
    y_trigger_before_md = np.array([dict_id_position[list_dus_run[i]][1] for i in range(len(list_dus_run)) if list_dus_run[i] in triggering_dus_before_md])
    times = np.array([dict_id_time[list_dus_run[i]] for i in range(len(list_dus_run)) if list_dus_run[i] in triggering_dus_before_md])
    times = (times - np.min(times))*1e6 # we put the minimum time to 0 to have a better color scale in the plot, but it doesn’t change anything to the results since we are only interested in the relative times between the DUs

    x_issue_md = np.array([dict_id_position[list_dus_run[i]][0] for i in range(len(list_dus_run)) if list_dus_run[i] in du_issue_md])
    y_issue_md = np.array([dict_id_position[list_dus_run[i]][1] for i in range(len(list_dus_run)) if list_dus_run[i] in du_issue_md])

    plt.scatter(-y, x, color = "k", s = 10, label = "All DUs")

    sc = plt.scatter(-y_trigger_before_md, x_trigger_before_md, c = times, cmap = "viridis", s = 500, label = "T1 triggered", marker = ".")
    #plt.scatter(x_issue_md, y_issue_md, color = "red", marker = "x", s = 50, label = "Issue MD")

    plt.colorbar(sc, label = "Trigger time (μs)")
    plt.xlabel("X [km]")
    plt.ylabel("Y [km]")
    plt.legend()
    plt.title(f"E = {energy:.2e} EeV, Zenith = {zenith:.2f} °")
    plt.show()


    plt.figure()

    x_trigger_after_md = np.array([dict_id_position[list_dus_run[i]][0] for i in range(len(list_dus_run)) if list_dus_run[i] in triggering_dus_before_md and list_dus_run[i] not in du_issue_md])
    y_trigger_after_md = np.array([dict_id_position[list_dus_run[i]][1] for i in range(len(list_dus_run)) if list_dus_run[i] in triggering_dus_before_md and list_dus_run[i] not in du_issue_md])
    times = np.array([dict_id_time[list_dus_run[i]] for i in range(len(list_dus_run)) if list_dus_run[i] in triggering_dus_before_md and list_dus_run[i] not in du_issue_md])
    times = (times - np.min(times))*1e6 

    plt.scatter(-y, x, color = "k", s = 10, label = "All DUs")
    sc = plt.scatter(-y_trigger_after_md, x_trigger_after_md, c = times, cmap = "viridis", s = 500, label = "Saved", marker = ".")

    plt.colorbar(sc, label = "Trigger time (μs)")
    plt.xlabel("X [km]")
    plt.ylabel("Y [km]")
    plt.legend()
    plt.title(f"E = {energy:.2e} EeV, Zenith = {zenith:.2f} °")
    plt.show()
            
    


    d_input.close()
    return

def plot_trace_event(sim_number, event_number):

    f_sample = 500e6 # Hz, ADC sampling rate



    with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", "r") as f:
        first_judge_data = json.load(f)

    data_fixed = [row["fixed"] for row in first_judge_data]
    event_numbers = [row[1] for row in data_fixed]
    #print(event_numbers)
    idx_our_event_in_json = np.where(np.array(event_numbers) == event_number)[0][0]
    print("idx_our_event_in_json : ", idx_our_event_in_json)

    triggering_data = first_judge_data[idx_our_event_in_json]["triggering_events_2"] #_2 
    triggering_du_ids = np.array([event[0] for event in triggering_data])
    triggering_channels = np.array([event[1] for event in triggering_data])
    channels_number = np.array([0 if ch == "X" else 1 for ch in triggering_channels]) # switch from "X" and "Y" to 0 and 1, to be able to use them as indices
    print("triggering_du_ids : ", triggering_du_ids)
    print("triggering_channels : ", triggering_channels)
    print("energy of the event is ", 10**(first_judge_data[idx_our_event_in_json]["fixed"][3])*1e-9, "EeV")

    data_directory = f"/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}" # ensure correct formating 
    d_input = dh.DataDirectory(data_directory)
    #print(d_input.print())

    trun, tadc = d_input.trun, d_input.tadc_l1

    tadc.get_event(event_number, 1)
    du_id = np.array(tadc.du_id)
    idx_triggering_events = 0

    for i in range(len(triggering_du_ids)):
        #if triggering_du_ids[i] != 33:
            #continue

        #if i>0:break

        idx_du = np.where(du_id == triggering_du_ids[i])[0][0] # we take the first du that triggered, but we could also look at the others
        while first_judge_data[idx_our_event_in_json]["triggering_events"][idx_triggering_events][0] != triggering_du_ids[i]:
            idx_triggering_events += 1

        print("which du are we plotting :", idx_du, du_id[idx_du])
        tadc_trace = np.array(tadc.trace_ch) # we focus only on the 2 (3) first columns : x, y and z
        print(np.shape(du_id), np.shape(tadc_trace))
        #tadc_trace_X = tadc_trace[idx_du][0] # trace of the first du
        tadc_trace_X = tadc_trace[idx_du][0]
        tadc_trace_Y = tadc_trace[idx_du][1]
        print("idx max traces : ", np.argmax(tadc_trace_X), np.argmax(tadc_trace_Y))
        

        tadc_X_filt = notch_filter(tadc_trace_X, 39e6, 0.9, f_sample)
        tadc_Y_filt = notch_filter(tadc_trace_Y, 39e6, 0.9, f_sample)

        tadc_X_filt = filter_traces_bandpass(tadc_X_filt, coeff_file='/sps/grand/cprevotat/grand/grand/grand/exposure/lowpass115MHz.txt')
        tadc_Y_filt = filter_traces_bandpass(tadc_Y_filt, coeff_file='/sps/grand/cprevotat/grand/grand/grand/exposure/lowpass115MHz.txt')

        print(np.max(tadc_X_filt), np.max(tadc_Y_filt))
        
        x = np.arange(len(tadc_trace_X))
        plt.figure()
        #plt.plot(x, tadc_trace_Y, alpha = 0.5, label = "Y")
        #plt.plot(x, tadc_trace_X, alpha = 1, label = "X")
        if channels_number[i] == 0:
            plt.plot(2*x, tadc_X_filt, alpha = 1, label = "Sim", linewidth = 0.8)
        else:
            plt.plot(2*x, tadc_Y_filt, alpha = 1, label = "Sim", linewidth = 0.8)
        plt.xlabel("Time [ns]")
        plt.ylabel("ADC counts")
        plt.legend()
        plt.show()

        # now we add the noise from the MD traces
        MD_file_t0_1 = first_judge_data[idx_our_event_in_json]["triggering_events"][idx_triggering_events][-2].replace("GrandRoot", "raw").replace(".root", ".bin")
        print(first_judge_data[idx_our_event_in_json]["triggering_events"][idx_triggering_events][-1])
        MD_file_t0_2 = first_judge_data[idx_our_event_in_json]["triggering_events"][idx_triggering_events][-1].replace("GrandRoot", "raw").replace(".root", ".bin")

        if MD_file_t0_1 == "failed" or MD_file_t0_2 == "failed":
            print("No MD trace found for this event, we skip the noise addition")
            continue

        if MD_file_t0_1 != MD_file_t0_2:
            list_files = [MD_file_t0_1, MD_file_t0_2]
        else:
            list_files = [MD_file_t0_1]

        t0 = first_judge_data[idx_our_event_in_json]["fixed"][-1] # this is the t0 of the event, which we will use to extract the noise from the MD traces at the right time
        print("t0 : ", t0, "MD_file_t0_1 : ", MD_file_t0_1, du_id[idx_du])
        MD_traces, _, _, _, _ = extract_noise_from_t0([MD_file_t0_1], target_timestamp = t0, target_duid = dict_duID_to_febID[du_id[idx_du]])

        if type(MD_traces) is list:

            our_tadc_trace = tadc_trace_X if channels_number[i] == 0 else tadc_trace_Y

            first_part_real_tadc_trace = our_tadc_trace[:len(MD_traces[channels_number[i]])] + MD_traces[channels_number[i]] # the MD noise (trace) is twice smaller than the traces in the simulations
            real_tadc_trace = np.concatenate((first_part_real_tadc_trace, our_tadc_trace[len(MD_traces[channels_number[i]]):] + MD_traces[channels_number[i]+2]))

            real_tadc_trace_filt = notch_filter(real_tadc_trace, 39e6, 0.9, f_sample)
            real_tadc_filt = filter_traces_bandpass(real_tadc_trace_filt, coeff_file='/sps/grand/cprevotat/grand/grand/grand/exposure/lowpass115MHz.txt')


            
            x = np.arange(len(real_tadc_trace))

            plt.figure()
            plt.plot(2*x, np.concatenate((MD_traces[channels_number[i]], MD_traces[channels_number[i]+2])), alpha = 1, label = "MD trace", linewidth = 0.8)
            plt.xlabel("Time [ns]")
            plt.ylabel("ADC counts")
            plt.ylim(1.1 * np.min(real_tadc_filt), 1.1 * np.max(real_tadc_filt))
            plt.title(f"Channel {channels_number[i]}")
            plt.legend()
            plt.show()

            plt.figure()
            #plt.plot(x, real_tadc_trace, alpha = 0.5, label = "Real trace with noise")
            #plt.plot(0.5*x * f_sample, tadc_X_filt, alpha = 0.5, label = "Sim X", linewidth = 0.8)
            #plt.plot(0.5*x * f_sample, tadc_Y_filt, alpha = 0.5, label = "Sim Y", linewidth = 0.8)
            plt.plot(2*x, real_tadc_filt, alpha = 1, label = "Sim + MD noise", linewidth = 0.8)
            plt.xlabel("Time [ns]")
            plt.ylabel("ADC counts")
            plt.ylim(1.1 * np.min(real_tadc_filt), 1.1 * np.max(real_tadc_filt))
            plt.title(f"Filtered ADC Channel {channels_number[i]}")
            plt.legend()
            plt.show()

            """"""
            fft_real_tadc_trace = np.fft.fft(our_tadc_trace)
            frequencies = 1e-6*np.fft.fftfreq(len(our_tadc_trace), d=1/f_sample)
            plt.figure()
            plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_real_tadc_trace)[:len(frequencies)//2], alpha = 1, label = "FFT_sim", linewidth = 0.8)
            plt.xscale("log")
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("Amplitude")
            plt.title(f"Channel {channels_number[i]}")
            plt.legend()
            plt.show()
            """"""

            


    d_input.close()

    return


def plot_2D_efficiency():
    print("Running plot_2D_efficiency()")
    E_grid = np.logspace(8, 11, 10) # in GeV
    zenith_grid = np.linspace(65, 88, 10)

    mid_E_grid = np.sqrt(E_grid[:-1] * E_grid[1:])
    mid_zenith_grid = (zenith_grid[:-1] + zenith_grid[1:]) / 2

    energies_simulated_events = []
    zeniths_simulated_events = []

    energies_triggered_events = []
    zeniths_triggered_events = []
    deleted_events = 0

    energies_all = []
    zeniths_all = []

    sim_numbers = [i for i in range(0, 150)]
    for sim_number in sim_numbers:
        with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", "r") as f:
            data = json.load(f)


        for count, row in enumerate(data):    sim_numbers = [i for i in range(0, 150)]
    for sim_number in sim_numbers:
        with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", "r") as f:
            data = json.load(f)


        for count, row in enumerate(data):
            if row["fixed"][9] == 0: # ie our event was not simulated because it was not detected in the very simple trigger I use in judge_trigger_event_du_level_channel_level.py
                deleted_events += 1
                #continue
            energy = 10**(row["fixed"][3])
            zenith = row["fixed"][4]
            detected = False

            energies_all.append(energy)
            zeniths_all.append(zenith)
            
            if len(row["triggering_events_2"]) > 0: 
                antennas = []
                saved_dus = []
                for liste in row["triggering_events_2"]:
                    if (np.abs(liste[2]) < 10) and (np.abs(liste[3]) < 10): # if the closest MD trace is more than 10 seconds away from the t0, we consider that the event should be discarded
                        saved_dus.append(liste[0])

                unique_saved_dus = np.unique(saved_dus)

                if len(unique_saved_dus) >= 5: 
                    detected = True

            if len(row["triggering_events"]) > 0: # here we look at antennas for which max adc trace > 15, and that for at least 5 antennas
                antennas = []
                for liste in row["triggering_events"]:
                    antennas.append(liste[0])
                unique_antennas = np.unique(antennas)
                if len(unique_antennas) >= 5:
                    energies_simulated_events.append(energy)
                    zeniths_simulated_events.append(zenith)

            
            if detected:
                energies_triggered_events.append(energy)
                zeniths_triggered_events.append(zenith)

    print("number of events, triggered, simulated and all : ", len(energies_triggered_events), len(energies_simulated_events), len(energies_all))
    print(f"Deleted events: {deleted_events}")
    histogram_simulated, _, _ = np.histogram2d(energies_simulated_events, zeniths_simulated_events, bins=[E_grid, zenith_grid])
    histogram_triggered, _, _ = np.histogram2d(energies_triggered_events, zeniths_triggered_events, bins=[E_grid, zenith_grid])
    histogram_tot, _, _ = np.histogram2d(energies_all, zeniths_all, bins = [E_grid, zenith_grid])
    print(np.sum(histogram_triggered, axis = (0, 1)))

    plt.figure()
    #plt.plot(mid_E_grid*1e9, np.sum(histogram_simulated, axis=1), label = "Simulated events")  
    #plt.plot(mid_E_grid*1e9, np.sum(histogram_triggered, axis=1) / np.sum(histogram_simulated, axis = 1), label = "Triggered events")
    plt.plot(mid_E_grid*1e9, histogram_triggered[:, 5] / histogram_simulated[:, 5], label = "Efficiency")
    plt.xscale('log')
    plt.xlabel('Energy [eV]')
    plt.ylabel('Number of events')
    plt.legend()
    plt.show()


    ratio = histogram_triggered/ histogram_simulated # should be defined in all our grid
    ratio = histogram_triggered / histogram_tot

    plt.figure()
    #plt.imshow(ratio.T, origin='lower', extent=[E_grid[0]*1e9, E_grid[-1]*1e9, zenith_grid[5], zenith_grid[7]], aspect='auto', cmap='rainbow')
    plt.pcolor(mid_E_grid*1e9, mid_zenith_grid, ratio.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Efficiency')
    plt.xscale('log')
    plt.xlabel('Energy [eV]')
    plt.ylabel('Zenith angle [degrees]')
    print("ratio : ", np.nanmax(ratio))

    plt.show()


def plot_Tquiet_cut():
    print("Running plot_Tquiet_cut()")

    convert_channel_number = {"X": 0, "Y": 1}

    array_DUs = running_DUs()

    sim_numbers = [i for i in range(0, 150)]
    killer = []
    triggering_events = []
    tag = None
    for sim_number in sim_numbers:
        print("sim_number : ", sim_number)
        with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", "r") as f:
            data = json.load(f)

        path_to_file = f"/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}" # ensure correct formating
        d_input = dh.DataDirectory(path_to_file) # read the root file ?

        tshower_l0=d_input.tshower_l0
        tadc_l1 = d_input.tadc_l1

        for count, row in enumerate(data):
            if "killed Tquiet" in row:
                print(row)
                tag = "killed Tquiet"
                row["killed_Tquiet"] = row.pop("killed Tquiet")
            else:
                tag = "not Tquiet"
            if len(row["killed_Tquiet"]) > 0:
                killer.extend([x[-1] for x in row["killed_Tquiet"]])
                triggering_events.extend([x[-1] for x in row["triggering_events"]])


            if tag == "killed Tquiet":
                print("killed Tquiet in row")
                t0 = row["fixed"][-3]
                MD_filename = row["triggering_event"][-1].replace("GrandRoot", "raw").replace(".root", ".bin")
                for idx in range(len(row["killed_Tquiet"])):
                    event_number = row["fixed"][1]
                    run_number = row["fixed"][0] # I think it’s constant and equal to 1
                    tadc_l1.get_event(event_number, run_number)
                    du_ids = tadc_l1.du_id
                    du_id_killed, channel = row["killed_Tquiet"][idx][0], convert_channel_number[row["killed_Tquiet"][idx][1]]
                    print("du_id_killed : ", du_id_killed, "channel : ", channel)
                    #print("let’s compare du ids : ", du_ids)
                    #print("array_DUs : ", array_DUs)
                    #print(len(du_ids), len(array_DUs))
                    #print(array_DUs, du_id_killed)
                    for (idx_du, du_id) in enumerate(du_ids):
                        if du_id not in array_DUs:
                            continue
                        if du_id == du_id_killed:
                            tadc_trace = np.array(tadc_l1.trace_ch[idx_du][channel])
                            #print("hello : ", idx_du, np.shape(tadc_l1.trace_ch))
                            if MD_filename == "failed":
                                print("failed", du_id_killed, channel)
                                continue
                            MD_traces, delta_t1, delta_t2, file1, file2 = extract_noise_from_t0([MD_filename], target_timestamp = t0, target_duid = dict_duID_to_febID[du_id_killed])
                            if type(MD_traces) is list: # if not list then we just skip the rest and write an empty list

                                #real_tadc_trace = add_noise_to_simulations(MD_data_name, event_number_md, run_md, tadc_trace, T1_idx[du_id_n], channels_number[du_id_n]+1, du_ids[du_id_n])

                                first_part_real_tadc_trace = tadc_trace[:len(MD_traces[channel])] + MD_traces[channel] # the MD noise (trace) is twice smaller than the traces in the simulations
                                if len(tadc_trace) == len(MD_traces[channel]):
                                    real_tadc_trace = first_part_real_tadc_trace
                                else:
                                    #print(du_id_n, channels_number[du_idx], len(MD_traces))
                                    real_tadc_trace = np.concatenate((first_part_real_tadc_trace, tadc_trace[len(MD_traces[channel]):] + MD_traces[channel+2])) 

                            else: continue


                            tadc_trace_filt = notch_filter(real_tadc_trace, 39e6, 0.9, f_sample)
                            tadc_trace_filt = filter_traces_bandpass(tadc_trace_filt, coeff_file='/sps/grand/cprevotat/grand/grand/grand/exposure/lowpass115MHz.txt')
                            #print("filtered trace : ", tadc_trace_filt)

                            #print("values to plot : ", f_sample * np.arange(len(tadc_trace_filt)) * 1e-9, tadc_trace_filt)
                            FLT0_trig_params = get_FLT0_trigger_parameters(FLT0_trig_params_file)
                            print("testing the trigger : ", FLT0.trigger_FLT0(tadc_trace_filt, FLT0_trig_params))

                            plt.figure()
                            plt.plot(f_sample * np.arange(len(tadc_trace_filt)) * 1e-9, tadc_trace, label = "tadc trace")
                            plt.plot(f_sample * np.arange(len(MD_traces[0]))*1e-9, MD_traces[0], alpha = 1, label = "MD trace", linewidth = 0.8)
                            plt.plot(f_sample * np.arange(len(tadc_trace_filt))*1e-9, tadc_trace_filt, alpha = 1, label = "Sim + MD noise", linewidth = 0.8)
                            plt.axhline(y = 60, color = "green", linestyle = "--", label = "Th1")
                            plt.axvline(x = f_sample * 100*1e-9, label = "IDX 100")
                            plt.xlabel("Time [ns]")
                            plt.ylabel("ADC counts")
                            plt.title(f"DU {du_id_killed}, channel {channel}, sim number {sim_number}, event number {event_number}")
                            plt.legend()
                            plt.show()

                            

                        









    
    print(len(killer), len(triggering_events))
    print(killer)

    count_Tquiet = 0
    print(np.unique(killer))
    for serial_killer in np.unique(killer):
        print(serial_killer, np.sum(np.array(killer) == serial_killer))
    #for i in range(len(killer)):
        #if killer[i] == "Tquiet":
            #count_Tquiet += 1
    #print(count_Tquiet, len(killer))
    print(len(killer))

        
    



def high_E_T1(sim_number = 42):
    sim_numbers = [i for i in range(0, 150)]
    FLT0_trig_params_file = "/sps/grand/cprevotat/grand/grand/grand/exposure/dict_trig_params_fir.csv"

    killer_dict = {"Tquiet" : [], "NoT1" : [], "Tsepmax" : [], "nc_crossings" : [], "Passed" : []}

    # I want to get all the filenames that contain "MD" in the directory
    directory = "/sps/grand/data/gp80/GrandRoot/2026/05/"
    list_MD_filenames = [f for f in os.listdir(directory) if "MD" in f]


    for sim_number in sim_numbers:
        print("Dealing with sim_number : ", sim_number)
        data_directory = f"/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}"
        d_input = dh.DataDirectory(data_directory) # read the root file
        tadc = d_input.tadc_l1

        with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", "r") as f:
            data = json.load(f)

        idx_json = 0
        for count, row in enumerate(data):
            
            energy = 10**(row["fixed"][3])
            zenith = row["fixed"][4]

            if (energy > 10**(9.)) & (zenith > 70) & row["fixed"][9] == 1: # ie our event was simulated because it was detected in the very simple trigger I use in judge_trigger_event_du_level_channel_level.py
                # 3s per event in here
                ## get some noise
                random_file = directory + str(np.random.choice(list_MD_filenames))
                file_root = dh.DataFile(random_file)
                adc_tree = file_root.tadc
                md_du_ids = adc_tree.du_id
                N_du_ids = len(md_du_ids)
                event_list = adc_tree.get_list_of_events()
                N_events = np.shape(event_list)[0]
                random_event = np.random.randint(0, N_events)
                adc_tree.get_event(event_list[random_event][0], event_list[random_event][1])

                md_trace = np.array(adc_tree.trace_ch)
                md_trace_X = md_trace[np.random.randint(0, N_du_ids)][0]
                md_trace_Y = md_trace[np.random.randint(0, N_du_ids)][1]

                if np.shape(md_trace_X)[0] == 512:
                    md_trace_X = np.concatenate((md_trace_X, md_trace_X[::-1])) # so that the noise is continuous

                if np.shape(md_trace_Y)[0] == 512:
                    md_trace_Y = np.concatenate((md_trace_Y, md_trace_Y[::-1]))

                file_root.close()
                adc_tree.close_file()

                """
                print("sim number and count number : ", sim_number, count)
                print("fixed : ", row["fixed"])
                print("triggering events : ", row["triggering_events"])
                print("triggering events 2 : ", row["triggering_events_2"])
                print("killed md : ", row["killed_md"])
                if "killed Tquiet" in row:
                    row["killed_Tquiet"] = row.pop("killed Tquiet")
                    print("killed Tquiet : ", row["killed_Tquiet"])
                """
                event_number = row["fixed"][1]
                run_number = row["fixed"][0]

                tadc.get_event(event_number, run_number)
                du_ids = tadc.du_id
                for (idx_du, du_id) in enumerate(du_ids):
                    tadc_trace_X = np.array(tadc.trace_ch[idx_du][0])
                    tadc_trace_Y = np.array(tadc.trace_ch[idx_du][1])


                    tadc_trace_X = tadc_trace_X + md_trace_X
                    tadc_trace_Y = tadc_trace_Y + md_trace_Y

                    tadc_X_filt = notch_filter(tadc_trace_X, 39e6, 0.9, f_sample)
                    tadc_Y_filt = notch_filter(tadc_trace_Y, 39e6, 0.9, f_sample)

                    tadc_X_filt = filter_traces_bandpass(tadc_X_filt, coeff_file='/sps/grand/cprevotat/grand/grand/grand/exposure/lowpass115MHz.txt')
                    tadc_Y_filt = filter_traces_bandpass(tadc_Y_filt, coeff_file='/sps/grand/cprevotat/grand/grand/grand/exposure/lowpass115MHz.txt')

                    if tadc_X_filt.max() > 30:
                        trigger_parameters = get_FLT0_trigger_parameters(FLT0_trig_params_file)
                        _, _, _, killer = FLT0.trigger_FLT0(tadc_X_filt, trigger_parameters)
                        killer_dict[killer].append(energy)

                    if tadc_Y_filt.max() > 30:
                        trigger_parameters = get_FLT0_trigger_parameters(FLT0_trig_params_file)
                        _, _, _, killer = FLT0.trigger_FLT0(tadc_Y_filt, trigger_parameters)
                        killer_dict[killer].append(int(energy)) 

            idx_json += 1

        d_input.close()
        tadc.close_file()

        print(len(killer_dict["Passed"]))

    with open("/sps/grand/cprevotat/grand/efficiency/dict_killer_high_E.json", "w") as f:
        json.dump(killer_dict, f, indent=4)

    return

def plot_killer_high_E():
    with open("/sps/grand/cprevotat/grand/efficiency/dict_killer_high_E.json", "r") as f:
        killer_dict = json.load(f)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    count = 0
    for key in killer_dict.keys():
        axs[count // 3, count % 3].hist(killer_dict[key], bins = np.logspace(9, 11, 20), alpha = 0.5, label = key)
        axs[count // 3, count % 3].set_xscale("log")
        axs[count // 3, count % 3].set_xlabel("Energy [GeV]")
        axs[count // 3, count % 3].set_ylabel("Number of events")
        axs[count // 3, count % 3].legend()
        count += 1

    plt.tight_layout()
    plt.show()
    



def T3_trigger(datafile, N_required_DUs = 5, want_all = False): # data should be a json file, this function should check that we have more than N_dus triggering on the event, and implement the causality cut
# returns a list of bool, True if triggered, False otherwise, and then we can extract the parameters that we want


    rtk_positions = np.genfromtxt("./efficiency/gp65_rtksort.txt").T
    for i in range(len(rtk_positions[0])):
        if rtk_positions[0][i] in dict_febID_to_duID:
            #print(rtk_positions[0][i])
            rtk_positions[0][i] = dict_febID_to_duID[rtk_positions[0][i]] # so that it starts at 0 as in the simulations
            #print(rtk_positions[0][i])
        else:
            print("Warning: feb_id ", rtk_positions[0][i], " not found in the dictionary. It will be ignored.")
            rtk_positions[0][i] = -1  # Mark as invalid
    dict_DuID_to_RotatedPosition = {int(rtk_positions[0][i]): adapted_CausalityCut_Kwen.rotate_and_shift_coordinates(np.array([rtk_positions[1][i], rtk_positions[2][i], rtk_positions[3][i]])) for i in range(len(rtk_positions[0])) if rtk_positions[0][i] != -1}

    with open(datafile, "r") as f:
        data = json.load(f)

    data_fixed = [row["fixed"] for row in data]
    data_triggered2 = [row["triggering_events_2"] for row in data]
    list_bools = []
    list_dus_TimeNs = []

    causality_cut = 0
    N_events = 0

    """
    for i in range(0, len(data_fixed)):
        #if data_fixed[i][4] > max_zenith_degrees: # if the zenith is too large, we don’t consider the event, even if it triggered, because it’s not in the range of our simulations
            #continue
        energies_all = np.concatenate([energies_all, np.array([10**data_fixed[i][3]])])
        zenith_all = np.concatenate([zenith_all, np.array([data_fixed[i][4]])])
    """
    for i in range(0, len(data_triggered2)):
        #if data_fixed[i][4] > max_zenith_degrees: # if the zenith is too large, we don’t consider the event, even if it triggered, because it’s not in the range of our simulations
            #continue

        N_triggered_dus = 0
        du_ids = []
        times_ns = []



        if len(data_triggered2[i]) < N_required_DUs: # we cannot trigger in this case, no need to go further
            list_bools.append(False)
            continue

        for j in range(len(data_triggered2[i])):

            if (np.abs(data_triggered2[i][j][2]) < 10) & (np.abs(data_triggered2[i][j][3]) < 10): # distance in time to closest MD event, should be smaller than 10s (ie we should not miss any MD file)
                du_ids.append(data_triggered2[i][j][0])
                times_ns.append(data_triggered2[i][j][4]) # last element is time ns, first T1 crossing with respect to shower core time (whatever this last element is, as long as it’s the same for all dus of an event)


        if len(np.unique(du_ids)) >= N_required_DUs: 

            N_events += 1

            # here we implement the causality cut
            list_dus_TimeNs.append([(du_ids[i], times_ns[i]) for i in range(len(du_ids))])
            detection_status = adapted_CausalityCut_Kwen.optimized_read_matching_times_graph(dict_DuID_to_RotatedPosition, list_dus_TimeNs[-1], min_detectors = N_required_DUs) # we can change the max_time_diff_ns if we want to be more or less strict

            if not detection_status:
                causality_cut += 1

            #if detection_status == False:
                #print(f"Event {i} failed the causality cut. list_dus_TimeNs : {list_dus_TimeNs}")

            #energies_triggered = np.concatenate([energies_triggered, np.array([10**data_fixed[i][3]])])
            #zenith_triggered = np.concatenate([zenith_triggered, np.array([data_fixed[i][4]])])
            #azimuth_triggered = np.concatenate([azimuth_triggered, np.array([data_fixed[i][5]])]) # don’t consider the cluster part for now
            list_bools.append(detection_status) # it is sufficient as long as we don’t have a more detailed T3 trigger implemented
        else:
            list_bools.append(False)

    if want_all:
        list_bools = [True for i in range(len(data_fixed))]  # if we want all the events, return True everywhere
    print(f"Out of {N_events} events with at least {N_required_DUs} triggered DUs, {causality_cut} failed the causality cut.")
    return data, list_bools



def core_position_triggered_events():

    # get the position of the antennas

    not_working_DUs = np.array([103, 109, 1010, 1011, 1012, 1013, 1014, 1016, 1017, 1018, 1029, 1030, 1033, 1034, 1039, 1055, 1056, 1086, 1089, 1090, 1092, 1093, 1094])
    dict = generate_correspondance_febID_to_duID()
    not_working_DUs = np.array([dict[du] for du in not_working_DUs])
    print("not_working_DUs : ", not_working_DUs)

    sim_name = "/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_0000"

    d_input = dh.DataDirectory(sim_name)
    trun_l1, tadc_l1, tshower_l0 = d_input.trun_l1, d_input.tadc_l1, d_input.tshower_l0
    #print(trun_l1)

    event_list = tadc_l1.get_list_of_events()
    #print(event_list, np.shape(event_list))
    nb_events = len(event_list)

    for event_number, run_number in event_list:

        trun_l1.get_run(run_number)
        print(trun_l1.du_id)

        liste_DUs = trun_l1.du_id
        position_DUs = trun_l1.du_xyz
        break

    print(liste_DUs)
    print(position_DUs)


    array_dus = running_DUs()
    print("running DUs : ", array_dus)

    sim_numbers = [i for i in range(0, 150)]
    positions_x = np.array([])
    positions_y = np.array([])
    energies = np.array([])
    for sim_number in sim_numbers:
        data, list_bools = T3_trigger(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", N_required_DUs = 5)
        data_fixed = [row["fixed"] for row in data]
        local_positions_x = np.array([-row[6] for row in data_fixed]) # using all elements, so that indices is computed the way it should, can be commented if we are not looking for a precise event
        local_positions_y = np.array([row[7] for row in data_fixed])
        indices = np.where((local_positions_x < -5500) & (local_positions_y < -10000))[0]
        positions_x = np.concatenate([positions_x, np.array([-row[6] for row in data_fixed])[list_bools]]) # keeping only triggering events
        positions_y = np.concatenate([positions_y, np.array([row[7] for row in data_fixed])[list_bools]])
        energies = np.concatenate([energies, 10**np.array([row[3] for row in data_fixed])[list_bools]])
        if indices.size > 0:
            if (np.isin(local_positions_x[indices[0]], positions_x)) and (np.isin(local_positions_y[indices[0]], positions_y)): # we have to select the event that triggered
                print("Our event has sim number ", sim_number, " and event number ", data_fixed[indices[0]][1], " and position x ", local_positions_x[indices[0]], " and position y ", local_positions_y[indices[0]])

    print(np.min(energies), np.max(energies))
    print("We have ", len(positions_x), " triggered events in total out of 14950 simulated events")
    plt.figure()
    # I want the color of the points to be proportional to the energy of the event, so I will use a colormap
    sc = plt.scatter(positions_x, positions_y, c = np.log10(energies), cmap = "rainbow", s = 3, label = "Core position of triggering events")
    plt.colorbar(sc, label = "Energy of triggering events [GeV]")
    #plt.scatter(positions_x, positions_y, s = 3, color = "red", label = "Core position of triggering events")
    for i in range(len(liste_DUs)):
        if (liste_DUs[i] in array_dus) and (liste_DUs[i] not in not_working_DUs):
            plt.scatter( -position_DUs[i][1], position_DUs[i][0], s = 20, color = "black", marker = "x")
            plt.annotate(liste_DUs[i], (-position_DUs[i][1], position_DUs[i][0]), fontsize = 14)

    plt.xlabel("x [m]", fontsize = 16)
    plt.ylabel("y [m]", fontsize = 16)
    #plt.ylim((-2200, 5000)) #2200, 5000
    #plt.xlim(-4400, 2400)
    plt.title("Core position of triggering events", fontsize = 16)
    plt.show()



    bins_X = np.linspace(-6500, 4400, 25)
    bins_Y = np.linspace(-11300, 5200, 25)
    mid_bins_X = (bins_X[:-1] + bins_X[1:]) / 2
    mid_bins_Y = (bins_Y[:-1] + bins_Y[1:]) / 2
    histogram = np.histogram2d(positions_x, positions_y, bins=[bins_X, bins_Y])[0]

    print("Total number of triggered events:", np.sum(histogram, axis=(0, 1)))
    plt.figure()
    plt.imshow(histogram.T, origin='lower', extent=[bins_X[0], bins_X[-1], bins_Y[0], bins_Y[-1]], aspect='auto', cmap='rainbow')
    plt.colorbar(label='Number of triggered events')
    #plt.scatter(positions_x, positions_y, s = 3, color = "red", label = "Core position of triggering events")
    for i in range(len(liste_DUs)):
        if liste_DUs[i] in array_dus and (liste_DUs[i] not in not_working_DUs):
            plt.scatter( -position_DUs[i][1], position_DUs[i][0], s = 20, color = "black", marker = "x")
            plt.annotate(liste_DUs[i], (-position_DUs[i][1], position_DUs[i][0]), fontsize = 14)

    plt.xlabel("x [m]", fontsize = 16)
    plt.ylabel("y [m]", fontsize = 16)
    #plt.ylim((-2200, 5000)) #2200, 5000
    #plt.xlim(-4400, 2400)
    plt.title("Core position of triggering events", fontsize = 16)
    plt.show()

    return


def investigate_du12():
    """
    investigate the reason why we have a bit less events around du12 (from the plot I sent 
    on commissioning efficiency in August 2026) : is it statistical, or DUs are not working
    properly at that time ?
    """

    bins_X = np.linspace(-6500, 4400, 25)
    bins_Y = np.linspace(-11300, 5200, 25)
    bins_E = np.logspace(8, 11, 15) # in GeV
    bins_zenith = np.linspace(65, 90, 20)
    mid_bins_X = (bins_X[:-1] + bins_X[1:]) / 2
    bin_sizeX = bins_X[1] - bins_X[0]
    mid_bins_Y = (bins_Y[:-1] + bins_Y[1:]) / 2
    bin_sizeY = bins_Y[1] - bins_Y[0]
    mid_bins_E = np.sqrt(bins_E[:-1] * bins_E[1:])
    mid_bins_zenith = (bins_zenith[:-1] + bins_zenith[1:]) / 2

    coordinates_bin = [250, 750] # in meters
    idx_x = np.argmin(np.abs(mid_bins_X - coordinates_bin[0]))
    idx_y = np.argmin(np.abs(mid_bins_Y - coordinates_bin[1]))


    sim_numbers = [i for i in range(0, 150)]
    for sim_number in sim_numbers:
        with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", "r") as f:
            print("sim_number : ", sim_number)
            data = json.load(f)
            for row in data:
                x = row["fixed"][6]
                y = row["fixed"][7]
                if x > mid_bins_X[idx_x] - bin_sizeX/2 and x < mid_bins_X[idx_x] + bin_sizeX/2 and y > mid_bins_Y[idx_y] - bin_sizeY/2 and y < mid_bins_Y[idx_y] + bin_sizeY/2:
                    print("\nWe found an event, here are the values : \n", row["fixed"], "\n")
                    print(row["triggering_events"], "\n")
                    print(row["triggering_events_2"], "\n")
                    print(row["killed_md"], "\n")
                    # I want to throw an exeption here, if this line returns an error, then do print(row["killed Tquiet"])
                    
                    try:
                        value = row["killed_Tquiet"]
                    except KeyError:
                        value = row["killed Tquiet"]

                    print(value, "\n")
                    



    energies = []
    zeniths = []
    x, y, z = [], [], []

    for sim_number in range(0, 150):
        f = np.genfromtxt(f"/sps/grand/cprevotat/grand/efficiency/out_extract_infos/out_sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}.txt", dtype = str).T
        energies += list((f[2]))
        zeniths += list(f[3])
        x += list(f[5])
        y += list(f[6])
        z += list(f[7])

    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    energies = np.array(energies).astype(float)
    zeniths = np.array(zeniths).astype(float)
    print(np.shape(x), np.shape(y), np.shape(energies), np.shape(zeniths))
    histogram = np.histogramdd((x, y, energies, zeniths), bins=[bins_X, bins_Y, bins_E, bins_zenith])[0]
    print("Coordinates of our bin : ", mid_bins_X[idx_x], mid_bins_Y[idx_y])
    our_bin = histogram[idx_x, idx_y, :, :]
    our_binleft = histogram[idx_x-1, idx_y, :, :]
    our_binright = histogram[idx_x+1, idx_y, :, :]
    print("Total number of events in our bin : ", np.sum(our_bin, axis = (0, 1)))
    print("Total number of events in left bin : ", np.sum(our_binleft, axis = (0, 1)))
    print("Total number of events in right bin : ", np.sum(our_binright, axis = (0, 1)))
    plt.figure()
    plt.plot(mid_bins_E, np.sum(our_bin, axis = 1), label = "All events")
    plt.plot(mid_bins_E, np.sum(our_binleft, axis = 1), label = "Left bin")
    plt.plot(mid_bins_E, np.sum(our_binright, axis = 1), label = "Right bin")
    plt.plot(mid_bins_E, np.sum(histogram, axis = (0, 1, 3)), label = "All events in all bins")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(mid_bins_zenith, np.sum(our_bin, axis = 0), label = "All events")
    plt.plot(mid_bins_zenith, np.sum(our_binleft, axis = 0), label = "Left bin")
    plt.plot(mid_bins_zenith, np.sum(our_binright, axis = 0), label = "Right bin")
    plt.plot(mid_bins_zenith, np.sum(histogram, axis = (0, 1, 2)), label = "All events in all bins")
    plt.legend()
    plt.show()

    return 

def compare_Coreas_Zhaires():
    sim_numbers_Coreas = [i for i in range(0, 150)]
    sim_numbers_Zhaires = [i for i in range(1, 14)]

    list_Coreas = []
    list_Zhaires = []

    for sim_number in sim_numbers_Coreas:
        list_Coreas.append(np.genfromtxt(f"/sps/grand/cprevotat/grand/efficiency/out_extract_infos/out_sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}.txt", dtype = str).T)

        if sim_number % 10 == 0:
            print("sim_number : ", sim_number)

    for sim_number in sim_numbers_Zhaires:
        list_Zhaires.append(np.genfromtxt(f"/sps/grand/cprevotat/grand/efficiency/out_extract_infos/out_sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-NJ_{sim_number:04d}.txt", dtype = str).T)

    data_Coreas = np.concatenate(list_Coreas, axis = 1)
    data_Zhaires = np.concatenate(list_Zhaires, axis = 1)
    data_Zhaires[1, :] = np.where(data_Zhaires[1, :] == "Fe^56", "5626.0", data_Zhaires[1, :])

    del list_Coreas, list_Zhaires
    gc.collect()

    data_Coreas = np.concatenate((data_Coreas[0:2], data_Coreas[3:]), axis = 0) # remove the DU_ID column, which is not a float
    data_Zhaires = np.concatenate((data_Zhaires[0:2], data_Zhaires[3:]), axis = 0) # remove the DU_ID column, which is not a float
    data_Coreas = data_Coreas.astype(float)
    data_Zhaires = data_Zhaires.astype(float)

    data_Coreas = data_Coreas[:, data_Coreas[2, :] <= np.max(data_Zhaires[2, :])] # remove events with energy bigger than Zhaires
    data_Zhaires = data_Zhaires[:, data_Zhaires[2, :] >= np.min(data_Coreas[2, :])] # remove events with energy < 10^7 GeV
    data_Zhaires = data_Zhaires[:, data_Zhaires[8, :] >= np.min(data_Coreas[8, :])] 

    mask_protons_Coreas = data_Coreas[1, :] == 14
    mask_protons_Zhaires = data_Zhaires[1, :] == 2212
    data_Coreas_protons = data_Coreas[:, mask_protons_Coreas]
    data_Zhaires_protons = data_Zhaires[:, mask_protons_Zhaires]

    print("Shape data Coreas, shape data Coreas protons : ", np.shape(data_Coreas), np.shape(data_Coreas_protons))
    print("Shape data Zhaires, shape data Zhaires protons : ", np.shape(data_Zhaires), np.shape(data_Zhaires_protons))

    data_Coreas_iron = data_Coreas[:, data_Coreas[1, :] == 5626.0]
    data_Zhaires_iron = data_Zhaires[:, data_Zhaires[1, :] == 5626.0]

    X_max_gram_Coreas_protons = data_Coreas_protons[7, :]
    X_max_gram_Zhaires_protons = data_Zhaires_protons[7, :]
    X_max_gram_Coreas_iron = data_Coreas_iron[7, :]
    X_max_gram_Zhaires_iron = data_Zhaires_iron[7, :]

    E_primary_Coreas_protons = data_Coreas_protons[2]
    E_primary_Zhaires_protons = data_Zhaires_protons[2]
    E_primary_Coreas_iron = data_Coreas_iron[2]
    E_primary_Zhaires_iron = data_Zhaires_iron[2]

    E_em_Coreas_protons = data_Coreas_protons[3]
    E_em_Zhaires_protons = data_Zhaires_protons[3]
    E_em_Coreas_iron = data_Coreas_iron[3]
    E_em_Zhaires_iron = data_Zhaires_iron[3]

    zenith_Coreas_protons = data_Coreas_protons[8]
    zenith_Zhaires_protons = data_Zhaires_protons[8]
    zenith_Coreas_iron = data_Coreas_iron[8]
    zenith_Zhaires_iron = data_Zhaires_iron[8]

    x_Coreas_protons = data_Coreas_protons[10]
    y_Coreas_protons = data_Coreas_protons[11]
    z_Coreas_protons = data_Coreas_protons[12]
    x_Zhaires_protons = data_Zhaires_protons[10]
    y_Zhaires_protons = data_Zhaires_protons[11]
    z_Zhaires_protons = data_Zhaires_protons[12]

    azimuth_Coreas_protons = data_Coreas_protons[9]
    azimuth_Zhaires_protons = data_Zhaires_protons[9]


    # in this part we look for two events in Coreas and Zhaires that would be similar in energy, Xmax, Eem, zenith and core position, maybe it doesn’t exist
    ratio = E_primary_Coreas_protons[:, None] / E_primary_Zhaires_protons[None, :]

    candidate_i, candidate_j = np.where((ratio > 0.95) & (ratio < 1.05))
    print(f"Found {len(candidate_i)} candidate pairs with similar energy")
    mask = (
    (X_max_gram_Coreas_protons[candidate_i] /
     X_max_gram_Zhaires_protons[candidate_j] > 0.95)
    &
    (X_max_gram_Coreas_protons[candidate_i] /
     X_max_gram_Zhaires_protons[candidate_j] < 1.05)
    &
    (E_em_Coreas_protons[candidate_i] /
     E_em_Zhaires_protons[candidate_j] > 0.95)
    &
    (E_em_Coreas_protons[candidate_i] /
     E_em_Zhaires_protons[candidate_j] < 1.05)
    &
    (zenith_Coreas_protons[candidate_i] /
     zenith_Zhaires_protons[candidate_j] > 0.95)
    &
    (zenith_Coreas_protons[candidate_i] /
     zenith_Zhaires_protons[candidate_j] < 1.05)
    &
    (np.abs(x_Coreas_protons[candidate_i] - x_Zhaires_protons[candidate_j]) < 50)
    &
    (np.abs(y_Coreas_protons[candidate_i] - y_Zhaires_protons[candidate_j]) < 50)
    &
    (np.abs(z_Coreas_protons[candidate_i] - z_Zhaires_protons[candidate_j]) < 50)
    )

    candidate_i = candidate_i[mask]
    candidate_j = candidate_j[mask]

    print(f"Found {len(candidate_i)} matching pairs")

    print("Candidate pairs (Coreas index, Zhaires index):")
    for i, j in zip(candidate_i, candidate_j):
        print(f"  ({i}, {j})")
        print(f"    Coreas: E={E_primary_Coreas_protons[i]}, Xmax={X_max_gram_Coreas_protons[i]}, E_em={E_em_Coreas_protons[i]}, zenith={zenith_Coreas_protons[i]}, azimuth={azimuth_Coreas_protons[i]}, core=({x_Coreas_protons[i]}, {y_Coreas_protons[i]}, {z_Coreas_protons[i]})")
        print(f"    Zhaires: E={E_primary_Zhaires_protons[j]}, Xmax={X_max_gram_Zhaires_protons[j]}, E_em={E_em_Zhaires_protons[j]}, zenith={zenith_Zhaires_protons[j]}, azimuth={azimuth_Zhaires_protons[j]}, core=({x_Zhaires_protons[j]}, {y_Zhaires_protons[j]}, {z_Zhaires_protons[j]})")


    """
    E_bins = np.logspace(7.5, 10, 30)
    mid_E_bins = np.sqrt(E_bins[:-1] * E_bins[1:])

    print("max energy Coreas and Zhaires : ", np.max(E_primary_Coreas_protons), np.max(E_primary_Zhaires_protons))

    hist_E_Coreas_protons = np.histogram(E_primary_Coreas_protons, bins = E_bins, density = True)[0]
    hist_E_Zhaires_protons = np.histogram(E_primary_Zhaires_protons, bins = E_bins, density = True)[0]
    hist_E_Coreas_iron = np.histogram(E_primary_Coreas_iron, bins = E_bins, density = True)[0]
    hist_E_Zhaires_iron = np.histogram(E_primary_Zhaires_iron, bins = E_bins, density = True)[0]

    plt.figure()
    plt.plot(mid_E_bins, hist_E_Coreas_protons * mid_E_bins, label = "Coreas protons", color = "red", ls = "--")
    plt.plot(mid_E_bins, hist_E_Zhaires_protons * mid_E_bins, label = "Zhaires protons", color = "red")
    plt.plot(mid_E_bins, hist_E_Coreas_iron * mid_E_bins, label = "Coreas iron", color = "blue", ls = "--")
    plt.plot(mid_E_bins, hist_E_Zhaires_iron * mid_E_bins, label = "Zhaires iron", color = "blue")
    plt.xscale("log")
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    bins_ratio = np.linspace(0.5, 1, 20)
    mid_bins_ratio = (bins_ratio[:-1] + bins_ratio[1:]) / 2

    hist_ratioE_Coreas_protons = np.histogram(E_em_Coreas_protons / E_primary_Coreas_protons, bins = bins_ratio, density = True)[0]
    hist_ratioE_Zhaires_protons = np.histogram(E_em_Zhaires_protons / E_primary_Zhaires_protons, bins = bins_ratio, density = True)[0]
    hist_ratioE_Coreas_iron = np.histogram(E_em_Coreas_iron / E_primary_Coreas_iron, bins = bins_ratio, density = True)[0]
    hist_ratioE_Zhaires_iron = np.histogram(E_em_Zhaires_iron / E_primary_Zhaires_iron, bins = bins_ratio, density = True)[0]

    plt.figure()
    plt.plot(mid_bins_ratio, hist_ratioE_Coreas_protons, label = "Coreas protons", color = "red", ls = "--")
    plt.plot(mid_bins_ratio, hist_ratioE_Zhaires_protons, label = "Zhaires protons", color = "red")
    plt.plot(mid_bins_ratio, hist_ratioE_Coreas_iron, label = "Coreas iron", color = "blue", ls = "--")
    plt.plot(mid_bins_ratio, hist_ratioE_Zhaires_iron, label = "Zhaires iron", color = "blue")

    plt.xlabel(r"$\frac{E_{em}}{E_{primary}}$")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    bins_Xmax = np.linspace(550, 1000, 20)
    mid_bins_Xmax = (bins_Xmax[:-1] + bins_Xmax[1:]) / 2
    bins_zenith = np.linspace(65, 88, 20)
    mid_bins_zenith = (bins_zenith[:-1] + bins_zenith[1:]) / 2

    hist_Xmax_zenith_Coreas_protons = np.histogram2d(X_max_gram_Coreas_protons, zenith_Coreas_protons, bins = [bins_Xmax, bins_zenith], density = True)[0]
    hist_Xmax_zenith_Zhaires_protons = np.histogram2d(X_max_gram_Zhaires_protons, zenith_Zhaires_protons, bins = [bins_Xmax, bins_zenith], density = True, weights = 1 / np.log10(np.cos(np.radians(zenith_Zhaires_protons))))[0]
    hist_Xmax_zenith_Coreas_iron = np.histogram2d(X_max_gram_Coreas_iron, zenith_Coreas_iron, bins = [bins_Xmax, bins_zenith], density = True)[0]
    hist_Xmax_zenith_Zhaires_iron = np.histogram2d(X_max_gram_Zhaires_iron, zenith_Zhaires_iron, bins = [bins_Xmax, bins_zenith], density = True, weights = 1 / np.log10(np.cos(np.radians(zenith_Zhaires_iron))))[0]

    plt.figure()
    plt.plot(mid_bins_zenith, np.sum(hist_Xmax_zenith_Coreas_protons, axis = 0), label = "Coreas protons", color = "red", ls = "--")
    plt.plot(mid_bins_zenith, np.sum(hist_Xmax_zenith_Zhaires_protons, axis = 0), label = "Zhaires protons", color = "red")
    plt.plot(mid_bins_zenith, np.sum(hist_Xmax_zenith_Coreas_iron, axis = 0), label = "Coreas iron", color = "blue", ls = "--")
    plt.plot(mid_bins_zenith, np.sum(hist_Xmax_zenith_Zhaires_iron, axis = 0), label = "Zhaires iron", color = "blue")
    plt.xlabel("Zenith [°]")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    im1 = axs[0, 0].imshow(hist_Xmax_zenith_Coreas_protons.T, origin='lower', extent=[bins_Xmax[0], bins_Xmax[-1], bins_zenith[0], bins_zenith[-1]], aspect='auto', cmap='rainbow')
    axs[0, 0].set_title("Coreas protons")
    axs[0, 0].set_xlabel("Xmax [g/cm²]")
    axs[0, 0].set_ylabel("Zenith [°]")
    fig.colorbar(im1, ax=axs[0, 0])

    im2 = axs[0, 1].imshow(hist_Xmax_zenith_Zhaires_protons.T, origin='lower', extent=[bins_Xmax[0], bins_Xmax[-1], bins_zenith[0], bins_zenith[-1]], aspect='auto', cmap='rainbow')
    axs[0, 1].set_title("Zhaires protons")
    axs[0, 1].set_xlabel("Xmax [g/cm²]")
    axs[0, 1].set_ylabel("Zenith [°]")
    fig.colorbar(im2, ax=axs[0, 1])

    im3 = axs[1, 0].imshow(hist_Xmax_zenith_Coreas_iron.T, origin='lower', extent=[bins_Xmax[0], bins_Xmax[-1], bins_zenith[0], bins_zenith[-1]], aspect='auto', cmap='rainbow')
    axs[1, 0].set_title("Coreas iron")
    axs[1, 0].set_xlabel("Xmax [g/cm²]")
    axs[1, 0].set_ylabel("Zenith [°]")
    fig.colorbar(im3, ax=axs[1, 0])

    im4 = axs[1, 1].imshow(hist_Xmax_zenith_Zhaires_iron.T, origin='lower', extent=[bins_Xmax[0], bins_Xmax[-1], bins_zenith[0], bins_zenith[-1]], aspect='auto', cmap='rainbow')
    axs[1, 1].set_title("Zhaires iron")
    axs[1, 1].set_xlabel("Xmax [g/cm²]")
    axs[1, 1].set_ylabel("Zenith [°]")
    fig.colorbar(im4, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(mid_bins_zenith, np.mean(hist_Xmax_zenith_Coreas_protons, axis = 0), label = "Coreas protons", color = "red", ls = "--")
    plt.plot(mid_bins_zenith, np.mean(hist_Xmax_zenith_Zhaires_protons, axis = 0), label = "Zhaires protons", color = "red")
    plt.plot(mid_bins_zenith, np.mean(hist_Xmax_zenith_Coreas_iron, axis = 0), label = "Coreas iron", color = "blue", ls = "--")
    plt.plot(mid_bins_zenith, np.mean(hist_Xmax_zenith_Zhaires_iron, axis = 0), label = "Zhaires iron", color = "blue")
    plt.xlabel("Zenith [°]")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(mid_bins_zenith, np.std(hist_Xmax_zenith_Coreas_protons, axis = 0), label = "Coreas protons", color = "red", ls = "--")
    plt.plot(mid_bins_zenith, np.std(hist_Xmax_zenith_Zhaires_protons, axis = 0), label = "Zhaires protons", color = "red")
    plt.plot(mid_bins_zenith, np.std(hist_Xmax_zenith_Coreas_iron, axis = 0), label = "Coreas iron", color = "blue", ls = "--")
    plt.plot(mid_bins_zenith, np.std(hist_Xmax_zenith_Zhaires_iron, axis = 0), label = "Zhaires iron", color = "blue")
    plt.xlabel("Zenith [°]")
    plt.ylabel("Standard Deviation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    hist_Xmax_Coreas_protons = np.histogram(X_max_gram_Coreas_protons, bins = bins_Xmax, density = True, weights = E_primary_Coreas_protons / (np.log10(np.cos(np.radians(zenith_Coreas_protons)))))[0]
    hist_Xmax_Zhaires_protons = np.histogram(X_max_gram_Zhaires_protons, bins = bins_Xmax, density = True, weights = E_primary_Zhaires_protons / (np.log10(np.cos(np.radians(zenith_Zhaires_protons)))))[0]
    hist_Xmax_Coreas_iron = np.histogram(X_max_gram_Coreas_iron, bins = bins_Xmax, density = True, weights = E_primary_Coreas_iron / (np.log10(np.cos(np.radians(zenith_Coreas_iron)))))[0]
    hist_Xmax_Zhaires_iron = np.histogram(X_max_gram_Zhaires_iron, bins = bins_Xmax, density = True, weights = E_primary_Zhaires_iron / (np.log10(np.cos(np.radians(zenith_Zhaires_iron)))))[0]

    plt.figure()
    plt.plot(mid_bins_Xmax, hist_Xmax_Coreas_protons, label = "Coreas protons", color = "red", ls = "--")
    plt.plot(mid_bins_Xmax, hist_Xmax_Zhaires_protons, label = "Zhaires protons", color = "red")
    plt.plot(mid_bins_Xmax, hist_Xmax_Coreas_iron, label = "Coreas iron", color = "blue", ls = "--")
    plt.plot(mid_bins_Xmax, hist_Xmax_Zhaires_iron, label = "Zhaires iron", color = "blue")
    plt.xlabel("Xmax [g/cm²]")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    hist_Xmax_Eprimary_Coreas_protons = np.histogram2d(X_max_gram_Coreas_protons, E_primary_Coreas_protons, bins = [bins_Xmax, E_bins], density = True, weights = E_primary_Coreas_protons)[0]
    hist_Xmax_Eprimary_Zhaires_protons = np.histogram2d(X_max_gram_Zhaires_protons, E_primary_Zhaires_protons, bins = [bins_Xmax, E_bins], density = True, weights = E_primary_Zhaires_protons)[0]
    hist_Xmax_Eprimary_Coreas_iron = np.histogram2d(X_max_gram_Coreas_iron, E_primary_Coreas_iron, bins = [bins_Xmax, E_bins], density = True, weights = E_primary_Coreas_iron)[0]
    hist_Xmax_Eprimary_Zhaires_iron = np.histogram2d(X_max_gram_Zhaires_iron, E_primary_Zhaires_iron, bins = [bins_Xmax, E_bins], density = True, weights = E_primary_Zhaires_iron)[0]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    im1 = axs[0, 0].imshow(hist_Xmax_Eprimary_Coreas_protons.T, origin='lower', extent=[bins_Xmax[0], bins_Xmax[-1], E_bins[0], E_bins[-1]], aspect='auto', cmap='rainbow')
    axs[0, 0].set_title("Coreas protons")
    axs[0, 0].set_xlabel("Xmax [g/cm²]")
    axs[0, 0].set_ylabel("E primary [GeV]")
    fig.colorbar(im1, ax=axs[0, 0])

    im2 = axs[0, 1].imshow(hist_Xmax_Eprimary_Zhaires_protons.T, origin='lower', extent=[bins_Xmax[0], bins_Xmax[-1], E_bins[0], E_bins[-1]], aspect='auto', cmap='rainbow')
    axs[0, 1].set_title("Zhaires protons")
    axs[0, 1].set_xlabel("Xmax [g/cm²]")
    axs[0, 1].set_ylabel("E primary [GeV]")
    fig.colorbar(im2, ax=axs[0, 1])

    im3 = axs[1, 0].imshow(hist_Xmax_Eprimary_Coreas_iron.T, origin='lower', extent=[bins_Xmax[0], bins_Xmax[-1], E_bins[0], E_bins[-1]], aspect='auto', cmap='rainbow')
    axs[1, 0].set_title("Coreas iron")
    axs[1, 0].set_xlabel("Xmax [g/cm²]")
    axs[1, 0].set_ylabel("E primary [GeV]")
    fig.colorbar(im3, ax=axs[1, 0])

    im4 = axs[1, 1].imshow(hist_Xmax_Eprimary_Zhaires_iron.T, origin='lower', extent=[bins_Xmax[0], bins_Xmax[-1], E_bins[0], E_bins[-1]], aspect='auto', cmap='rainbow')
    axs[1, 1].set_title("Zhaires iron")
    axs[1, 1].set_xlabel("Xmax [g/cm²]")
    axs[1, 1].set_ylabel("E primary [GeV]")
    fig.colorbar(im4, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(mid_E_bins, np.mean(hist_Xmax_Eprimary_Coreas_protons, axis = 0), label = "Coreas protons", color = "red", ls = "--")
    plt.plot(mid_E_bins, np.mean(hist_Xmax_Eprimary_Zhaires_protons, axis = 0), label = "Zhaires protons", color = "red")
    plt.plot(mid_E_bins, np.mean(hist_Xmax_Eprimary_Coreas_iron, axis = 0), label = "Coreas iron", color = "blue", ls = "--")
    plt.plot(mid_E_bins, np.mean(hist_Xmax_Eprimary_Zhaires_iron, axis = 0), label = "Zhaires iron", color = "blue")
    plt.xlabel("E primary [GeV]")
    plt.ylabel("Xmax [g/cm²]")
    plt.legend()
    plt.tight_layout()
    plt.show()  

    plt.figure()
    plt.plot(mid_E_bins, np.std(hist_Xmax_Eprimary_Coreas_protons, axis = 0), label = "Coreas protons", color = "red", ls = "--")
    plt.plot(mid_E_bins, np.std(hist_Xmax_Eprimary_Zhaires_protons, axis = 0), label = "Zhaires protons", color = "red")
    plt.plot(mid_E_bins, np.std(hist_Xmax_Eprimary_Coreas_iron, axis = 0), label = "Coreas iron", color = "blue", ls = "--")
    plt.plot(mid_E_bins, np.std(hist_Xmax_Eprimary_Zhaires_iron, axis = 0), label = "Zhaires iron", color = "blue")
    plt.xlabel("E primary [GeV]")
    plt.ylabel("Xmax [g/cm²]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """

def compare_Coreas_Zhaires_ADF():
    sim_numbers_Coreas = [i for i in range(0, 90)] #150
    sim_numbers_Zhaires = [i for i in range(12, 14)] #14

    list_Coreas = []
    list_Zhaires = []

    for sim_number in sim_numbers_Coreas:
        list_Coreas.append(np.genfromtxt(f"/sps/grand/cprevotat/grand/efficiency/out_extract_infos/out_sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}.txt", dtype = str).T)

        # add the sim number at the end of each list
        list_Coreas[-1] = np.vstack((list_Coreas[-1], np.full(list_Coreas[-1].shape[1], sim_number)))

        if sim_number % 10 == 0:
            print("sim_number : ", sim_number)

    for sim_number in sim_numbers_Zhaires:
        list_Zhaires.append(np.genfromtxt(f"/sps/grand/cprevotat/grand/efficiency/out_extract_infos/out_sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-NJ_{sim_number:04d}.txt", dtype = str).T)
        list_Zhaires[-1] = np.vstack((list_Zhaires[-1], np.full(list_Zhaires[-1].shape[1], sim_number)))

    data_Coreas = np.concatenate(list_Coreas, axis = 1)
    data_Zhaires = np.concatenate(list_Zhaires, axis = 1)
    data_Zhaires[1, :] = np.where(data_Zhaires[1, :] == "Fe^56", "5626.0", data_Zhaires[1, :])

    del list_Coreas, list_Zhaires
    gc.collect()

    data_Coreas = np.concatenate((data_Coreas[0:2], data_Coreas[3:]), axis = 0) # remove the DU_ID column, which is not a float
    data_Zhaires = np.concatenate((data_Zhaires[0:2], data_Zhaires[3:]), axis = 0) # remove the DU_ID column, which is not a float
    data_Coreas = data_Coreas.astype(float)
    data_Zhaires = data_Zhaires.astype(float)

    data_Coreas = data_Coreas[:, data_Coreas[2, :] <= np.max(data_Zhaires[2, :])] # remove events with energy bigger than Zhaires
    data_Zhaires = data_Zhaires[:, data_Zhaires[2, :] >= np.min(data_Coreas[2, :])] # remove events with energy < 10^7 GeV
    data_Zhaires = data_Zhaires[:, data_Zhaires[8, :] >= np.min(data_Coreas[8, :])] 

    mask_protons_Coreas = data_Coreas[1, :] == 14
    mask_protons_Zhaires = data_Zhaires[1, :] == 2212
    data_Coreas_protons = data_Coreas[:, mask_protons_Coreas]
    data_Zhaires_protons = data_Zhaires[:, mask_protons_Zhaires]

    print("Shape data Coreas, shape data Coreas protons : ", np.shape(data_Coreas), np.shape(data_Coreas_protons))
    print("Shape data Zhaires, shape data Zhaires protons : ", np.shape(data_Zhaires), np.shape(data_Zhaires_protons))


    X_max_gram_Coreas_protons = data_Coreas_protons[7, :]
    X_max_gram_Zhaires_protons = data_Zhaires_protons[7, :]


    E_primary_Coreas_protons = data_Coreas_protons[2]
    E_primary_Zhaires_protons = data_Zhaires_protons[2]


    E_em_Coreas_protons = data_Coreas_protons[3]
    E_em_Zhaires_protons = data_Zhaires_protons[3]


    zenith_Coreas_protons = data_Coreas_protons[8]
    zenith_Zhaires_protons = data_Zhaires_protons[8]


    azimuth_Coreas_protons = data_Coreas_protons[9]
    azimuth_Zhaires_protons = data_Zhaires_protons[9]

    event_number_Coreas_protons = data_Coreas_protons[0]
    event_number_Zhaires_protons = data_Zhaires_protons[0]
    Xmax_x_Coreas_protons = data_Coreas_protons[4]
    Xmax_y_Coreas_protons = data_Coreas_protons[5]
    Xmax_z_Coreas_protons = data_Coreas_protons[6]
    Xmax_x_Zhaires_protons = data_Zhaires_protons[4]
    Xmax_y_Zhaires_protons = data_Zhaires_protons[5]
    Xmax_z_Zhaires_protons = data_Zhaires_protons[6]


    # in this part we look for two events in Coreas and Zhaires that would be similar in energy, Xmax, Eem, zenith and core position, maybe it doesn’t exist
    ratio = E_em_Coreas_protons[:, None] / E_em_Zhaires_protons[None, :]

    candidate_i, candidate_j = np.where((ratio > 0.98) & (ratio < 1.02))
    print(f"Found {len(candidate_i)} candidate pairs with similar energy")
    mask = (
    (np.abs(X_max_gram_Coreas_protons[candidate_i] -
        X_max_gram_Zhaires_protons[candidate_j]) < 20)
    &
    (np.abs(zenith_Coreas_protons[candidate_i] -
        zenith_Zhaires_protons[candidate_j]) < 2)
    &
    (np.abs(azimuth_Coreas_protons[candidate_i] -
        azimuth_Zhaires_protons[candidate_j]) < 2)
    )

    candidate_i = candidate_i[mask]
    candidate_j = candidate_j[mask]

    print(f"Found {len(candidate_i)} matching pairs")
    print(f"Found {len(candidate_j)} matching pairs")

    print("Candidate pairs (Coreas index, Zhaires index):")
    """
    for i, j in zip(candidate_i, candidate_j):
        print(f"  ({i}, {j})")
        print(f"    Coreas: E={E_primary_Coreas_protons[i]}, Xmax={X_max_gram_Coreas_protons[i]}, E_em={E_em_Coreas_protons[i]}, zenith={zenith_Coreas_protons[i]}, azimuth={azimuth_Coreas_protons[i]}, core=({x_Coreas_protons[i]}, {y_Coreas_protons[i]}, {z_Coreas_protons[i]})")
        print(f"    Zhaires: E={E_primary_Zhaires_protons[j]}, Xmax={X_max_gram_Zhaires_protons[j]}, E_em={E_em_Zhaires_protons[j]}, zenith={zenith_Zhaires_protons[j]}, azimuth={azimuth_Zhaires_protons[j]}, core=({x_Zhaires_protons[j]}, {y_Zhaires_protons[j]}, {z_Zhaires_protons[j]})")
    """

    candidate_i = np.sort(candidate_i, axis = -1)
    candidate_j = np.sort(candidate_j, axis = -1)
    old_sim_number_Coreas = -1
    old_sim_number_Zhaires = -1
    path_Coreas = f"/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_"
    path_Zhaires = f"/sps/grand/DC2.1rc4/GP300ZHAireS-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-NJ_"

    plt.figure()
    plt.scatter([], [], label = "Coreas", c = "red")
    plt.scatter([], [], label = "Zhaires", c = "blue")

    count = 0
    for i, j in zip(candidate_i, candidate_j):
        Xmax_grams_Z, E_em_Z, zenith_Z, azimuth_Z, XSource_Z = X_max_gram_Zhaires_protons[j], E_em_Zhaires_protons[j], zenith_Zhaires_protons[j], azimuth_Zhaires_protons[j], np.array([Xmax_x_Zhaires_protons[j], Xmax_y_Zhaires_protons[j], Xmax_z_Zhaires_protons[j]])
        Xmax_grams_C, E_em_C, zenith_C, azimuth_C, XSource_C = X_max_gram_Coreas_protons[i], E_em_Coreas_protons[i], zenith_Coreas_protons[i], azimuth_Coreas_protons[i], np.array([Xmax_x_Coreas_protons[i], Xmax_y_Coreas_protons[i], Xmax_z_Coreas_protons[i]])
        sim_number_Coreas = data_Coreas_protons[-1, i]
        sim_number_Zhaires = data_Zhaires_protons[-1, j]

        if sim_number_Coreas != old_sim_number_Coreas:
            old_sim_number_Coreas = sim_number_Coreas
            path_C = path_Coreas + f"{int(sim_number_Coreas):04d}"

        if sim_number_Zhaires != old_sim_number_Zhaires:
            old_sim_number_Zhaires = sim_number_Zhaires
            path_Z = path_Zhaires + f"{int(sim_number_Zhaires):04d}"

        paths = [path_C, path_Z]
        for path in paths:
            sim = "Coreas" if path == path_C else "Zhaires"
            event_number = event_number_Coreas_protons[i] if sim == "Coreas" else event_number_Zhaires_protons[j]

            Xmax_grams, E_em, zenith, azimuth, XSource = (Xmax_grams_C, E_em_C, zenith_C, azimuth_C, XSource_C) if sim == "Coreas" else (Xmax_grams_Z, E_em_Z, zenith_Z, azimuth_Z, XSource_Z)

            zenith = np.radians(zenith)
            azimuth = np.radians(azimuth)

            d_input = dh.DataDirectory(path)
            tEfield = d_input.tefield_l1
            tshower = d_input.tshower_l0
            trun = d_input.trun_l1

            events_list = tshower.get_list_of_events()
            print(np.shape(events_list))

            idx = [i for i, x in enumerate(events_list) if x[0] == event_number][0]
            print("idx : ", idx)
            tshower.get_event(events_list[idx][0], events_list[idx][1])
            tEfield.get_event(events_list[idx][0], events_list[idx][1])
            trun.get_run(events_list[idx][1])

            trace = tEfield.trace


            du_id = np.array(tEfield.du_id)
            du_id_trun = np.array(trun.du_id)

            antenna_positions = np.array([trun.du_xyz[i] for i in range(len(du_id_trun)) if du_id_trun[i] in du_id])
            peak_amplitudes = np.array([ext.get_peak_amplitude(trace[i], channels=[0, 1, 2]) for i in range(len(du_id))])
            #xmax_pos = np.array([tshower.xmax_pos_shc for i in range(len(du_id))])
            xmax_pos = np.array(tshower.xmax_pos_shc)

            #ct = np.cos(zenith); st = np.sin(zenith); cp = np.cos(azimuth); sp = np.sin(azimuth)
            #K = np.array([-st*cp,-st*sp,-ct])
            r_xmax = np.linalg.norm(xmax_pos)
            print("r, xmax : ", r_xmax, xmax_pos)
            Xmax = swf.compute_Xsource_cartesian_coords(zenith, azimuth, r_xmax, groundAltitude = cons.groundAltitude)[0]
            print("Xmax : ", Xmax)
            l_ant = an.distance_source_antenna(antenna_positions, Xmax)
            omega = an.omega(zenith, azimuth, antenna_positions, Xmax)

            #t_seconds = np.array(tEfield.du_seconds)
            #t_du_nanoseconds = np.array(tEfield.du_nanoseconds)
            #tants = t_seconds + t_du_nanoseconds * 1e-9 - np.min(t_seconds + t_du_nanoseconds * 1e-9)
            omega_cr = np.radians(1.5)
            delta_omega = np.radians(1.5)
            adf_values = peak_amplitudes / l_ant /  (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2. - 1. )/delta_omega)**2.)
            plt.scatter(np.degrees(omega), adf_values, c = "r" if sim == "Coreas" else "b")

            """

            #print(zenith, azimuth, tants, antenna_positions)
            _, _, r_xmax_swf, _ = swf.recons_swf(zenith, azimuth, tants, antenna_positions)
            print("Xmax_pos : ", XSource)
            XSource = swf.compute_Xsource_cartesian_coords(zenith, azimuth, r_xmax_swf, groundAltitude = cons.groundAltitude)[0]
            # beware I very probably have different coordinate systems
            #print("antenna_positions : ", antenna_positions)
            print("Xmax pos : ", XSource)
            theta_adf, phi_adf, delta_omega, amplitude = adf.recons_ADF(zenith, azimuth, peak_amplitudes, antenna_positions, XSource)
            print("reconstructed")
            eta, omega, omega_cr, l_ant, adf_value = adf.ADF_parameters(theta_adf, phi_adf, delta_omega, amplitude, antenna_positions, XSource, groundAltitude=cons.groundAltitude, Bvec = cons.Bvec)
            print("got parameters")
            print(omega_cr, omega)
            print(np.shape(l_ant), np.shape(amplitude), np.shape(omega_cr), np.shape(delta_omega))
            omega, f_adf = adf.ADF_fun(np.mean(l_ant), amplitude, np.mean(omega_cr), delta_omega)
            print("got ADF")


            color = "r" if sim == "Coreas" else "b"
            plt.scatter(omega, f_adf, c = color)
            """
            count += 1
        if count ==10:
            break

        

    plt.xlim((0, 3))
    plt.xlabel("Omega")
    plt.ylabel("f_ADF")
    plt.title("Comparison of ADF between Coreas and Zhaires")
    plt.legend()
    plt.show()

        
def Xmax_Functionof_E_zenith(primary_type):
    #primary type should be either "proton" or "iron"
    sim_numbers = [i for i in range(1, 14)]
    primary_to_keep = [14, 2212] if primary_type == "proton" else [5626.0]

    list_data = []
    for sim_number in sim_numbers:
        list_data.append(np.genfromtxt(f"/sps/grand/cprevotat/grand/efficiency/out_extract_infos/out_sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-NJ_{sim_number:04d}.txt", dtype = str).T)

    data = np.concatenate(list_data, axis = 1)
    data[1, :] = np.where(data[1, :] == "Fe^56", "5626.0", data[1, :]) # only if we study Zhaires

    del list_data
    gc.collect()

    data = np.concatenate((data[0:2], data[3:]), axis = 0) # remove the DU_ID column, which is not a float
    data = data.astype(float)
    # please correct the line above : 
    mask = np.isin(data[1], primary_to_keep)
    data = data[:, mask]

    X_max_grams = data[7, :]
    E_prim = data[2]
    zenith = data[8]
    E_em = data[3]

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 12))

    ax0.scatter(E_prim, X_max_grams, s = 0.5)
    ax0.set_xscale("log")
    ax0.set_xlabel("Primary Energy [GeV]")
    ax0.set_ylabel("Xmax [g/cm²]")
    ax0.set_title(f"Xmax vs Primary Energy for {primary_type.capitalize()}")

    ax1.scatter(zenith, X_max_grams, s = 0.5)
    ax1.set_xlabel("Zenith Angle [deg]")
    ax1.set_ylabel("Xmax [g/cm²]")
    ax1.set_title(f"Xmax vs Zenith Angle for {primary_type.capitalize()}")

    ax2.scatter(E_em, E_em / E_prim, s = 0.5)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Electromagnetic Energy [GeV]")
    ax2.set_ylabel(r"$\frac{E_{\mathrm{em}}}{E_{\mathrm{prim}}}$")
    ax2.set_title(f"Xmax vs Electromagnetic Energy for {primary_type.capitalize()}")

    plt.show()








if __name__ == "__main__":

    f_sample = 500e6 # Hz, ADC sampling rate
    FLT0_trig_params_file = "/sps/grand/cprevotat/grand/grand/grand/exposure/dict_trig_params_fir.csv"

    #for sim_number in range(11, 14):
        #print("sim_number : ", sim_number)
        #extract_infos(f"/sps/grand/DC2.1rc4/GP300ZHAireS-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-NJ_{sim_number:04d}") # commands to extract the infos we need from the simulations
        #extract_infos(f"/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}") # commands to extract the infos we need from the simulations
        #data = np.genfromtxt(f"/sps/grand/cprevotat/grand/efficiency/out_extract_infos/out_sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}.txt", dtype = str).T
        #print(data[-1])
    #core_position_triggered_events()
    #investigate_du12()
    #compare_antenna_positions()

    #compare_Coreas_Zhaires()

    #path_to_file = "/sps/grand/DC2.1rc4/GP300ZHAireS-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-NJ_0001" #     # may find the COREAS simulations in this directory : /sps/grand/DC2_Coreas/old_sims/COREAS, but should ask someone for that (to know what they are etc) ; we don’t care for now as the structure should be similar to the Zhaires one
    #path_to_file = "/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_0000"
    #extract_infos(path_to_file) # commands to extract the infos we need from the simulations
    #high_E_T1(42)
    #plot_killer_high_E()
    #compute_Nevents()
    #print("NCRs : ")
    #compute_NCRs() # less up to date
    #plot_antennas_event()
    #plot_trace_event(sim_number = 26, event_number = 3295) # event_number = 93
    #plot_2D_efficiency()
    #look_at_results(Oma = True)
    #plot_Tquiet_cut()
    #plot_MD_trace()
    #compare_Coreas_Zhaires_ADF()
    Xmax_Functionof_E_zenith("proton")


    #plot_trace_sims()
    #plot_allowed_range_t0()
    """

    #plot_times_for_file("/sps/grand/data/gp80/GrandRoot/2025/09/GP80_20250922_153758_RUN10158_CD_20dB-GP65-Y2float-26DUs-filter-fixed-noise-source-CD-100000-1.root")

    plot_allowed_range_t0()
    #define_allowed_range_t0()
    data = np.loadtxt("./efficiency/out_extract_infos/allowed_times_t0_2025_07.txt", dtype = str).T # load the output file of the previous function, to check the results
    min_times = data[1].astype(float)
    max_times = data[2].astype(float)

    mask = (max_times > 1.8e9) | (min_times > 1.8e9) | (max_times < 1e9) | (min_times < 1e9) 
    print("number of files in which I have weird values : ", np.shape(mask[mask == True]), "out of ", np.shape(min_times))
    min_times = min_times[~mask]
    max_times = max_times[~mask]

    merged_min, merged_max = merge_intervals(min_times, max_times)
    print(np.shape(merged_min))

    plt.figure()
    for i in range(0, len(merged_min)):
        plt.plot([merged_min[i], merged_max[i]], [1 / (i+1), 1 / (i+1)], color = "blue")
    plt.xlabel("Min time (s)")
    plt.ylabel("Max time (s)")
    plt.title("Merged allowed range for t0")
    plt.yscale("log")
    plt.show()
    """

    """
    data = np.loadtxt("./efficiency/out_extract_infos/allowed_times_t0_2026_02.txt", dtype = str).T # load the output file of the previous function, to check the results
    min_times = data[1].astype(float)
    max_times = data[2].astype(float)

    mask = (max_times > 1.8e9) | (min_times > 1.8e9)
    print("number of files in which I have weird values : ", np.shape(mask[mask == True]), "out of ", np.shape(min_times))
    min_times = min_times[~mask]
    max_times = max_times[~mask]
    plt.figure()
    liste_index = [i for i in range(0, len(min_times))]
    plt.plot(liste_index, min_times, "x", label = "min times")
    plt.plot(liste_index, max_times, "x", label = "max times")
    plt.xlabel("File index")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.title("Min and max times for each file")
    plt.legend()
    plt.show()
    """
    #define_allowed_range_t0()
    #plot_allowed_range_t0()
    #plot_trace_sims()
    #plot_times_for_file("/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260414_143613_RUN343_MD_20dB-GP65-58DUs-10s-512trace-FY2Float-newdataformat-noeventbuilder-0009.root")
    #plot_times_for_file('/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260430_231212_RUN474_MD_20dB-50DUs-512trace-FY2Float-Normal-WOonlinefilter-new-cs-daq-chengwei-0001.root')
    #plot_times_for_file("/sps/grand/data/gp80/GrandRoot/2026/05/GP80_20260525_201608_RUN10369_CD_20dB-GP65-58DUs-512trace-FY2Float-Normal-Wonlinefilter-new-cs-daq-v1p0-cw-CT-60-40ADC-CD-100000-25.root")
    #plot_times_for_file('/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260430_114730_RUN10347_CD_20dB-GP65-50DUs-512trace-FY2Float-Normal-WOonlinefilter-new-cs-daq-chengwei-CD-100000-2095.root')

    #compare_antenna_positions()
    #extract_infos("/sps/grand/data/gp80/GrandRoot/2025/10/GP80_20251029_125535_RUN10179_CD_20dB-GP65-57dus-CD-100000-30.root")
    #compare_antenna_numbers()
    #plot_trace_sims()
    #plot_Auger_spectrum()
    #get_trigger_position("2025/11/GP80_20251125_072256_RUN10188_CD_20dB-GP65-44DUs-512trace-FY2Float-CD-100000-25.root")
    #sim_distribution_E_angles()
    #get_idx_1st_T1crossing("2025/10/GP80_20251031_154319_RUN10180_CD_20dB-GP65-57dus-CD-100000-2.root", data_dict)
    #get_idx_1st_T1crossing("2026/04/GP80_20260420_064958_RUN10303_CD_20dB-GP65-58DUs-512trace-FY2Float-Normal-Wonlinefilter-new-cs-daq-chengwei-CD-100000-185.root", data_dict)
    #get_idx_1st_T1crossing("2026/04/GP80_20260408_013034_RUN10230_CD_20dB-GP65-58DUs-512trace-FY2Float-CD-100000-450.root", data_dict)
    #get_idx_1st_T1crossing("2026/01/GP80_20260131_234617_RUN10203_CD_20dB-GP65-51DUs-512trace-FY2Float-dunhuangsiteTestGPU-CD-100000-42.root", data_dict)
    #get_idx_1st_T1crossing("2026/05/GP80_20260531_235958_RUN12_CD_20dB-GP65-8DUs-testNutrigCorreXY-CD-100000-139.root", data_dict)
    #get_idx_1st_T1crossing("2026/05/GP80_20260525_201608_RUN10369_CD_20dB-GP65-58DUs-512trace-FY2Float-Normal-Wonlinefilter-new-cs-daq-v1p0-cw-CT-60-40ADC-CD-100000-25.root", data_dict)
    #get_idx_1st_T1crossing("2026/06/GP80_20260610_195508_RUN10380_CD_20dB-GP65-58DUs-512trace-FY2Float-Normal-Wonlinefilter-new-cs-daq-v1p0-cw-HCT-90-70ADC-CD-100000-592.root", data_dict)

    #judge_trigger_true_events("2026/04/GP80_20260407_194133_RUN340_MD_20dB-GP65-58DUs-10s-512trace-FY2Float-0006.root", event_number_md = 15127, run_md = 340, sim_number = 84) # the second argument is the simulation number, I should ensure that it corresponds to the data file, but for now I just take one of the simulations that I have already analyzed for the distribution of E and angles
    #plot_MD_trace()
    #look_at_results(Oma = True)
    #plot_t_Coreas_simulations()
    #extract_noise_from_t0(["/sps/grand/data/gp80/raw/2026/04/GP80_20260430_231212_RUN474_MD_20dB-50DUs-512trace-FY2Float-Normal-WOonlinefilter-new-cs-daq-chengwei-0001.bin"], target_timestamp = 1.8e9, target_duid = 1071) 
    #plot_times_for_file("/sps/grand/data/gp80/GrandRoot/2026/03/GP80_20260301_004053_RUN324_MD_20dB-50DUs-10s-512trace-FY2Float-dunhuangsiteTestGPU-0081.root")
    #plot_times_for_file('/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260430_231212_RUN474_MD_20dB-50DUs-512trace-FY2Float-Normal-WOonlinefilter-new-cs-daq-chengwei-0001.root')
    #plot_times_for_file("/sps/grand/data/gp80/GrandRoot/2026/03/GP80_20260331_234503_RUN337_MD_20dB-GP65-42DUs-10s-512trace-FY2Float-dunhuangsiteTestGPU-0002.root")
    #plot_Nevents_per_DU_in_file('/sps/grand/data/gp80/raw/2026/05/GP80_20260503_183453_RUN474_MD_20dB-50DUs-512trace-FY2Float-Normal-WOonlinefilter-new-cs-daq-chengwei-0037.bin')
    #plot_Nevents_per_DU_in_file('/sps/grand/data/gp80/GrandRoot/2026/05/GP80_20260531_235958_RUN12_CD_20dB-GP65-8DUs-testNutrigCorreXY-CD-100000-139.root')
    
    """
     # the real code here : 
    ordered_files, t_min_ordered, t_max_ordered, liste_idx_in_intervals = prepare_for_drawing_t0(filename = "/sps/grand/cprevotat/grand/efficiency/out_extract_infos/allowed_times_t0gps_2026_05.txt")
    with open("/sps/grand/cprevotat/grand/efficiency/out_extract_infos/time_period_2026_05.txt", "w") as f:
        # The total observation time (considering only times when the detector was running) is (in seconds) :
        total_time = np.sum(t_max_ordered - t_min_ordered)
        f.write(str(total_time) )
    # with t_min_ordered and t_max_ordered I do get the total time period we are looking at, I can use it to get the time period

    sim_numbers = [i for i in range(0, 1)]
    for sim_number in sim_numbers:
        print("dealing with simulation number ", sim_number, "out of ", len(sim_numbers))
        t0, idx = draw_t0(100, t_min_ordered, t_max_ordered)

        liste_files = []
        for i in range(0, len(t0)):
            sublist = []
            for j in range(0, len(liste_idx_in_intervals[idx[i]])):
                sublist.append(str(ordered_files[liste_idx_in_intervals[idx[i]][j]]))
            liste_files.append(sublist)


        #print("t0 : ", t0)
        #print("files corresponding to t0 : ", liste_files)
        #print("dealing with simulation number ", sim_number, "out of ", len(sim_numbers))
        print("number of files corresponding to t0 : ", [len(liste_files[i]) for i in range(0, len(t0))])

        new_list = []


        for i in range(0, len(t0)):
            #if i % 20 == 0:
                #print("Looking for the time number ", i, "out of ", len(t0))
            new_list.append([liste_files[i][j].replace("GrandRoot", "raw").replace(".root", ".bin") for j in range(0, len(liste_files[i]))]) # could be enough for now : we only look at the files corresponding to where I extracted the times from ; might be some border effects when we then consider that we have to extract the 2 closest events and not only the closest
            #extract_noise_from_t0(new_list, target_timestamp = t0[i], target_duid = 1071) 
        print("start judge_trigger_true_events")
        #judge_trigger_true_events(new_list, sim_number = sim_number, t0 = t0) 
        cProfile.run("judge_trigger_true_events(new_list, sim_number = sim_number, t0 = t0)", "/sps/grand/cprevotat/grand/efficiency/profile_results_listdu2.prof")
        """
    


    """
    path = "/sps/grand/data/gp80/GrandRoot/2026/05/"
    list_files = [str(f) for f in Path(path).rglob("*.root") if f.is_file()]
    list_files = [f for f in list_files if "CD" in f] # select all the CD files

    df = pd.DataFrame({"filename": list_files})
    dt_str = df["filename"].str.extract(r"(\d{8}_\d{6})")[0] # extract datetime string

    dt = pd.to_datetime(dt_str, format="%Y%m%d_%H%M%S")     # convert to datetime

    # vectorized extraction
    df["year"]   = dt.dt.year
    df["month"]  = dt.dt.month
    df["day"]    = dt.dt.day
    df["hour"]   = dt.dt.hour
    df["minute"] = dt.dt.min
    df["second"] = dt.dt.second
    # convert this in unix time
    df['datetime'] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": df["day"], "hour": df["hour"], "minute": df["minute"], "second": df["second"]}).astype(np.int64)//10**6

    # now order the dataframe based on the datetime column
    df = df.sort_values(by="datetime").reset_index(drop=True)
    print(df["datetime"])

    #t0 = np.array([1777821978, 1777825578, 1777820978]) # example timestamps
    sim_numbers = [i for i in range(80, 85)]
    # collect the t0s and the dus involved in the trigger
    for n_sim in sim_numbers: # loop of 150
        print("dealing with simulation number ", n_sim, "out of ", len(sim_numbers))
        t0 = []
        event_ids = []
        du_ids_triggering_2 = []
        with open(f"/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{n_sim:04d}.json", "r") as f:
            data = json.load(f)
        for count, row in enumerate(data): # loop of 100
            N_triggering_dus = 0
            du_ids = [] 
            for du_level in row["triggering_events_2"]:
                if np.abs(du_level[2]) < 10 and np.abs(du_level[3]) < 10:
                    du_ids.append(du_level[0])

            du_ids = np.unique(du_ids)
            #print(du_ids)
            if len(du_ids) >= 5:
                t0.append(row["fixed"][-1])
                event_ids.append(row["fixed"][1])
                du_ids_triggering_2.append(du_ids)
        
        if len(t0) == 0:
            print("No event with 5 or more triggering DUs for simulation number ", n_sim)
            continue

        t0_array  = np.array(t0)

        # so now I can load my nanoseconds from the simulations

        d_input_sim = dh.DataDirectory(f"/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{n_sim:04d}")
        tadc_l1_sim = d_input_sim.tadc_l1
        
        #print("t0, event_ids : ", t0, event_ids)



        for idx_loop, timestamp in enumerate(t0_array): # again loop around 20

            tadc_l1_sim.get_event(event_ids[idx_loop], 1) # event_number, run_number
            du_ids = tadc_l1_sim.du_id
            t0_ns = np.array(tadc_l1_sim.du_nanoseconds)*1e-9 # seems to stop at 0.02 s, so I can add 0.98s
            t0_ns = t0_ns[np.isin(du_ids, du_ids_triggering_2[idx_loop])] # keep only the time of the dus that passed all the trigger conditions so far
            CR_id_tns = [[(int(du_ids_triggering_2[idx_loop][i]), round(float(t0_ns[i]), 9)) for i in range(len(du_ids_triggering_2[idx_loop]))]] # list of tuples (du_id, t0_ns) for the dus that passed all the trigger conditions so far
            print("CR_id_tns : ", CR_id_tns)


            t0_ns_in_s  = t0_ns + np.random.randint(0, 98, 1)/100 # add the same high number of ns to all times related to the same event
            if len(np.where(t0_ns > 1.)[0]) > 0:
                print("We have some t0_ns that are larger than 1., we should check that")
                print(t0_ns)


            t0 = timestamp + t0_ns_in_s# then indeed the ns should be collected from the time difference of the simulated events
            min_t0 = np.min(t0) # from a given t0 in seconds (same for all dus, I build different t0, corresponding to time delays between dus)
            idx_t0 = np.searchsorted(df["datetime"], min_t0)-1 # this part works, indices is the index of the file in which I should find the t0

            #print("t0_ns_in_s : ", t0_ns_in_s)
            #print("t0 : ", t0)
            #print(t0[1:] - t0[:-1])
            #print("dealing with idx_loop", idx_loop, "shapes of t0 and trigger 2 : ", len(t0), len(du_ids_triggering_2[idx_loop]))
            filename = df["filename"][idx_t0]
            d_input_data = dh.DataFile(filename)
            #print(d_input)
            tadc_data, t_rawvoltage_data =  d_input_data.tadc_l1, d_input_data.trawvoltage
            #print(tadc, t_rawvoltage)

            N_events = tadc_data.get_entries()

            d_input_data.close()

            #filename = "/sps/grand/data/gp80/GrandRoot/2025/12/GP80_20251231_153039_RUN10194_CD_20dB-GP65-43DUs-512trace-FY2Float-CD-100000-87.root"
            filename = "/sps/grand/data/gp80/GrandRoot/2026/05/GP80_20260518_211103_RUN10367_CD_20dB-GP65-58DUs-512trace-FY2Float-Normal-Wonlinefilter-new-cs-daq-v1p0-cw-CT-60-40ADC-CD-100000-112.root"
            filename = "/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260430_121521_RUN10349_CD_20dB-GP65-50DUs-512trace-FY2Float-Normal-Wonlinefilter-new-cs-daq-chengwei-CD-100000-3.root"

            file_root = dh.DataFile(filename)
            trun = file_root.trun
            tadc = file_root.tadc
            run_number = tadc.get_list_of_events()[0][1]
            trun.get_run(run_number)
            du_id = tadc.du_id

            print(du_id)

            adc_tree = file_root.tadc
            N_events = adc_tree.get_entries()

            for i in range(0, N_events):
                adc_tree.get_entry(i)
                print(np.array(adc_tree.du_id), np.array(adc_tree.du_seconds))

            
            #for i in range(N_events):
                #tadc_data.get_entry(i)
                #du_ids = tadc_data.du_id
                #du_seconds = np.array(tadc_data.du_seconds)
                #print(du_ids, du_seconds)
            

            draw_count = tadc_data.draw("time_seconds[]:time_nanoseconds[]", "", "goff") # time_seconds : time of the first triggering DU, du_seconds is time for each du
            #draw_count = t_rawvoltage.draw("gps_time[]", "", "goff")

            ll_events_seconds = tadc_data.get_v1() # ll : low level
            times_s = np.frombuffer(ll_events_seconds, dtype=np.float64, count = draw_count).copy()

            ll_events_nanoseconds = tadc_data.get_v2()
            times_ns = np.frombuffer(ll_events_nanoseconds, dtype=np.float64, count = draw_count).copy()

            times = times_s + times_ns*1e-9

            #times = times[(times < 1.8e9) & (times > 1.7e9)]

            
            #for i in range(0, len(times)-1):
                #if times[i] > times[i+1]:
                    #print("issues with those times", times[i], times[i+1])
            
            #plt.figure()
            #plt.plot([i for i in range(0, len(times))], times-np.min(times), "x")
            #plt.show()

            #print("times : ", times)

            event_idx = np.searchsorted(times, timestamp)-1

            remaining = 300 # parameter to be tuned, the smaller the better as long as len(history) is 1000 at the end
            starting_idx = event_idx - remaining

            events_to_build_history = [] # list of list tuples, [[(id1, du_time)1, (id2, du_time), ...]] for the events that we will use to build the history
            file_idx = idx_t0

            while remaining > 0 and file_idx >= 0:

                if file_idx != idx_t0: # we already loaded the file with index idx_t0
                    d_input_data = dh.DataFile(df["filename"][file_idx])
                    tadc_data = d_input_data.tadc_l1
                    n_entries = tadc_data.get_entries()
                    end = n_entries # previous files : go until the end of the file

                else:
                    # current file: only take events BEFORE idx_time
                    end = event_idx

                take = min(remaining, end)
                start = end - take # is at least 0

                list_to_add = []
                for i in range(start, end):

                    tadc_data.get_entry(i)

                    du_id = tadc_data.du_id
                    du_ns = np.array(tadc_data.du_seconds)
                    print(du_id, du_ns)

                    event = []
                    for k in range(len(du_id)):
                        event.append((dict_febID_to_duID[du_id[k]], du_ns[k]))

                    list_to_add.append(event)
                    #print("event : ", event)

                events_to_build_history = list_to_add + events_to_build_history # add the new elements (of the last treated file) at the begining

                remaining -= take
                file_idx -= 1
                event_idx = None   # only used for first file

                d_input_data.close() # close the file we just loaded

            #print(events_to_build_history)

            #print("events_to_build_history : ", events_to_build_history)


            history, detected = process_signals(events_to_build_history) # first time to build the history
            if np.shape(history)[0] < 20: # 
                #print("filename is :", df["filename"][idx_t0], t0)
                print("Our history is too small here : ", np.shape(history))
                #print(history)
            
            history, detected = process_signals(CR_id_tns, history=history) # this time applying the T3 trigger
            #print("history : ", history)
            #print("CR_id_tns : ", CR_id_tns)

            print("detection with the T3 trigger  :", detected)
            
            #if not detected:
                #print("Event not detected by the T3 trigger, we should check that")
                #print("t0 : ", t0)
                #print("history : ", history)
                #print("CR_id_tns : ", CR_id_tns)
            
        d_input_sim.close()
        """






            



        # if it’s not too long : select the say 2000 events preceeding the t0 (which means getting the times of all the events in the file as long as 
        #they are smaller than t0, and if we lack some of them, then we should open the previous file) ; 
        # once we collected those events, we should build the history (maybe modifying a bit the code so that we don’t have to write / read data in a txt file)
        #) using the function of Pengyxiong, and then finally apply the T3 trigger cut
        # we should only do that events detected by more than 5 DUs, ie not all the events
        # 
        #then open the corresponding files, start building the histo

    

    """
    
    filename = "/sps/grand/data/gp80/GrandRoot/2026/04/GP80_20260430_121521_RUN10349_CD_20dB-GP65-50DUs-512trace-FY2Float-Normal-Wonlinefilter-new-cs-daq-chengwei-CD-100000-3.root"
    filename = "/sps/grand/data/gp80/GrandRoot/2026/02/GP80_20260228_235925_RUN10213_CD_20dB-GP65-51DUs-512trace-FY2Float-Wonlinefilter20evts-newthread2-CD-100000-1305.root"

    d_input = dh.DataFile(filename)   

    trun_l0, tadc_l1 = d_input.trun_l0, d_input.tadc

    event_list = tadc_l1.get_list_of_events()
    nb_events  = len(event_list)
    if nb_events == 0: sys.exit("No events in the file. Exiting.")

    ### Start event loop
    previous_run = None
    events = 0

    array_dus = running_DUs() # when we don’t consider the complete layout

    dict_duID_to_febID = generate_correspondance_duID_to_febID() # this is to be able to use the FLT0 parameters for each du, which are given in febID, but we want from duIDs
    dict_febID_to_duID = generate_correspondance_febID_to_duID() # this is to be able to use the FLT0 parameters for each du, which are given in febID, but we want from duIDs

    tadc_l1.get_entry(0)
    du_ids = np.array(tadc_l1.du_id)
    print(du_ids)
    print(tadc_l1.du_seconds)

    trun_l0.get_entry(0)
    print(trun_l0.du_id)

    d_input.close()


    data_path = "/sps/grand/data/gp80/GrandRoot"

    file_name = "GP80_20260430_121521_RUN10349_CD_20dB-GP65-50DUs-512trace-FY2Float-Normal-Wonlinefilter-new-cs-daq-chengwei-CD-100000-3.root"
    # file_name = "GP80_20251027_062734_RUN10174_CD_20dB-GP65-Y2float-46dus-10history-CD-100000-31.root"
    file_name = "GP80_20260512_111449_RUN10360_CD_20dB-GP65-58DUs-512trace-FY2Float-Normal-Wonlinefilter-new-cs-daq-v1p0-chengwei-CD-100000-14.root"

    year = file_name.split("_")[1][0:4]
    month = file_name.split("_")[1][4:6]

    file_path = os.path.join(data_path, year, month, file_name)
    print(file_path)

    file_root = dh.DataFile(file_path)

    adc_tree = file_root.tadc
    voltage_tree = file_root.trawvoltage

    event_list = adc_tree.get_list_of_events()
    nb_events = len(event_list)
    for event in range(nb_events):
        adc_tree.get_entry(event)
        # file_root.tadc.get_entry(event)
        voltage_tree.get_entry(event)

        DU_ids = np.array(adc_tree.du_id)
        if len(DU_ids) > 5:

            print(DU_ids)

            number_of_ant = len(DU_ids)
            list_antennas = np.arange(number_of_ant) # annoying part, here we take all antennas

            time_seconds = np.array(adc_tree.du_seconds)
            time_nanoseconds = np.array(adc_tree.du_nanoseconds)

            time_seconds_0 = time_seconds
            argmin_seconds = np.argmin(time_seconds_0)
            time_nanoseconds_0 = time_nanoseconds - time_nanoseconds[argmin_seconds]

            time = time_seconds_0 + time_nanoseconds_0  * 1e-9 # in s

            print(time)

    file_root.close()
    """