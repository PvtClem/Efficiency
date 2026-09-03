import sys
import os
import grand.dataio.data_handling as groot
import grand.manage_log as mlg
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
sys.path.append(os.path.abspath("/sps/grand/cprevotat/grand/grand/grand/exposure/")) # this is not very clean, but it was not to mess with Sei’s code, and not copy everything here
from utils import *
sys.path.append(os.path.abspath("/sps/grand/cprevotat/grand/")) # this is not very clean, but it was not to mess with Sei’s code, and not copy everything here
from functions import *

import json

#this function comes from Sei’s code, but here I change it a bit, so that it accepts different thresholds for each channel and each du


### Set input data and an output file
### data_directory: directory where you have GRAND root files
### FLT0_trig_params_file: file containing the FLT0 parameters,
### please see dict_trig_params_fir.csv for its format
# output directory : directory where you want to write your output files
output_directory, FLT0_trig_params_file = sys.argv[1], sys.argv[2]


for i in range(0, 1):

    data_directory = f"/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{i:04d}" # ensure correct formating 

    #output  = open('/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level/'+data_directory[data_directory.find('sim'):] + ".txt",'w')
    #output.write("# run number, event number, primary type, primary energy, zenith angle, azimuth angle, shower core position (x,y,z), trigger flag, list of triggered DUs\n")

    data_output = [] # list of dict, each dict correspond to an event


    print('out_judge_trigger/'+data_directory[data_directory.find('sim'):], data_directory)

    ### Read GRAND root data
    d_input = groot.DataDirectory(data_directory)
    trun_l1, tadc_l1, tshower_l0 = d_input.trun_l1, d_input.tadc_l1, d_input.tshower_l0

    event_list = tadc_l1.get_list_of_events()
    nb_events  = len(event_list)
    if nb_events == 0: sys.exit("No events in the file. Exiting.")

    ### Start event loop
    previous_run = None
    events = 0

    array_dus = running_DUs() # when we don’t consider the complete layout

    dict_duID_to_febID = generate_correspondance_duID_to_febID() # this is to be able to use the FLT0 parameters for each du, which are given in febID, but we want from duIDs
    dict_febID_to_duID = generate_correspondance_febID_to_duID() # this is to be able to use the FLT0 parameters for each du, which are given in febID, but we want from duIDs

    for event_number, run_number in event_list:

        #print("looking at event number ", event_number, " of run ", run_number)

        assert isinstance(event_number, int)
        assert isinstance(run_number, int)

        tadc_l1.get_event(event_number, run_number)
        tshower_l0.get_event(event_number, run_number)
        #tvoltage_l1.get_event(event_number, run_number)
        #print("t shower : ", tshower_l0)
        #print("N tested cores : ", tshower_l0.tested_cores) # this is to check that we have the same number of tested cores in each event, and that it is consistent with the number of cores we are going to loop over
        
        if previous_run != run_number:
            trun_l1.get_run(run_number)
            previous_run = run_number
            
        du_id = np.array(tadc_l1.du_id)
        #du_id = du_id[np.isin(du_id, array_dus)]
        
        tadc_trace = np.array(tadc_l1.trace_ch) # dimension: (N_du, XYZ, samples)

        f_sample = 500e6 # Hz, ADC sampling rate
        t_res = int(1. / f_sample * 1.e9) # ns, ADC time resolution
        rel_trace_start_time = calculate_relative_trace_start_time(tshower_l0, tadc_l1, t_res)

        trig_chnl_list = []

        data_output.append({}) # add a dict for each event

    
        # Start DU loop
        list_max_tadc_X_filt = []
        list_max_tadc_Y_filt = []
        for du_id_n in range(du_id.shape[0]):

            if du_id[du_id_n] not in array_dus: 
                continue 


            #print("looking at du_id ", du_id[du_id_n], du_id_n)


            tadc_trace_X = tadc_trace[du_id_n][0]
            tadc_trace_Y = tadc_trace[du_id_n][1]
            #print("max : ", np.max(tadc_trace_X), np.max(tadc_trace_Y))
            #tadc_trace_Z = tadc_trace[du_id_n][2]
            rel_start_time = rel_trace_start_time[du_id_n]

            ### Filtering of the traces
            ### The following lines applies
            ### 1. notch filter with a notch frequency of 39 MHz, &
            ### 2. FIR filter only passing the signals below 115 MHz
            tadc_X_filt = notch_filter(tadc_trace_X, 39e6, 0.9, f_sample)
            tadc_Y_filt = notch_filter(tadc_trace_Y, 39e6, 0.9, f_sample)
            #tadc_Z_filt = notch_filter(tadc_trace_Z, 39e6, 0.9, f_sample)

            tadc_X_filt = filter_traces_bandpass(tadc_X_filt, coeff_file='/sps/grand/cprevotat/grand/grand/grand/exposure/lowpass115MHz.txt')
            tadc_Y_filt = filter_traces_bandpass(tadc_Y_filt, coeff_file='/sps/grand/cprevotat/grand/grand/grand/exposure/lowpass115MHz.txt')
            #tadc_Z_filt = filter_traces_bandpass(tadc_Z_filt, coeff_file='/sps/grand/cprevotat/grand/grand/grand/exposure/lowpass115MHz.txt')

            ### Discuss the FLT0 trigger in the channel level
            ### The function "trigger_FLT0" is used (developed by M. Guelfand),
            ### but here we use a wrap function which gives a relative trigger time(s) of the DU of interest.
            feb_id_n = dict_duID_to_febID[du_id[du_id_n]] # this is to be able to use the FLT0 parameters for each du, which are given in febID, but we want from duIDs
            #print(feb_id_n)
            #FLT0_trig_params_X = get_FLT0_trigger_parameters_du_level(FLT0_trig_params_file, feb_id_n, 'X')
            #FLT0_trig_params_Y = get_FLT0_trigger_parameters_du_level(FLT0_trig_params_file, feb_id_n, 'Y')
            #FLT0_trig_time_X, first_T1_idx_X = get_FLT0_trigger_time(tadc_X_filt, FLT0_trig_params_X, rel_start_time, t_res, for_efficiency_T1_idx=True) # t_res : ADC time resolution
            #FLT0_trig_time_Y, first_T1_idx_Y = get_FLT0_trigger_time(tadc_Y_filt, FLT0_trig_params_Y, rel_start_time, t_res, for_efficiency_T1_idx=True) # t_res : ADC time resolution
            #FLT0_trig_time_Z = get_FLT0_trigger_time(tadc_Z_filt, FLT0_trig_params, rel_start_time, t_res)

            if np.max(tadc_X_filt) > 30: # Then we should keep track of the value, to be able to discuss how this parameter affects the computation
                trig_chnl_list.append([du_id[du_id_n], 'X']) 
                #print("du_id ", du_id[du_id_n], " triggered on X channel with max value ", np.max(tadc_X_filt))
                list_max_tadc_X_filt.append(np.max(tadc_X_filt)) 
            if np.max(tadc_Y_filt) > 30:
                trig_chnl_list.append([du_id[du_id_n], 'Y']) # first filter on the events / channels that we want to keep ; below this value, they are not going to trigger (said Olivier)
                #print("du_id ", du_id[du_id_n], " triggered on Y channel with max value ", np.max(tadc_Y_filt))
                list_max_tadc_Y_filt.append(np.max(tadc_Y_filt))



            ### Make a list of the relative trigger time(s) of all channels
            #for trig_time_m in FLT0_trig_time_X: trig_chnl_list.append([du_id[du_id_n], trig_time_m, 'X', first_T1_idx_X]) # list the du_ids that triggered, the relative trigger time(s) and the channel that triggered # I think there is only one time in FLT0_trig_time, but I keep it as a list for now in case we want to consider multiple crossings of the threshold
            #for trig_time_m in FLT0_trig_time_Y: trig_chnl_list.append([du_id[du_id_n], trig_time_m, 'Y', first_T1_idx_Y])
            #for trig_time_m in FLT0_trig_time_Z: trig_chnl_list.append([du_id[du_id_n], trig_time_m, 'Z'])

            ### END of DU loop

        ### Discuss a coincidence between DUs
        ### A condition to claim a coincidence is
        ### any any_du DUs or more within a time window of t_window nanoseconds


        any_du = 5
        t_window = 1e4
        used_channels = ['X', 'Y'] # Now only X & Y channels are used to discuss a coincidence # so indeed we don’t need to use the Z channel
        #array_trig, array_trig_du_list = discuss_coincidence(trig_chnl_list, any_du, t_window, used_channels)
        array_trig_du_list = []
        new_trig_du_list = [x for x in trig_chnl_list if x[1] in used_channels] # this line I added # add the triggering if the channel is in Y or X

        list_max_trace_X_towrite = []
        list_max_trace_Y_towrite = []


        #print(len(trig_chnl_list), len(new_trig_du_list))
        array_trig = 0

        unique_du_ids = set([x[0] for x in new_trig_du_list]) # keep only unique du_ids
        if len(unique_du_ids) >= any_du: # check whether we have more than 5 triggering DUs, without caring for the time coincidence
            array_trig = 1
            array_trig_du_list = new_trig_du_list
            list_max_trace_X_towrite = [x for x in list_max_tadc_X_filt]
            list_max_trace_Y_towrite = [x for x in list_max_tadc_Y_filt]


        # I want that if "X" is in array_tirg_du_list[k], then max_traces_to_write[k] = max_tadc_X_filt, and if "Y" is in array_tirg_du_list[k], then max_traces_to_write[k] = max_tadc_Y_filt
        max_traces_to_write = []
        kx, ky = 0, 0
        for k in range(len(array_trig_du_list)):
            if array_trig_du_list[k][1] == 'X':
                max_traces_to_write.append(list_max_trace_X_towrite[kx])
                kx += 1
            elif array_trig_du_list[k][1] == 'Y':
                max_traces_to_write.append(list_max_trace_Y_towrite[ky])
                ky += 1
            else:
                max_traces_to_write.append(None) # this should not happen, but just in case

        """
        output.write('{:d}'.format(tshower_l0.run_number)+' ')
        output.write('{:9d}'.format(tshower_l0.event_number)+' ')
        output.write('{:9s}'.format(tshower_l0.primary_type)+' ')
        output.write('{:9.2e}'.format(tshower_l0.energy_primary)+' ')
        output.write('{:9.2f}'.format(tshower_l0.zenith)+' ')
        output.write('{:9.2f}'.format(tshower_l0.azimuth)+' ')
        output.write('{:9.2f}'.format(tshower_l0.shower_core_pos[0])+' ')
        output.write('{:9.2f}'.format(tshower_l0.shower_core_pos[1])+' ')
        output.write('{:9.2f}'.format(tshower_l0.shower_core_pos[2])+' ')
        #output.write('{:6d}'.format(array_trig)+'\n')
        output.write('{:6d}'.format(array_trig)+' ')
        """
        data_output[-1]["fixed"] = [int(tshower_l0.run_number), int(tshower_l0.event_number), float(tshower_l0.primary_type), round(float(np.log10(tshower_l0.energy_primary)), 5), round(float(tshower_l0.zenith), 4), round(float(tshower_l0.azimuth), 4), round(float(tshower_l0.shower_core_pos[0]), 4), round(float(tshower_l0.shower_core_pos[1]), 4), round(float(tshower_l0.shower_core_pos[2]), 3), int(array_trig)]
        data_output[-1]["triggering_events"] = []

        """
        for k in range(len(array_trig_du_list)):
            output.write(str(int(array_trig_du_list[k][0]))+' ')
            output.write(str(int(array_trig_du_list[k][1]))+' ')
            if k < len(array_trig_du_list)-1:
                output.write(str(array_trig_du_list[k][2])+' ')  #array_trig_du_list: list of [du_id, t_trig, arm]
            else:
                output.write(str(array_trig_du_list[i][2]) + " ")  #array_trig_du_list: list of [du_id, t_trig, arm]
        output.write('\n')
        """
        for k in range(len(array_trig_du_list)):
            data_output[-1]["triggering_events"].append([int(array_trig_du_list[k][0]), str(array_trig_du_list[k][1]), round(float((max_traces_to_write[k])), 4)]) #array_trig_du_list: list of [du_id, arm]

    d_input.close()
        
    with open(f"{output_directory}/test_sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{i:04d}_th30.json", "w") as f:
        # I want to write a header :
        #f.write("# for fixed : run number, event number, primary type, log10(primary energy), zenith angle, azimuth angle,  core position (x,y,z), trigger flag \n , For triggereing events : du_id, arm, max value tadc Xfilt, max value tadc Yfilt, \n")    
        json.dump(data_output, f, indent = 2) 


print('judge_trigger_event_level_du_level_channel_level.py ended')
