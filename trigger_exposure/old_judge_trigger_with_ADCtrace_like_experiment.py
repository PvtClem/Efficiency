'''
This program discusses the trigger in the realistic situation.

The trigger logic implemented in offline_FLT0_trigger.py is used
to discuss the trigger of each channel of a DU.

Currently, only the trigger of the X & Y channels is focused on
(the trigger of the Z channel is NOT focused on).
See the GP300_log.docx for referrence, where you can see only the trigger threshold
of the X & Y channels are discussed.

Also, the MGuelfand's study of FLT0 trigger also focuses on the X & Y channels.
(See her slides used in GRAND2025 meeting @ Warsaw)

The ADC trace is filtered with 
1. the FIR filter (cut > 115 MHz, see filter_traces_bandpass in utils.py) + one notch filter @ 39 MHz (see apply_notch_filter in utils.py), or 
2. four notch filter @ 39, 119.4, 132, & 137.8 MHz.
Currnetly, 1. is the case in the real experiment.

Then the trigger is discussed using the filtered ADC trace.
The trigger parameters are as follows:
1. For FIR + one notch filters:
   t_quiet = 500 (ns),
   t_period = 500 (ns),
   t_sepmax = 50 (ns),
   nc_min, nc_max = 2, 7 &
   th1, th2 = 55, 48 (ADC counts).

2. For four notch filters:
   t_quiet = 500 (ns),
   t_period = 500 (ns),
   t_sepmax = 50 (ns),
   nc_min, nc_max = 2, 7 &
   th1, th2 = 70, 60 (ADC counts).
These values are saved in dict_trig_params_fir.csv and dict_trig_params_notch.csv, 
either of which are fed into the execution of this program. 

Currently the trigger is discussed independently for the X & Y channels,
and OR of X & Y is taken to issue (or not) the trigger of the DU.

The conditions, currently implemented in this program, to find a coincidence between DUs is,
any 4 DUs within a time window of 10,000 ns (= 10 micro seconds). 
See the values of the parameters any_du & time_window.
These are the parameters used in the current experiment (Bohao, as of June 2025).

### Arguments of this program:
1st argument: filter_mode (='none', 'fir', or 'notch', currently)
2nd argument: dict_trig_params_file (='dict_trig_params_fir.csv' for filter_mode='fir', and
                                      'dict_trig_params_notch.csv' for filter_mode='notch')
3rd argument: directory, which specifies the directory where simulation data is analyzed.
              All direcotires are listed in COREAS-AN_sim_data_directory.txt.

How to execute:
If you want to apply fir filter to the ADC time trace:
python judge_trigger_with_ADCtrace_like_experiment.py fir dict_trig_params_fir.csv /sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_0000

If you want to execute the program to analyse all simulation data at once,
you can execute job.sh in SLURM by executing "sh submit_job.sh".
Please make sure to edit the value of the 'thisdir' parameter in job.sh & submit_job.sh before the execution.
'''

import csv
import sys
import grand.dataio.data_handling as groot
import grand.manage_log as mlg
#import raw_root_trees as RawTrees
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from utils import *
from offline_FLT0_trigger import trigger_FLT0

### Trigger parameters
# any_du or more DUs should be triggered within time_window (ns)
any_du = 4 
time_window = 1e4

### Other parameters
### time resolution of our sampling (= 1 / f_sample)
time_resolution = 2 # ns
f_sample = 500e6 # (Hz)

filter_mode = sys.argv[1]
dict_trig_params_file = sys.argv[2]
directory = sys.argv[3]
out_dir   = "out_judge_trigger_with_ADCtrace_like_experiment/"

d_input = groot.DataDirectory(directory)
output  = open(out_dir+directory[directory.find('sim'):],'w')

### For the calculation related to GP65:
### Read the ID of the GP65 DUs.
### Only these DUs are used for discussing a trigger
#gp65_du = np.array([[-1, -1, -1, -1]])
#gp65_du = np.concatenate([gp65_du, np.loadtxt('../GP65_layout_for_read.txt')], axis=0)
#gp65_du = np.delete(gp65_du, 0, axis=0)
#gp65_du_id = gp65_du[:,0].astype(int)
#gp65_du_px, gp65_du_py, gp65_du_pz = gp65_du[:,1], gp65_du[:,2], gp65_du[:,3]
###print("gp65_du:", gp65_du)
###print("gp65_du.shape[0]", gp65_du.shape[0])
###print("gp65_du_id:", gp65_du_id)

tadc_l1 = d_input.tadc_l1
trun_l1 = d_input.trun_l1
tshower_l0 = d_input.tshower_l0

events_list = tadc_l1.get_list_of_events()
nb_events = len(events_list)

if nb_events == 0:
    sys.exit("No events in the file. Exiting.")

####################################################################################
# start looping over the events
####################################################################################
previous_run = None
events = 0

for event_number,run_number in events_list:
    assert isinstance(event_number, int)
    assert isinstance(run_number, int)

    tadc_l1.get_event(event_number, run_number)
    tshower_l0.get_event(event_number, run_number)
    
    if previous_run != run_number:
        trun_l1.get_run(run_number)
        previous_run = run_number
        
    run_num = tshower_l0.run_number
    eve_num = tshower_l0.event_number
    pritype = tshower_l0.primary_type
    energy  = tshower_l0.energy_primary
    zenith  = tshower_l0.zenith
    azimuth = tshower_l0.azimuth
    shower_core_pos = tshower_l0.shower_core_pos
    output.write('{:d}'.format(run_num)+' ')
    output.write('{:9d}'.format(eve_num)+' ')
    output.write('{:9s}'.format(pritype)+' ')
    output.write('{:9.2e}'.format(energy)+' ')
    output.write('{:9.2f}'.format(zenith)+' ')
    output.write('{:9.2f}'.format(azimuth)+' ')
    output.write('{:9.2f}'.format(shower_core_pos[0])+' ')
    output.write('{:9.2f}'.format(shower_core_pos[1])+' ')
    output.write('{:9.2f}'.format(shower_core_pos[2])+' ')

    du_id = np.array(tadc_l1.du_id)
    tadc_trace = np.array(tadc_l1.trace_ch) # dimension: (# of du_id, XYZ, samples)
    rel_trace_start_time_ns = calculate_relative_trace_start_time(tshower_l0, tadc_l1, time_resolution)

    trig_time_du_id = []
    du_id_and_arm = []
  
    for n in range(du_id.shape[0]):

        ### For the calculation related to GP65:
        ### Read the ID of the GP65 DUs.
        ### Only these DUs are used for discussing a trigger
        #if np.any(gp65_du_id == du_id[n]) == False: continue
        ###print("Now du_id[n] is", du_id[n])
        
        tadc_trace_X = tadc_trace[n][0]
        tadc_trace_Y = tadc_trace[n][1]
        tadc_trace_Z = tadc_trace[n][2]
        
        tadc_trace_X = filter_trace(filter_mode, tadc_trace_X, f_sample)
        tadc_trace_Y = filter_trace(filter_mode, tadc_trace_Y, f_sample)
        tadc_trace_Z = filter_trace(filter_mode, tadc_trace_Z, f_sample)

        ### Read trigger parameters (t_quiet, t_period, etc.)
        dict_trig_params = csv.DictReader(open(dict_trig_params_file))
        for row in dict_trig_params: trigger_parameters = row
        trigger_parameters = {k: int(v) for k, v in trigger_parameters.items()}
        ###print("trigger_parameters:", trigger_parameters)
        
        T1_idxs_X, T1_amps_X, NC_vals_X = trigger_FLT0(tadc_trace_X, trigger_parameters)
        T1_idxs_Y, T1_amps_Y, NC_vals_Y = trigger_FLT0(tadc_trace_Y, trigger_parameters)
        T1_idxs_Z, T1_amps_Z, NC_vals_Z = trigger_FLT0(tadc_trace_Z, trigger_parameters)
        
        ### Now only the trigger of the X & Y arms are discused
        if T1_idxs_X or T1_idxs_Y:
            for m in range(len(T1_idxs_X)):
                trig_time_ns = rel_trace_start_time_ns[n] + T1_idxs_X[m] * time_resolution
                trig_time_du_id.append([trig_time_ns, du_id[n], 'X'])
                du_id_and_arm.append([du_id[n], 'X'])
            for m in range(len(T1_idxs_Y)):
                trig_time_ns = rel_trace_start_time_ns[n] + T1_idxs_Y[m] * time_resolution
                trig_time_du_id.append([trig_time_ns, du_id[n], 'Y'])
                du_id_and_arm.append([du_id[n], 'Y'])
            #for m in range(len(T1_idxs_Z)):
            #    trig_time_ns = rel_trace_start_time_ns[n] + T1_idxs_Z[m] * time_resolution
            #    trig_time_du_id.append([trig_time_ns, du_id[n], 'Z'])
            
    # Discuss a coincidence between DUs
    trig, trig_list = discuss_coincidence(trig_time_du_id, any_du, time_window)

    output.write('{:6d}'.format(trig)+'\n')
    #print("trig_list:", trig_list)
    #output.write('{:6d}'.format(has_duplicates(du_id_and_arm))+'\n')

print('End of judge_trigger_with_ADCtrace_like_experiment.py')
