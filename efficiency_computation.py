# code for evaluating the efficiency of GP80 (or 65 for all I know)

import grand.dataio.data_handling as dh

import sys
import os
sys.path.append(os.path.abspath("/sps/grand/cprevotat/grand/grand/grand/exposure/")) # this is not very clean, but it was not to mess with Sei’s code, and not copy everything here
from utils import calculate_PAO_spectrum, calculate_relative_trace_start_time, get_FLT0_trigger_parameters_du_level, get_FLT0_trigger_parameters, get_FLT0_trigger_time, notch_filter, filter_traces_bandpass, running_DUs, integrate_PAO_spectrum
import functions
import adapted_CausalityCut_Kwen

sys.path.append(os.path.abspath("/sps/grand/cprevotat/grand/efficiency/")) # this is not very clean, but it was not to mess with Sei’s code, and not copy everything here
#from dict_th1_th2 import data_dict


import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os.path


from pathlib import Path
import gc
import json
import cProfile

import struct
from datetime import datetime, timezone




"""
This code deals with the efficiency computation, starting from drawing t0 for each event
to writing into output files which events (and their associated parameters)
passed the trigger, up to the T3 trigger.
Before running it, one should run something like that : 
#python /sps/grand/cprevotat/grand/judge_trigger_event_du_level_channel_level.py "/sps/grand/cprevotat/grand/grand/grand/exposure/dict_trig_params_fir.csv" 
#for 1 simulation file 30s and 900 MB, 1h12 for the 150 files
#the data directory is by default set to use Coreas no noise simulations, 
this can be change directly in judge_trigger_event_du_level_channel_level.py
this line create the json files, and writes at the du level which events / DUs have a
going trace above a given threshold (in ADC count), so that we don’t simulate too many
DUs / events (and it also checks whether we have more than 5 DUs that meet this condition)

"""





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
    return


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



def extract_noise_from_t0_duIdlist(InputDataFiles, target_timestamp, target_duid): # seems to be the location of the data files, timestamp in second, duid that we are looking for (should be feb)
    #now target_duid is a list of the target_duid, so that we read the file only once per t0 time
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

    for filename in InputDataFiles:

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


def judge_trigger_true_events(output_directory, list_list_MD_files, sim_number, t0): 
    # output_directory : where the json files (output files) should already be (from judge_trigger_event......py)
    # list_list_MD_files : a list of list of MD files, each sublist correspond to a t0 ;
    # sim_number : number of the simulation we are processing


    data_directory = f"/sps/grand/DC2_Coreas/Coreas_nonoise/sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}" # ensure correct formating 
    with open(f"{output_directory}/test_sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", "r") as f:
        first_judge_data = json.load(f)


    run_numbers = [row["fixed"][0] for row in first_judge_data] # select the run numbers of the events that passed the first trigger
    event_numbers = [row["fixed"][1] for row in first_judge_data] # select the event numbers of the events that passed the first trigger

    triggering_events = [row["triggering_events"] for row in first_judge_data] 
    #print(triggering_events)


    ### Read GRAND root data
    d_input = dh.DataDirectory(data_directory)
    tadc_l1, tshower_l0 = d_input.tadc_l1, d_input.tshower_l0

    #dict_duID_to_febID = generate_correspondance_duID_to_febID() # this is to be able to use the FLT0 parameters for each du, which are given in febID, but we want from duIDs
    #dict_febID_to_duID = generate_correspondance_febID_to_duID() # this is to be able to use the FLT0 parameters for each du, which are given in febID, but we want from duIDs

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
                
    with open(f"{output_directory}/test_sim_Dunhuang_20170331_220000_RUN1_CD_GP300-no-noise_{sim_number:04d}_th30.json", "w") as f:
        json.dump(first_judge_data, f, indent = 2) # I’d say that indent is useless, only for readability


    return 


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


# the real code here : 

if __name__ == "__main__":

    output_directory = "/sps/grand/cprevotat/grand/efficiency/out_judge_trigger_du_channel_level" # this is where the json files (output files) should already be (from judge_trigger_event......py)

    dict_duID_to_febID = functions.generate_correspondance_duID_to_febID()
    dict_febID_to_duID = functions.generate_correspondance_febID_to_duID()

    f_sample = 500e6 # Hz, ADC sampling rate
    FLT0_trig_params_file = "/sps/grand/cprevotat/grand/grand/grand/exposure/dict_trig_params_fir.csv"

    ordered_files, t_min_ordered, t_max_ordered, liste_idx_in_intervals = prepare_for_drawing_t0(filename = "/sps/grand/cprevotat/grand/efficiency/out_extract_infos/allowed_times_t0gps_2026_05.txt")
    with open("/sps/grand/cprevotat/grand/efficiency/out_extract_infos/time_period_2026_05.txt", "w") as f:
        # The total observation time (considering only times when the detector was running) is (in seconds) :
        total_time = np.sum(t_max_ordered - t_min_ordered)
        f.write(str(total_time) )
    # with t_min_ordered and t_max_ordered I do get the total time period we are looking at, I can use it to get the time period

    sim_numbers = [i for i in range(0, 1)]
    for sim_number in sim_numbers:
        print("dealing with simulation number ", sim_number, "out of ", len(sim_numbers))
        t0, idx = draw_t0(100, t_min_ordered, t_max_ordered) #it’s 100 because we have at max 100 events in Coreas files, for Zhaires it should be more

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
        #judge_trigger_true_events(output_directory, new_list, sim_number = sim_number, t0 = t0) 
        cProfile.run("judge_trigger_true_events(output_directory,new_list, sim_number = sim_number, t0 = t0)", "/sps/grand/cprevotat/grand/efficiency/profile_results_listdu2.prof")
        