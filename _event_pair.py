import re
from itertools import combinations
from collections import defaultdict, deque
import sys

# code adapted from Pengxiong’s code, which can be found here : https://github.com/pengxiongma/Reject_BackgroundSource/blob/main/_event_pair.py

def parse_line(line):
    "loading events"
    match = re.match(r"(\d+\.\d+):\s*\[(.*)\]", line)
    if not match:
        return None, []
    
    timestamp = float(match.group(1))
    detectors = []
    for item in match.group(2).split("),"):
        item = item.strip(" ()")
        if not item:
            continue
        detector_id, time = map(int, item.split(","))
        detectors.append((detector_id, time))
    
    return timestamp, detectors

def calculate_pair_differences(detectors):
    """difference of time for each pair"""
    pairs = defaultdict(int)
    for (id1, t1), (id2, t2) in combinations(detectors, 2):
        pair_key = tuple(sorted((id1, id2)))
        pairs[pair_key] = t1 - t2
    return pairs

def process_signals(events_to_build_history, threshold=20, history_size=20, min_pair=5, history = None): # history = None : so that we can input an already existing one if we need to

    prev_pairs = None
    check_CR = True

    #print("events to build history : ", events_to_build_history)

    if history is None:
        check_CR = False
        history = deque(maxlen=history_size)  # repeat pairs saved to history

    for idx, detectors in enumerate(events_to_build_history):

        #print(detectors)
        detected = False

        current_pairs = calculate_pair_differences(detectors)
        is_duplicate = False

        # compare with history 
        for h_idx, historical_pairs in enumerate(history, 1):
            common_pairs = set(current_pairs.keys()) & set(historical_pairs.keys())
            similar_count = sum(
                1 for pair in common_pairs
                if abs(current_pairs[pair] - historical_pairs[pair]) < threshold
            )
            #print(similar_count)
            if similar_count >= min_pair:
                if check_CR== 6:
                    print("\n Our event in T3 trigger is : ", events_to_build_history)
                    print("current pairs", current_pairs)
                    print("historical pairs", {key: historical_pairs[key] for key in historical_pairs if key in current_pairs})
                    print("the check results in :", [abs(current_pairs[pair] - historical_pairs[pair]) for pair in common_pairs])
                    print("result of T3 check is :", common_pairs)
                    print("similar count is :", similar_count)
                is_duplicate = True
                break

        # compare with previous event
        if not is_duplicate and prev_pairs:
            common_pairs = set(current_pairs.keys()) & set(prev_pairs.keys())
            similar_count = sum(
                1 for pair in common_pairs
                if abs(current_pairs[pair] - prev_pairs[pair]) < threshold
            )
            if similar_count >= min_pair:
                is_duplicate = True

        if is_duplicate:
            # only save repeating  pairs into history
            history.append(current_pairs)
            continue

        # save event
        prev_pairs = current_pairs
        detected = True # if we reach this point then we have the T3 trigger

    return history, detected


"""
def main():
    
    output_file = sys.argv[1] #
    
    try:
    
        result = process_signals(events_to_build history)
        
        with open(output_file, 'w') as f:
            for line in result:
                f.write(line + "\n")
        print(f"Done, results saved into {output_file}")
        print(f"lines of raw file: {len(input_lines)}，left lines: {len(result)}")
    
    except FileNotFoundError:
        print(f"Error：not find {input_file}")
    except Exception as e:
        print(f"error in processing: {str(e)}")

if __name__ == "__main__":
    main()

"""