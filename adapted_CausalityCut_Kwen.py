import numpy as np
import sys
from collections import defaultdict

# 光速常数 (speed of light) (m/ns)
C_LIGHT_NS = 299792458.0 / 1e9

def load_data_from_file(filename):
    """Load detector position data from a file"""
    data = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 4:
                        duid = parts[0]
                        try:
                            coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                            data[duid] = coords
                        except ValueError:
                            print(f"Warning: Could not parse coordinates for DU {duid}")
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)
    return data 

def rotate_and_shift_coordinates(coord):
    """
    完全复现 _plot_time_ratio.py 中的坐标变换逻辑 ; Fully reproduce the coordinate transformation logic in plot_time_ratio.py
    """
    rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # that’s identity right ?
    rotated = np.dot(rotation_matrix, coord)
    
    # 应用特定的平移偏移量, Apply a specific translation offset ; if we translate all antennas with the same shift and we compute time differences we really don’t care about that right ?
    rotated[0] -= -3455.6213993713604
    rotated[1] -= -1277.747
    rotated[2] -= 24.02 
    
    return rotated

def precompute_distances(detector_positions): # detector_positions is a dictionary mapping DU IDs to their 3D coordinates
    """预计算所有探测器对之间的距离, Pre-calculate the distances between all pairs of detectors"""
    du_ids = list(detector_positions.keys())
    n = len(du_ids)
    dist_map = {}
    
    for du_id in du_ids:
        dist_map[(du_id, du_id)] = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            id1, id2 = du_ids[i], du_ids[j]
            dist = np.linalg.norm(detector_positions[id1] - detector_positions[id2])
            dist_map[(id1, id2)] = dist
            dist_map[(id2, id1)] = dist
            
    return dist_map

def check_causality_ratio(t1, t2, dist, c_ns=C_LIGHT_NS):
    """
    检查因果律比值 ; check the causality ratio
    逻辑 ; logic : Ratio = t_obs / t_theo
    规则 : Ratio 必须 >= 1.0 才是合法的 (允许微小浮点误差 >= 0.9999) Rule: Ratio must be >= 1.0 to be valid (minor floating-point errors >= 0.9999 are allowed).
    返回: (is_safe, ratio_value) ; back
    """
    if dist == 0:
        return True, float('inf')
    
    t_theo = dist / c_ns
    t_obs = abs(t1 - t2)
    
    if t_obs == 0:
        return True, 0.0
    
    ratio = t_obs / t_theo
    is_safe = ratio < 1.002 # ie it’s true when ratio is less than 1.002, while we should return true when the ratio is greater than 1 right ?
    
    return is_safe, ratio

def optimized_read_matching_times_graph(rotated_positions, matches_list, min_detectors=5):
    # matches_list is a list of tuples, where each tuple is (DU_ID, time_ns)
    # returns 0 if the event is indeed not valid, 1 if it is valid
    matching_times = []
    

    # 1. 预计算距离 ; estimate the distance
    dist_map = precompute_distances(rotated_positions)
    
    valid_events = 0     
    gps_time = 0. # we don’t care about that
    
    # === 步骤1: 解析节点 (所有触发都作为独立节点) === Step 1: Parse the nodes (all triggers are treated as separate nodes) 
    nodes = [] 
    node_info = {} 
    du_triggers = defaultdict(list) 
    
    node_id = 0

    if len(matches_list) < min_detectors:
        print(f"Careful, len(matches_list) < {min_detectors}, skipping this event")
        print(matches_list)
        return False
    
    for match in matches_list:

        if len(match) < 2:
            print(f"Warning: Invalid match format '{match}'. Skipping.")
            continue
        
        det_id = match[0]
        try:
            time_ns = int(match[1])
        except ValueError: continue
        
        if det_id not in rotated_positions: # this I guess should not happen
            #print("Warning: Detector ID {} not found in positions. Skipping.".format(det_id))
            continue 
        
        # 每个触发都是一个独立的节点，即使属于同一个 DU ; Each trigger is an independent node, even if it belongs to the same DU.
        nodes.append((det_id, time_ns, node_id)) # so nodes is a list of tuples, with DU_ID, time_ns and an associated identifier
        node_info[node_id] = (det_id, time_ns)
        du_triggers[det_id].append((time_ns, node_id))
        node_id += 1
    
    if len(du_triggers) < min_detectors: # should I set min_detectors to 5 ? ; that’s the same test as the one with matches_list above
        print(f"Careful, len(du_triggers) < {min_detectors}, skipping this event")
        return False
    
    # === 步骤2: 构建严格因果律图 === : construct a strict causal graph
    n_nodes = len(nodes)
    neighbors = {i: set() for i in range(n_nodes)} # neighbours is a dictionary of sets, key is node index
    
    for i in range(n_nodes):
        du_i, time_i, _ = nodes[i]
        for j in range(i + 1, n_nodes): # in one node you have basically one DU and the associated time
            du_j, time_j, _ = nodes[j]
            
            # 【关键修改】同一 DU 的不同触发之间不连边                      # [Key Change] No edges between different triggers within the same DU
            # 最大团算法会自动确保一个 DU 只选出一个节点（因为团内必须两两相连）;  # The maximum clique algorithm automatically ensures that only one node is selected per DU (since all nodes within a clique must be connected to each other)
            if du_i == du_j:
                continue
            
            dist = dist_map.get((du_i, du_j))
            if dist is None: continue
            
            is_safe, ratio = check_causality_ratio(time_i, time_j, dist)
            
            # 只有满足因果律 (Ratio >= 1.0) 才连边 : Edges are connected only if the causality rule is satisfied (Ratio >= 1.0).
            if is_safe:
                neighbors[i].add(j)
                neighbors[j].add(i)
    
    # === 步骤3: 寻找最大团 (贪心算法) === ; Step 3: Finding the Largest Cluster (Greedy Algorithm)
    # 此时不再需要 used_du 集合，因为图的结构已经保证了同一 DU 不会同时入选 ;  At this point, the `used_du` set is no longer needed, because the graph's structure ensures that the same DU cannot be selected more than once.
    node_degrees = {i: len(neighbors[i]) for i in range(n_nodes)}
    sorted_candidates = sorted(range(n_nodes), key=lambda x: node_degrees[x], reverse=True)
    
    clique = []
    
    for candidate in sorted_candidates:
        # 检查候选节点是否与当前团内所有节点相连 ; Check whether the candidate node is connected to all nodes in the current cluster
        is_compatible = all(member in neighbors[candidate] for member in clique)
        
        if is_compatible:
            clique.append(candidate)
    
    # === 步骤4: 强制后处理净化 (Hard Filter) === ; Step 4: Force Post-Processing Purification (Hard Filter)
    # 再次检查团内所有点对，剔除导致因果律违规 (Ratio < 1.0) 的节点 ; Recheck all pairs within the group and remove any nodes that violate the law of causality (Ratio < 1.0).
    final_clique = list(clique)
    changed = True
    while changed and len(final_clique) > min_detectors:
        changed = False
        worst_node = -1
        worst_violation_score = -1.0 
        
        for idx, node in enumerate(final_clique):
            du_n, time_n, _ = nodes[node]
            violation_score_sum = 0.0
            has_violation = False
            
            for other in final_clique:
                if other == node: continue
                du_o, time_o, _ = nodes[other]
                
                # 防御性检查，理论上团内不会有同 DU ; As a precautionary measure, in theory, there should be no one with the same DU within the group.
                if du_n == du_o: continue 
                
                dist = dist_map.get((du_n, du_o), 0.0)
                if dist == 0: continue
                
                is_safe, ratio = check_causality_ratio(time_n, time_o, dist)
                
                if not is_safe:
                    has_violation = True
                    score = 1.0 - ratio
                    violation_score_sum += score
            
            if has_violation:
                if violation_score_sum > worst_violation_score:
                    worst_violation_score = violation_score_sum
                    worst_node = node
        
        if worst_node != -1:
            final_clique.remove(worst_node)
            changed = True
    
    # === 步骤5: 最终验证与统计 === ;  Step 5: Final Verification and Statistics
    is_final_safe = True
    for i in range(len(final_clique)):
        if not is_final_safe: break
        for j in range(i+1, len(final_clique)):
            n1, n2 = final_clique[i], final_clique[j]
            d1, t1, _ = nodes[n1]
            d2, t2, _ = nodes[n2]
            if d1 == d2: continue
            dist = dist_map.get((d1, d2), 0)
            safe, _ = check_causality_ratio(t1, t2, dist)
            if not safe:
                is_final_safe = False
                break
    
    if is_final_safe and len(final_clique) >= min_detectors:
        valid_matches = []
        for node_id in final_clique:
            du_id, time_ns = node_info[node_id]
            valid_matches.append((du_id, time_ns))
        
        # 统计：现在 selected_du_set 的大小应该等于 final_clique 的大小 ; Statistics: The size of `selected_du_set` should now be equal to the size of `final_clique`.
        selected_du_set = {node_info[n][0] for n in final_clique}
        all_triggered_du = set(du_triggers.keys())
        removed_du = all_triggered_du - selected_du_set
        
        # 计算同 DU 多余触发：如果一个 DU 有 N 个触发，选了 1 个，则多余 N-1 个 ; # Calculating the number of redundant triggers for a DU: If a DU has N triggers and 1 is selected, there are N-1 redundant triggers.
        extra_triggers_count = sum(len(du_triggers[d]) - 1 for d in selected_du_set if len(du_triggers[d]) > 1)
        
        if removed_du or extra_triggers_count > 0:
            # 详细打印哪个 DU 的哪个时间被保留了，哪个被剔除了（可选）; Print a detailed list of which DU entries were retained and which were excluded at what times (optional)
            print( f"Retained {len(final_clique)}个DU, "
                    f"excluded DU={removed_du}, "
                    f"Eliminate redundant triggers with the same DU={extra_triggers_count}个.")
        
        matching_times.append((gps_time, valid_matches))
        valid_events += 1
        return True
    else:
        return False

    """
    # === 步骤6: 写入 output.txt === ( Write )
    if len(sys.argv) > 2:
        output_filename = sys.argv[2]
    else:
        output_filename = "output.txt"

    try:
        with open(output_filename, 'w') as f_out:
            for gps_time, matches in matching_times:
                matches_str_parts = [f"('{du}', {time})" for du, time in matches]
                matches_str = ", ".join(matches_str_parts)
                f_out.write(f"{gps_time}: [{matches_str}]\n")
        
        print(f"\n========================================")
        print(f"Processing complete!")
        print(f"Total lines processed: {total_lines}")
        print(f"Total valid events retained: {valid_events}")
        print(f"Strategy: Automatically select the optimal combination from multiple triggers within the same DU (Max Clique)")
        print(f"Strict validation criteria: (t_obs / t_theo) >= 1.0 for all pairs of DUs")
        print(f"Results saved to: {output_filename}")
        
    except IOError as e:
        print(f"Error writing to output file {output_filename}: {e}")

    return matching_times
    """

# I’d say that in raw_positions we should have DU_ID as key and associated coordinates