# stimuli_generation.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
from numpy.random import randint, choice as randchoice

def load_stimuli(stim_folder="assets"):
    """Load and categorize stimuli from the folder."""
    stim_set = [f for f in os.listdir(stim_folder) if f.endswith(".png")]
    unique_stim = [f for f in stim_set if "set" not in f]
    linking_stim = [f for f in stim_set if "set" in f]

    # separate linking stimuli by set
    linking_stim_by_set = {}
    for f in np.sort(linking_stim):
        set_id = f.split("_")[0]
        linking_stim_by_set.setdefault(set_id, []).append(f)

    return unique_stim, linking_stim_by_set

def generate_pairs(exp_info, unique_stim, linking_stim_by_set):
    """
    Generate AB, BC, CD stimulus groups for the SL-Obj-Inference task.
    
    Returns:
        abcd_groups (dict): dict with A, B, C, D groups (stimulus assignments)
    """
    # Seed random generator per subject for reproducibility
    np.random.seed(int(exp_info["subject"]))

    b_stim, c_stim = {}, {}
    pair_counter = 0

    # --- Generate B-C pairs ---
    for set_id, imgs in linking_stim_by_set.items():
        # first BC pair: img-02 + random(img-01/img-03)
        first_item = imgs[1]  # img-02
        second_item = np.random.choice([imgs[0], imgs[2]])
        if np.random.randint(0, 2) == 0:
            b_stim[f"B{pair_counter+1}"], c_stim[f"C{pair_counter+1}"] = first_item, second_item
        else:
            b_stim[f"B{pair_counter+1}"], c_stim[f"C{pair_counter+1}"] = second_item, first_item
        pair_counter += 1

        # second BC pair: img-04 + (img-03/img-05)
        first_item = imgs[3]  # img-04
        if imgs[2] not in c_stim.values() and imgs[2] not in b_stim.values():
            second_item = np.random.choice([imgs[2], imgs[4]])
        else:
            second_item = imgs[4]
        if np.random.randint(0, 2) == 0:
            b_stim[f"B{pair_counter+1}"], c_stim[f"C{pair_counter+1}"] = first_item, second_item
        else:
            b_stim[f"B{pair_counter+1}"], c_stim[f"C{pair_counter+1}"] = second_item, first_item
        pair_counter += 1

    # collect unused
    used_imgs = list(b_stim.values()) + list(c_stim.values())
    unused_imgs = [img for imgs in linking_stim_by_set.values() for img in imgs if img not in used_imgs]
    remaining_stim = unique_stim + unused_imgs

    # --- Generate A and D stimuli ---
    a_stim, d_stim = {}, {}
    used_sets = set()
    bc_pairs = list(zip(b_stim.values(), c_stim.values()))
    ad_stim_choices = remaining_stim.copy()

    for i, (b_img, c_img) in enumerate(bc_pairs):
        while True:
            # if no more valid candidates, restart all A/D assignments
            if len(ad_stim_choices) < 2:
                a_stim.clear()
                d_stim.clear()
                used_sets.clear()
                ad_stim_choices = remaining_stim.copy()
                i = -1  # reset loop to start (will increment to 0 in next iteration)
                bc_pairs = list(zip(b_stim.values(), c_stim.values()))
                break
            
            candidate_a = np.random.choice(ad_stim_choices)
            candidate_d = np.random.choice(ad_stim_choices)
            set_a = candidate_a.split("_")[0]
            set_d = candidate_d.split("_")[0]
            forbidden_sets = {b_img.split("_")[0], c_img.split("_")[0]}
            
            if (set_a != set_d and 
                set_a not in forbidden_sets and set_d not in forbidden_sets and
                set_a not in used_sets and set_d not in used_sets
            ):
                a_stim[f"A{i+1}"] = candidate_a
                d_stim[f"D{i+1}"] = candidate_d
                used_sets.update({set_a, set_d})
                ad_stim_choices.remove(candidate_a)
                ad_stim_choices.remove(candidate_d)
                break


    # shuffle the 6 rows of each stimulus assignment
    row_indices = np.arange(len(a_stim))
    np.random.shuffle(row_indices)
    # reorder each stim dict by the shuffled indices
    a_stim = {f"A{i+1}": a_stim[f"A{row_indices[i]+1}"] for i in range(len(a_stim))}
    b_stim = {f"B{i+1}": b_stim[f"B{row_indices[i]+1}"] for i in range(len(b_stim))}
    c_stim = {f"C{i+1}": c_stim[f"C{row_indices[i]+1}"] for i in range(len(c_stim))}
    d_stim = {f"D{i+1}": d_stim[f"D{row_indices[i]+1}"] for i in range(len(d_stim))}

    abcd_groups = {"A": a_stim, "B": b_stim, "C": c_stim, "D": d_stim}

    # --- Save assignments to CSV ---
    df_rows = []
    num_pairs = len(abcd_groups["A"])  # assume A, B, C, D all same length

    for pair_idx in range(num_pairs):
        for stim_label in ["A", "B", "C", "D"]:
            img = list(abcd_groups[stim_label].values())[pair_idx]
            df_rows.append({
                "subject": exp_info["subject"],
                "session": exp_info["session"],
                "group_num": pair_idx + 1,   # 1-based
                "stim_label": stim_label,
                "image": img
            })

    df = pd.DataFrame(df_rows)
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{exp_info['file_prefix']}_stim-assignments.csv"), index=False)

    return abcd_groups


def plot_stimuli(exp_info, abcd_groups, stim_folder="assets"):
    """Plot all assigned stimuli for QA."""
    fig, axes = plt.subplots(6, 4, figsize=(18, 18))
    for i, (group, items) in enumerate(abcd_groups.items()):
        for j, (label, img) in enumerate(items.items()):
            ax = axes[j, i]
            img_data = mpimg.imread(os.path.join(stim_folder, img))
            ax.imshow(img_data)
            ax.set_title(f"{group}{j+1}")
            ax.axis("off")
    plt.tight_layout()
    
    out_dir = "qa"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{exp_info['file_prefix']}_stim-groups.png")
    plt.savefig(save_path)
    plt.close()
    
    
def generate_obj_stream(exp_info, abcd_groups):
    """
    Generate the object stream for the test phase.
    Returns:
        obj_stream (list): list of dicts with 'trial_num', 'image', 'is_1back'
    """
    
    # --- Parameters ---
    num_stim = exp_info["num_stim|hid"]
    num_grps = exp_info["num_grps|hid"]
    prob_1back = exp_info["prob_1back|hid"]
    reps = exp_info["num_reps|hid"]
    stim_1back = int(reps * prob_1back * 2)         # number of 1-back trials for a stimulus
    n_pairs_stream1 = num_grps * 2                  # number of groups in the 1st visual stream (A-B and C-D)
    n_pairs_stream2 = num_grps                      # number of groups in the 2nd visual stream (B-C)
    n_pair_trls_stream1 = n_pairs_stream1 * reps    # total number of paired trials in the 1st visual stream
    n_pair_trls_stream2 = n_pairs_stream2 * reps    # total number of paired trials in the 2nd visual stream
    n_stim_trls_stream1 = n_pair_trls_stream1 * 2   # total number of trials in the 1st visual stream
    n_stim_trls_stream2 = n_pair_trls_stream2 * 2   # total number of trials in the 2nd visual stream
    n_stim_per_grp = num_stim // num_grps           # number of stimuli per group
    pair_idx1, pair_idx2 = [1, 0], [0, 1]           # 1-back index for the 1st and 2nd image in the pair
    #block_test_pairs = {"AB": 0, "BC": 1, "CD": 2}  # mapping of condition IDs to block test pairs
    # pair_indices_streamX = n_pairs_stream1/num_grps -> [0,1,2,3,4,5,0,1,2,3,4,5,...]
    pair_indices_stream1 = [i % num_grps for i in range(n_pairs_stream1)] # [0,1,2,3,4,5,0,1,2,3,4,5,...]
    pair_indices_stream2 = [i % num_grps for i in range(n_pairs_stream2)] # [0,1,2,3,4,5,...]
    n_each_bin = reps
    n_bins = 5
    n_bins_stream1 = n_pair_trls_stream1 // n_each_bin
    n_bins_stream2 = n_pair_trls_stream2 // n_each_bin
    bins_stream1 = getbins(n_pair_trls_stream1, n_bins)
    bins_stream2 = getbins(n_pair_trls_stream2, n_bins)

    print(f"stream 1: trials = {n_pair_trls_stream1}\t bins={bins_stream1}\n"
          f"stream 2: trials = {n_pair_trls_stream2}\t bins={bins_stream2}\n")

    n_trls_stream1 = n_stim_trls_stream1 + (n_stim_trls_stream1 * prob_1back)
    n_trls_stream2 = n_stim_trls_stream2 + (n_stim_trls_stream2 * prob_1back)
    
    success = False
    attempts = 0
    maxAttempts = 1000
    
    while not success and attempts < maxAttempts:
        attempts += 1
        try:
            indices_stream1 = getsequences(n_pairs_stream1, reps, pair_indices_stream1)
            indices_stream2 = getsequences(n_pairs_stream2, reps, pair_indices_stream2)
            
            print(f"indices stream 1 (len={len(indices_stream1)}): {indices_stream1}")
            print(f"indices stream 2 (len={len(indices_stream2)}): {indices_stream2}")

            list_1back_stream1 = zeroslike(indices_stream1)
            list_1back_stream2 = zeroslike(indices_stream2)

            group_1back_stream1, group_1back_stream2 = [], []
            exclude_1back_stream1, exclude_1back_stream2 = [], []
            
            assignOneBacks(n_pairs_stream1, indices_stream1, bins_stream1, exclude_1back_stream1,
                           list_1back_stream1, group_1back_stream1, "1st visual stream", stim_1back,
                           pair_idx1, pair_idx2)
            assignOneBacks(n_pairs_stream2, indices_stream2, bins_stream2, exclude_1back_stream2,
                           list_1back_stream2, group_1back_stream2, "2nd visual stream", stim_1back,
                           pair_idx1, pair_idx2)
            success = True
        except Exception as e:
            print(f"⚠️ Restarting attempt {attempts}: {e}")
    if not success:
        raise RuntimeError(f"Failed to assign 1-backs after {maxAttempts} attempts")     

    exposure_obj_stream = []
    
    return exposure_obj_stream

# ============================================================
# Core 1-Back Assignment Function
# ============================================================

def assignOneBacks(numPairs, idxByPair, bins, excluded, onebackArray,
                   onebackPair, label, stimOneback, pairIdx1, pairIdx2):
    for i in range(numPairs):
        idxPair = where(idxByPair, lambda val, _: val == i)
        binCounts = np.random.multinomial(stimOneback, [1/(len(bins)-1)]*(len(bins)-1))
        onebackIdxs = []

        for j in range(len(bins) - 1):
            validIdx = negbool(isin(idxPair, excluded))
            binFiltered = [idx for k, idx in enumerate(idxPair)
                           if validIdx[k] and bins[j] <= idx < bins[j + 1]]

            if len(binFiltered) >= binCounts[j]:
                onebackIdxs.extend(sample(binFiltered, binCounts[j]))
            else:
                print(f"⚠️ No valid samples for {label} (pair {i}, bin {j}) — restarting")
                raise ValueError(f"No valid samples for {label}")

        excluded.extend(expandselct(onebackIdxs))
        onebackIdxsShuff = shuffle(onebackIdxs)
        onebackPair.extend(onebackIdxsShuff)

        for k, idx in enumerate(onebackIdxsShuff):
            onebackArray[idx] = pairIdx1 if k < stimOneback / 2 else pairIdx2

# ============================================================
# Helper functions for 1-back logic
# ============================================================

def getbins(end, num):
    start = 1
    step = (end - start) / (num - 1)
    result = []
    for i in range(num):
        val = start + i * step
        result.append(int(val))
    result[-1] -= 2 # subtract 2 because of zero-based indexing and to ensure the last trial is not a 1-back trial
    return result

def range_list(n, start=0, step=1):
    return list(range(start, n, step))

def sample(array, count):
    return random.sample(array, count)

def shuffle(arr):
    a = arr.copy()
    random.shuffle(a)
    return a

def randint_int(min_, max_):
    return random.randint(min_, max_ - 1)

def zeroslike(arr):
    if isinstance(arr[0], list):
        return [[0 for _ in row] for row in arr]
    else:
        return [0 for _ in arr]

def where(arr, predicate):
    return [i for i, v in enumerate(arr) if predicate(v, i)]

def isin(array, values):
    valueset = set(values)
    return [x in valueset for x in array]

def negbool(bool_list):
    return [not b for b in bool_list]

def expandselct(indices, by=1):
    expanded = []
    for i in indices:
        expanded.extend([i - by, i, i + by])
    return expanded

def weightedchoice(population, weights):
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("No weights > 0")
    r = random.random() * total_weight
    for item, w in zip(population, weights):
        if w > 0:
            if r < w:
                return item
            r -= w
    raise ValueError("Choice error")

def sendback(value, number, lst):
    idx = len(lst) - 2
    for _ in range(number):
        while lst[idx] == value or lst[idx - 1] == value:
            idx -= 1
        lst.insert(idx, value)

def getsequences(nbvalues, repeats, pair_indices, distribution_bins=5):
    """
    Generate sequences with constraints and even distribution.
    
    Args:
        nbvalues: Number of unique indices (0 to nbvalues-1)
        repeats: How many times each index should appear
        pair_indices: List mapping each index to a pair group (for consecutive constraint)
        distribution_bins: Number of bins to distribute repetitions across for even distribution
    
    Returns:
        List of indices with constraints satisfied and even distribution
    """
    population = range_list(nbvalues)
    total_length = nbvalues * repeats
    
    print(f"getsequences: nbvalues={nbvalues}, repeats={repeats}, pair_indices={pair_indices}")
    print(f"population: {population}, distribution_bins: {distribution_bins}")
    
    # If distribution_bins is 1 or not specified properly, use original algorithm
    if distribution_bins <= 1:
        return _getsequences_original(nbvalues, repeats, pair_indices)
    
    # Create bins for even distribution
    bin_size = total_length // distribution_bins
    all_sequences = []
    
    for bin_idx in range(distribution_bins):
        bin_start = bin_idx * bin_size
        bin_end = (bin_idx + 1) * bin_size if bin_idx < distribution_bins - 1 else total_length
        bin_length = bin_end - bin_start
        
        # Calculate repetitions per bin for each index
        # Distribute evenly with remainder distributed across different bins for different indices
        bin_weights = []
        for i in range(nbvalues):
            base_reps = repeats // distribution_bins
            # Stagger the extra repetitions so different indices get extras in different bins
            if (i + bin_idx) % distribution_bins < (repeats % distribution_bins):
                base_reps += 1
            bin_weights.append(base_reps)
        
        # Generate sequence for this bin using similar logic to original
        bin_sequence = _generate_bin_sequence(population, bin_weights, pair_indices, all_sequences)
        all_sequences.extend(bin_sequence)
    
    return all_sequences

def _getsequences_original(nbvalues, repeats, pair_indices):
    """Original algorithm for fallback"""
    population = range_list(nbvalues)
    weights = [repeats] * nbvalues
    out = []
    prev = None
    prev_pair_group = None
    
    for _ in range(nbvalues * repeats):
        temp_weights = weights.copy()
        if prev is not None:
            temp_weights[prev] = 0
            for i, pair_group in enumerate(pair_indices):
                if pair_group == prev_pair_group:
                    temp_weights[i] = 0
        
        try:
            chosen = weightedchoice(population, temp_weights)
        except ValueError as e:
            if "No weights > 0" in str(e):
                sendback(prev, weights[prev], out)
                break
            else:
                raise
        
        out.append(chosen)
        weights[chosen] -= 1
        prev = chosen
        prev_pair_group = pair_indices[chosen]
    
    return out

def _generate_bin_sequence(population, bin_weights, pair_indices, prev_sequences):
    """Generate sequence for a single bin with constraints"""
    weights = bin_weights.copy()
    bin_sequence = []
    
    # Determine previous context from the last item in previous sequences
    prev = prev_sequences[-1] if prev_sequences else None
    prev_pair_group = pair_indices[prev] if prev is not None else None
    
    target_length = sum(bin_weights)
    consecutive_failures = 0
    max_failures = 3  # Allow some flexibility when constraints are too restrictive
    
    for _ in range(target_length):
        temp_weights = weights.copy()
        constraint_applied = False
        
        if prev is not None:
            # Don't repeat the exact same index consecutively
            temp_weights[prev] = 0
            constraint_applied = True
            
            # Don't use indices from the same pair group consecutively (if possible)
            pair_constrained_weights = temp_weights.copy()
            for i, pair_group in enumerate(pair_indices):
                if pair_group == prev_pair_group:
                    pair_constrained_weights[i] = 0
            
            # Use pair-constrained weights if we have valid choices
            if sum(pair_constrained_weights) > 0:
                temp_weights = pair_constrained_weights
            elif consecutive_failures < max_failures:
                # If pair constraint is too restrictive, track failures but keep trying
                consecutive_failures += 1
        
        # If no valid choices at all, we need to break the constraints
        if sum(temp_weights) == 0:
            temp_weights = weights.copy()
            # As a last resort, allow everything (this should be very rare)
            if sum(temp_weights) == 0:
                break
        
        try:
            chosen = weightedchoice(population, temp_weights)
            # Reset failure counter on successful choice
            if prev is None or pair_indices[chosen] != prev_pair_group:
                consecutive_failures = 0
        except ValueError:
            break
        
        bin_sequence.append(chosen)
        weights[chosen] -= 1
        prev = chosen
        prev_pair_group = pair_indices[chosen]
    
    return bin_sequence
