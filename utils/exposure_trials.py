# exposure_trials.py
"""
Exposure trial generation for the SL-Obj-Inference task.
This module handles the creation of exposure streams with temporal constraints.
"""

import os
import pandas as pd
import numpy as np
import random


def create_image_trials(abcd_groups, pair_idx_trials, stream_type="AB_CD", label="stream"):
    """
    Convert pair index trials to actual image trials.
    
    Args:
        abcd_groups: Dictionary with A, B, C, D stimulus assignments
        pair_idx_trials: List of pair indices for each trial
        stream_type: Type of stream - "AB_CD" for stream1, "BC" for stream2
        label: Label for debugging
        
    Returns:
        List of trial dictionaries with images
    """
    
    def get_pair_info(pair_idx, stream_type):
        """Get stimulus pair, labels, and indices for a given pair index."""
        if stream_type == "AB_CD":
            if pair_idx < len(abcd_groups["A"]):
                grp_pair_idx = pair_idx
                stream_pair_idx = pair_idx
                a_key = f"A{grp_pair_idx + 1}"
                b_key = f"B{grp_pair_idx + 1}"
                first_stim = abcd_groups["A"][a_key]
                second_stim = abcd_groups["B"][b_key]
                pair_label = "AB"
                first_stim_label = "A"
                second_stim_label = "B"
            else:
                grp_pair_idx = pair_idx - len(abcd_groups["A"])
                stream_pair_idx = pair_idx
                c_key = f"C{grp_pair_idx + 1}"
                d_key = f"D{grp_pair_idx + 1}"
                first_stim = abcd_groups["C"][c_key]
                second_stim = abcd_groups["D"][d_key]
                pair_label = "CD"
                first_stim_label = "C"
                second_stim_label = "D"
        
        elif stream_type == "BC":
            grp_pair_idx = pair_idx
            stream_pair_idx = pair_idx
            b_key = f"B{grp_pair_idx + 1}"
            c_key = f"C{grp_pair_idx + 1}"
            first_stim = abcd_groups["B"][b_key]
            second_stim = abcd_groups["C"][c_key]
            pair_label = "BC"
            first_stim_label = "B"
            second_stim_label = "C"
        
        else:
            raise ValueError(f"Unknown stream_type: {stream_type}")
        
        return {
            'first_stim': first_stim,
            'second_stim': second_stim,
            'grp_pair_idx': grp_pair_idx,
            'stream_pair_idx': stream_pair_idx,
            'pair_label': pair_label,
            'first_stim_label': first_stim_label,
            'second_stim_label': second_stim_label
        }
    
    trials = []
    trial_idx = 0
    
    for pair_trial_idx, pair_idx in enumerate(pair_idx_trials):
        pair_info = get_pair_info(pair_idx, stream_type)
        
        trials.extend([
            {
                'trial_num': trial_idx,
                'pair_trial_idx': pair_trial_idx,
                'stream_pair_idx': pair_info['stream_pair_idx'],
                'grp_pair_idx': pair_info['grp_pair_idx'],
                'stim_label_grp_num': f"{pair_info['first_stim_label']}{pair_info['grp_pair_idx'] + 1}",
                'pair_label': pair_info['pair_label'],
                'stim_label': pair_info['first_stim_label'],
                'image': pair_info['first_stim'],
                'position_in_pair': 0,
            },
            {
                'trial_num': trial_idx + 1,
                'pair_trial_idx': pair_trial_idx,
                'stream_pair_idx': pair_info['stream_pair_idx'],
                'grp_pair_idx': pair_info['grp_pair_idx'],
                'stim_label_grp_num': f"{pair_info['second_stim_label']}{pair_info['grp_pair_idx'] + 1}",
                'pair_label': pair_info['pair_label'],
                'stim_label': pair_info['second_stim_label'],
                'image': pair_info['second_stim'],
                'position_in_pair': 1,
            }
        ])
        trial_idx += 2
    
    total_trials = len(trials)
    print(f"   Results: {total_trials} individual exposure trials")
    
    return trials


import os
import pandas as pd


def generate_obj_stream(exp_info, abcd_groups, rank_dict, obj_dict):
    """
    Generate the object stream for the test phase.

    Returns:
        tuple:
            - obj_stream_data (pd.DataFrame): full exposure phase dataset
            - stream1_data (dict): trials and indices for stream 1 (AB/CD)
            - stream2_data (dict): trials and indices for stream 2 (BC)
    """

    # --- Setup parameters ---
    num_stim = exp_info["num_stim|hid"]
    num_grps = exp_info["num_grps|hid"]
    reps = exp_info["num_reps|hid"]
    n_bins = 5
    max_attempts = 2000
    

    # Stream sizes
    n_pairs_stream1 = num_grps * 2
    n_pairs_stream2 = num_grps
    n_pair_trls_stream1 = n_pairs_stream1 * reps
    n_pair_trls_stream2 = n_pairs_stream2 * reps
    
    break_interval = n_pair_trls_stream2

    # Stream indices
    pair_indices_stream1 = [i % num_grps for i in range(n_pairs_stream1)]
    pair_indices_stream2 = [i % num_grps for i in range(n_pairs_stream2)]

    # Bin distributions
    bins_stream1 = getbins(n_pair_trls_stream1, n_bins)
    bins_stream2 = getbins(n_pair_trls_stream2, n_bins)

    print(f"stream 1: trials = {n_pair_trls_stream1}\t bins={bins_stream1}\n"
          f"stream 2: trials = {n_pair_trls_stream2}\t bins={bins_stream2}\n")

    # --- Generate randomized pair index sequences ---
    for attempt in range(1, max_attempts + 1):
        try:
            stream1_pair_idx_trials = getsequences(
                nbvalues=n_pairs_stream1,
                repeats=reps,
                pair_indices=pair_indices_stream1,
                distribution_bins=n_bins
            )
            stream2_pair_idx_trials = getsequences(
                nbvalues=n_pairs_stream2,
                repeats=reps,
                pair_indices=pair_indices_stream2,
                distribution_bins=n_bins
            )
            break
        except Exception as e:
            print(f"⚠️ Restarting attempt {attempt}: {e}")
    else:
        raise RuntimeError(f"Failed to assign 1-backs after {max_attempts} attempts")

    # --- Create image trials for both streams ---
    stream1_trials = create_image_trials(abcd_groups, stream1_pair_idx_trials, stream_type="AB_CD", label="stream1")
    stream2_trials = create_image_trials(abcd_groups, stream2_pair_idx_trials, stream_type="BC", label="stream2")

    stream1_data = {'trials': stream1_trials, 'pair_idx': stream1_pair_idx_trials}
    stream2_data = {'trials': stream2_trials, 'pair_idx': stream2_pair_idx_trials}

    # --- Initialize dataset structure ---
    fields = [
        'task', 'task_tNum', 'block_num', 'block_tNum', 'stream', 'stream_num',
        'stream_tNum', 'pair_label', 'pair_trial_idx', 'stream_pair_idx', 'stim_label',
        'grp_num', 'stim_label_grp_num', 'image', 'object', 'position_in_pair',
        'rank_num', 'size', 'correct_resp'
    ]
    obj_stream_data = {k: [] for k in fields}

    # --- Helper functions ---
    def append_trial(trial, context):
        """Append a single exposure trial."""
        current_rank = rank_dict[trial['image']]
        current_obj = obj_dict[trial['image']]
        prev_rank = context['previous_rank']

        if context['block_tNum'] == 0:
            size, correct = 'none', 'none'
        else:
            if current_rank < prev_rank:  # 1 = largest, 24 = smallest
                size, correct = 'bigger', 'f'
            elif current_rank > prev_rank:
                size, correct = 'smaller', 'j'
            else:
                size, correct = 'N/A', 'N/A'

        obj_stream_data['task'].append('exposure')
        obj_stream_data['task_tNum'].append(context['trial_counter'])
        obj_stream_data['block_num'].append(context['block_num'])
        obj_stream_data['block_tNum'].append(context['block_tNum'])
        obj_stream_data['stream'].append(context['stream_label'])
        obj_stream_data['stream_num'].append(context['stream_num'])
        obj_stream_data['stream_tNum'].append(trial['trial_num'])
        obj_stream_data['pair_trial_idx'].append(trial['pair_trial_idx'])
        obj_stream_data['stream_pair_idx'].append(trial['stream_pair_idx'])
        obj_stream_data['grp_num'].append(trial['grp_pair_idx'])
        obj_stream_data['pair_label'].append(trial['pair_label'])
        obj_stream_data['stim_label'].append(trial['stim_label'])
        obj_stream_data['image'].append(trial['image'])
        obj_stream_data['object'].append(current_obj)
        obj_stream_data['position_in_pair'].append(trial['position_in_pair'])
        obj_stream_data['stim_label_grp_num'].append(trial['stim_label_grp_num'])
        obj_stream_data['rank_num'].append(current_rank)
        obj_stream_data['size'].append(size)
        obj_stream_data['correct_resp'].append(correct)

        context['previous_rank'] = current_rank
        context['block_tNum'] += 1
        context['trial_counter'] += 1

    def append_break(context):
        """Append a break trial."""
        for key in obj_stream_data:
            obj_stream_data[key].append(None)
        obj_stream_data['task'][-1] = 'break'  # overwrite last added task field
        context['block_num'] += 1
        context['block_tNum'] = 0
        context['previous_rank'] = None

    # --- Stream generation loop ---
    def process_stream(trials, stream_label, stream_num, context):
        context.update({
            'stream_label': stream_label,
            'stream_num': stream_num,
            'previous_rank': None,
            'block_tNum': 0,
        })
        for trial in trials:
            append_trial(trial, context)
            if context['trial_counter'] % break_interval == 0 and \
               context['trial_counter'] < (len(stream1_trials) + len(stream2_trials)):
                append_break(context)

    # Context for tracking trial/block counts
    context = {'trial_counter': 0, 'block_num': 0}

    # Process both streams sequentially
    process_stream(stream1_trials, 'AB_CD', stream_num=0, context=context)
    process_stream(stream2_trials, 'BC', stream_num=1, context=context)

    # --- Convert to DataFrame and save ---
    obj_stream_df = pd.DataFrame(obj_stream_data)
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"{exp_info['file_prefix']}_exposure-trials.csv")
    obj_stream_df.to_csv(outfile, index=False)

    return obj_stream_df, stream1_data, stream2_data


def getbins(end, num):
    start = 1
    step = (end - start) / (num - 1)
    result = []
    for i in range(num):
        val = start + i * step
        result.append(int(val))
    result[-1] -= 2
    return result


def range_list(n, start=0, step=1):
    return list(range(start, n, step))


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


def getsequences(nbvalues, repeats, pair_indices, distribution_bins=5):
    """
    Generate sequences with constraints and even distribution.
    """
    population = range_list(nbvalues)
    total_length = nbvalues * repeats
    
    if distribution_bins <= 1:
        raise ValueError("distribution_bins must be greater than 1")
    
    bin_size = total_length // distribution_bins
    all_sequences = []
    
    for bin_idx in range(distribution_bins):
        bin_start = bin_idx * bin_size
        bin_end = (bin_idx + 1) * bin_size if bin_idx < distribution_bins - 1 else total_length
        bin_length = bin_end - bin_start
        
        bin_weights = []
        for i in range(nbvalues):
            base_reps = repeats // distribution_bins
            if (i + bin_idx) % distribution_bins < (repeats % distribution_bins):
                base_reps += 1
            bin_weights.append(base_reps)
        
        bin_sequence = _generate_bin_sequence(population, bin_weights, pair_indices, all_sequences)
        all_sequences.extend(bin_sequence)
    
    return all_sequences


def _generate_bin_sequence(population, bin_weights, pair_indices, prev_sequences):
    """Generate sequence for a single bin with constraints"""
    weights = bin_weights.copy()
    bin_sequence = []
    
    prev = prev_sequences[-1] if prev_sequences else None
    prev_pair_group = pair_indices[prev] if prev is not None else None
    
    target_length = sum(bin_weights)
    consecutive_failures = 0
    max_failures = 3
    
    for _ in range(target_length):
        temp_weights = weights.copy()
        constraint_applied = False
        
        if prev is not None:
            temp_weights[prev] = 0
            constraint_applied = True
            
            pair_constrained_weights = temp_weights.copy()
            for i, pair_group in enumerate(pair_indices):
                if pair_group == prev_pair_group:
                    pair_constrained_weights[i] = 0
            
            if sum(pair_constrained_weights) > 0:
                temp_weights = pair_constrained_weights
            elif consecutive_failures < max_failures:
                consecutive_failures += 1
        
        if sum(temp_weights) == 0:
            temp_weights = weights.copy()
            if sum(temp_weights) == 0:
                break
        
        try:
            chosen = weightedchoice(population, temp_weights)
            if prev is None or pair_indices[chosen] != prev_pair_group:
                consecutive_failures = 0
        except ValueError:
            break
        
        bin_sequence.append(chosen)
        weights[chosen] -= 1
        prev = chosen
        prev_pair_group = pair_indices[chosen]
    
    return bin_sequence
