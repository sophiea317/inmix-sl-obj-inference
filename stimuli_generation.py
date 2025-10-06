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
    
    
def create_image_trials(abcd_groups, oneback_result, pair_idx_trials, stream_type="AB_CD", label="stream"):
    """
    Convert pair index trials and 1-back assignments to actual image trials.
    
    Args:
        abcd_groups: Dictionary with A, B, C, D stimulus assignments
        oneback_result: OneBackResult object with 1-back assignments
        pair_idx_trials: List of pair indices for each trial
        stream_type: Type of stream - "AB_CD" for stream1, "BC" for stream2
        label: Label for debugging
        
    Returns:
        List of trial dictionaries with images and 1-back indicators
    """
    
    if not oneback_result.success:
        raise ValueError(f"Cannot create trials from failed 1-back assignment: {oneback_result.error_message}")
    
    print(f"\n🎬 Creating image trials for {label} ({stream_type})")
    print(f"   Input: {len(pair_idx_trials)} pair trials, {len(oneback_result.oneback_array)} 1-back positions")
    
    def get_pair_info(pair_idx, stream_type):
        """Get stimulus pair, labels, and indices for a given pair index."""
        if stream_type == "AB_CD":
            # Stream 1: AB pairs (indices 0,1,2,3,4,5) and CD pairs (indices 6,7,8,9,10,11)
            if pair_idx < len(abcd_groups["A"]):
                # AB pair - grp_pair_idx is same as pair_idx
                grp_pair_idx = pair_idx
                stream_pair_idx = pair_idx  # 0-5
                a_key = f"A{grp_pair_idx + 1}"
                b_key = f"B{grp_pair_idx + 1}"
                first_stim = abcd_groups["A"][a_key]
                second_stim = abcd_groups["B"][b_key]
                pair_label = "AB"
                first_stim_label = "A"
                second_stim_label = "B"
            else:
                # CD pair
                grp_pair_idx = pair_idx - len(abcd_groups["A"])  # 0-5
                stream_pair_idx = pair_idx  # 6-11
                c_key = f"C{grp_pair_idx + 1}"
                d_key = f"D{grp_pair_idx + 1}"
                first_stim = abcd_groups["C"][c_key]
                second_stim = abcd_groups["D"][d_key]
                pair_label = "CD"
                first_stim_label = "C"
                second_stim_label = "D"
        
        elif stream_type == "BC":
            # Stream 2: BC pairs only - grp_pair_idx and stream_pair_idx are the same
            grp_pair_idx = pair_idx
            stream_pair_idx = pair_idx  # 0-5
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
    
    # Create flattened trial list
    trials = []
    trial_idx = 0
    
    for pair_trial_idx, pair_idx in enumerate(pair_idx_trials):
        # Get the stimulus pair info for this trial
        pair_info = get_pair_info(pair_idx, stream_type)
        
        # Check if this pair trial has a 1-back assignment
        oneback_marker = oneback_result.oneback_array[pair_trial_idx]
        
        if oneback_marker == 0:
            # Normal trial: [first, second]
            trials.extend([
                {
                    'trial_num': trial_idx,
                    'pair_trial_idx': pair_trial_idx,
                    'stream_pair_idx': pair_info['stream_pair_idx'],
                    'grp_pair_idx': pair_info['grp_pair_idx'],
                    'pair_label': pair_info['pair_label'],
                    'stim_label': pair_info['first_stim_label'],
                    'image': pair_info['first_stim'],
                    'position_in_pair': 0,  # first stimulus
                    'is_1back': 0,
                    'oneback_type': None
                },
                {
                    'trial_num': trial_idx + 1,
                    'pair_trial_idx': pair_trial_idx,
                    'stream_pair_idx': pair_info['stream_pair_idx'],
                    'grp_pair_idx': pair_info['grp_pair_idx'],
                    'pair_label': pair_info['pair_label'],
                    'stim_label': pair_info['second_stim_label'],
                    'image': pair_info['second_stim'],
                    'position_in_pair': 1,  # second stimulus
                    'is_1back': 0,
                    'oneback_type': None
                }
            ])
            trial_idx += 2
            
        else:
            # 1-back trial: add extra repetition
            # oneback_marker indicates which element to repeat: [1,0] = first, [0,1] = second
            if oneback_marker == [1, 0]:
                # Repeat first stimulus: [first, first, second]
                trials.extend([
                    {
                        'trial_num': trial_idx,
                        'pair_trial_idx': pair_trial_idx,
                        'stream_pair_idx': pair_info['stream_pair_idx'],
                        'grp_pair_idx': pair_info['grp_pair_idx'],
                        'pair_label': pair_info['pair_label'],
                        'stim_label': pair_info['first_stim_label'],
                        'image': pair_info['first_stim'],
                        'position_in_pair': 0,
                        'is_1back': 0,
                        'oneback_type': 'first_repeat'
                    },
                    {
                        'trial_num': trial_idx + 1,
                        'pair_trial_idx': pair_trial_idx,
                        'stream_pair_idx': pair_info['stream_pair_idx'],
                        'grp_pair_idx': pair_info['grp_pair_idx'],
                        'pair_label': pair_info['pair_label'],
                        'stim_label': pair_info['first_stim_label'],
                        'image': pair_info['first_stim'],  # repeated
                        'position_in_pair': 0,
                        'is_1back': 1,  # this is the 1-back
                        'oneback_type': 'first_repeat'
                    },
                    {
                        'trial_num': trial_idx + 2,
                        'pair_trial_idx': pair_trial_idx,
                        'stream_pair_idx': pair_info['stream_pair_idx'],
                        'grp_pair_idx': pair_info['grp_pair_idx'],
                        'pair_label': pair_info['pair_label'],
                        'stim_label': pair_info['second_stim_label'],
                        'image': pair_info['second_stim'],
                        'position_in_pair': 1,
                        'is_1back': 0,
                        'oneback_type': 'first_repeat'
                    }
                ])
                trial_idx += 3
                
            elif oneback_marker == [0, 1]:
                # Repeat second stimulus: [first, second, second]
                trials.extend([
                    {
                        'trial_num': trial_idx,
                        'pair_trial_idx': pair_trial_idx,
                        'stream_pair_idx': pair_info['stream_pair_idx'],
                        'grp_pair_idx': pair_info['grp_pair_idx'],
                        'pair_label': pair_info['pair_label'],
                        'stim_label': pair_info['first_stim_label'],
                        'image': pair_info['first_stim'],
                        'position_in_pair': 0,
                        'is_1back': 0,
                        'oneback_type': 'second_repeat'
                    },
                    {
                        'trial_num': trial_idx + 1,
                        'pair_trial_idx': pair_trial_idx,
                        'stream_pair_idx': pair_info['stream_pair_idx'],
                        'grp_pair_idx': pair_info['grp_pair_idx'],
                        'pair_label': pair_info['pair_label'],
                        'stim_label': pair_info['second_stim_label'],
                        'image': pair_info['second_stim'],
                        'position_in_pair': 1,
                        'is_1back': 0,
                        'oneback_type': 'second_repeat'
                    },
                    {
                        'trial_num': trial_idx + 2,
                        'pair_trial_idx': pair_trial_idx,
                        'stream_pair_idx': pair_info['stream_pair_idx'],
                        'grp_pair_idx': pair_info['grp_pair_idx'],
                        'pair_label': pair_info['pair_label'],
                        'stim_label': pair_info['second_stim_label'],
                        'image': pair_info['second_stim'],  # repeated
                        'position_in_pair': 1,
                        'is_1back': 1,  # this is the 1-back
                        'oneback_type': 'second_repeat'
                    }
                ])
                trial_idx += 3
                
            else:
                # Unknown marker format - treat as normal trial
                print(f"⚠️ Unknown 1-back marker format: {oneback_marker}, treating as normal trial")
                trials.extend([
                    {
                        'trial_num': trial_idx,
                        'pair_trial_idx': pair_trial_idx,
                        'stream_pair_idx': pair_info['stream_pair_idx'],
                        'grp_pair_idx': pair_info['grp_pair_idx'],
                        'pair_label': pair_info['pair_label'],
                        'stim_label': pair_info['first_stim_label'],
                        'image': pair_info['first_stim'],
                        'position_in_pair': 0,
                        'is_1back': 0,
                        'oneback_type': 'unknown_marker'
                    },
                    {
                        'trial_num': trial_idx + 1,
                        'pair_trial_idx': pair_trial_idx,
                        'stream_pair_idx': pair_info['stream_pair_idx'],
                        'grp_pair_idx': pair_info['grp_pair_idx'],
                        'pair_label': pair_info['pair_label'],
                        'stim_label': pair_info['second_stim_label'],
                        'image': pair_info['second_stim'],
                        'position_in_pair': 1,
                        'is_1back': 0,
                        'oneback_type': 'unknown_marker'  
                    }
                ])
                trial_idx += 2
    
    # Calculate summary statistics
    total_trials = len(trials)
    oneback_trials = sum(1 for trial in trials if trial['is_1back'] == 1)
    normal_pair_trials = len([idx for idx, marker in enumerate(oneback_result.oneback_array) if marker == 0])
    oneback_pair_trials = len([idx for idx, marker in enumerate(oneback_result.oneback_array) if marker != 0])
    
    print(f"   Results: {total_trials} individual trials")
    print(f"   - {normal_pair_trials} normal pairs → {normal_pair_trials * 2} trials")
    print(f"   - {oneback_pair_trials} 1-back pairs → {oneback_pair_trials * 3} trials")
    print(f"   - {oneback_trials} trials marked as 1-back repeats")
    
    return trials

def generate_obj_stream(exp_info, abcd_groups):
    """
    Generate the object stream for the test phase.
    Returns:
        dict: comprehensive dataset with trials, indices, and quality metrics
    """
    
    # --- Parameters ---
    num_stim = exp_info["num_stim|hid"]
    num_grps = exp_info["num_grps|hid"]
    prob_1back = exp_info["prob_1back|hid"]
    reps = exp_info["num_reps|hid"]
    stim_1back = int(reps * prob_1back * 2)         # number of 1-back trials for a stimulus
    print(f"Number of 1-back trials per stimulus: {stim_1back}")
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
            stream1_pair_idx_trials = getsequences(nbvalues=n_pairs_stream1, repeats=reps, pair_indices=pair_indices_stream1, distribution_bins=5)
            stream2_pair_idx_trials = getsequences(nbvalues=n_pairs_stream2, repeats=reps, pair_indices=pair_indices_stream2, distribution_bins=5)

            array_1back_stream1 = zeroslike(stream1_pair_idx_trials)
            array_1back_stream2 = zeroslike(stream2_pair_idx_trials)

            group_1back_stream1, group_1back_stream2 = [], []
            exclude_1back_stream1, exclude_1back_stream2 = [], []
            
            config_stream1 = OneBackConfig(
                num_pairs=n_pairs_stream1,
                total_onebacks_per_pair=stim_1back,   # should be = 8
                temporal_bins=bins_stream1,
                first_element_marker=pair_idx1,
                second_element_marker=pair_idx2,
                label="stream_1"
            )
            config_stream2 = OneBackConfig(
                num_pairs=n_pairs_stream2,
                total_onebacks_per_pair=stim_1back,   # should be = 8
                temporal_bins=bins_stream2,
                first_element_marker=pair_idx1,
                second_element_marker=pair_idx2,
                label="stream_2"
            )

            stream1_1back_df = assign_1backs(config_stream1, stream1_pair_idx_trials, exclude_1back_stream1.copy(), array_1back_stream1.copy())
            stream2_1back_df = assign_1backs(config_stream2, stream2_pair_idx_trials, exclude_1back_stream2.copy(), array_1back_stream2.copy())

            print(f"\n\nSTREAM 1")
            validate_1back_assignment(stream1_1back_df, config_stream1)
            print(f"\n\nSTREAM 2")
            validate_1back_assignment(stream2_1back_df, config_stream2)

            success = True
        except Exception as e:
            print(f"⚠️ Restarting attempt {attempts}: {e}")
    if not success:
        raise RuntimeError(f"Failed to assign 1-backs after {maxAttempts} attempts")

    # Convert to actual image trials with 1-back indicators
    stream1_trials = create_image_trials(
        abcd_groups, stream1_1back_df, stream1_pair_idx_trials, 
        stream_type="AB_CD", label="stream1"
    )
    
    stream2_trials = create_image_trials(
        abcd_groups, stream2_1back_df, stream2_pair_idx_trials,
        stream_type="BC", label="stream2"
    )

    stream1_data = {
            'trials': stream1_trials,
            'pair_indices': stream1_pair_idx_trials,
            'oneback_assignments': stream1_1back_df,
            'quality_metrics': stream1_1back_df.quality_metrics if stream1_1back_df.success else {}
    }
    stream2_data = {
            'trials': stream2_trials,
            'pair_indices': stream2_pair_idx_trials,
            'oneback_assignments': stream2_1back_df,
            'quality_metrics': stream2_1back_df.quality_metrics if stream2_1back_df.success else {}
    }

    # concatenate stream1 and stream2 trials into one obj_stream_data dataframe with columns: stream_num, trial_num, pair_trial_idx, stream_pair_idx, grp_pair_idx, pair_label, stim_label, image, position_in_pair, is_1back, oneback_type
    # stream_pair_idx is the index of the pair in the stream (0-11 for stream1, 0-5 for stream2)
    # grp_pair_idx is the index of the group pair (0-5 for both streams)
    # pair_label is the label of the pair (A-B, B-C, C-D, etc.)
    # stim_label is the label of the stimulus (A, B, C, D)
    obj_stream_data = {
        'stream_num': [],
        'trial_num': [],
        'pair_trial_idx': [],
        'stream_pair_idx': [],
        'grp_pair_idx': [],
        'pair_label': [],
        'stim_label': [],
        'image': [],
        'position_in_pair': [],
        'is_1back': [],
        'oneback_type': []
    }
    
    for trial in stream1_trials:
        obj_stream_data['stream_num'].append(1)
        obj_stream_data['trial_num'].append(trial['trial_num'])
        obj_stream_data['pair_trial_idx'].append(trial['pair_trial_idx'])
        obj_stream_data['stream_pair_idx'].append(trial['stream_pair_idx'])
        obj_stream_data['grp_pair_idx'].append(trial['grp_pair_idx'])
        obj_stream_data['pair_label'].append(trial['pair_label'])
        obj_stream_data['stim_label'].append(trial['stim_label'])
        obj_stream_data['image'].append(trial['image'])
        obj_stream_data['position_in_pair'].append(trial['position_in_pair'])
        obj_stream_data['is_1back'].append(trial['is_1back'])
        obj_stream_data['oneback_type'].append(trial['oneback_type'])

    for trial in stream2_trials:
        obj_stream_data['stream_num'].append(2)
        obj_stream_data['trial_num'].append(trial['trial_num'])
        obj_stream_data['pair_trial_idx'].append(trial['pair_trial_idx'])
        obj_stream_data['stream_pair_idx'].append(trial['stream_pair_idx'])
        obj_stream_data['grp_pair_idx'].append(trial['grp_pair_idx'])
        obj_stream_data['pair_label'].append(trial['pair_label'])
        obj_stream_data['stim_label'].append(trial['stim_label'])
        obj_stream_data['image'].append(trial['image'])
        obj_stream_data['position_in_pair'].append(trial['position_in_pair'])
        obj_stream_data['is_1back'].append(trial['is_1back'])
        obj_stream_data['oneback_type'].append(trial['oneback_type'])

    # convert to DataFrame
    obj_stream_data = pd.DataFrame(obj_stream_data)
    
    # write to CSV for QA
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    obj_stream_data.to_csv(os.path.join(out_dir, f"{exp_info['file_prefix']}_exposure-trials.csv"), index=False)
    
    return obj_stream_data, stream1_data, stream2_data
# ============================================================
# Core 1-Back Assignment Function
# ============================================================

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from collections import defaultdict

@dataclass
class OneBackConfig:
    """Configuration for 1-back assignment."""
    num_pairs: int
    total_onebacks_per_pair: int
    temporal_bins: List[int]
    first_element_marker: Any = field(default_factory=lambda: [1, 0])
    second_element_marker: Any = field(default_factory=lambda: [0, 1])
    exclusion_radius: int = 1
    min_spacing: int = 3
    label: str = "stream"
    
    def __post_init__(self):
        if self.total_onebacks_per_pair <= 0:
            raise ValueError(f"total_onebacks_per_pair must be > 0, got {self.total_onebacks_per_pair}")
        if len(self.temporal_bins) < 2:
            raise ValueError(f"Need at least 2 temporal bins, got {len(self.temporal_bins)}")
        if self.num_pairs <= 0:
            raise ValueError(f"num_pairs must be > 0, got {self.num_pairs}")

@dataclass
class OneBackResult:
    """Result of 1-back assignment."""
    oneback_array: List[Any]
    assignment_order: List[int]
    excluded_positions: List[int]
    quality_metrics: Dict[str, Any]
    success: bool = True
    error_message: str = ""

def find_pair_positions(sequence_indices: List[int], pair_id: int) -> List[int]:
    """Find all positions where a specific pair appears in the sequence."""
    return [i for i, val in enumerate(sequence_indices) if val == pair_id]

def distribute_across_bins(total_count: int, num_bins: int, method: str = "multinomial") -> List[int]:
    """Distribute a total count across bins using specified method."""
    if method == "multinomial":
        try:
            probabilities = [1.0 / num_bins] * num_bins
            return list(np.random.multinomial(total_count, probabilities))
        except ValueError:
            method = "even"  # Fallback to even distribution
    
    if method == "even":
        base_count = total_count // num_bins
        remainder = total_count % num_bins
        counts = [base_count] * num_bins
        for i in range(remainder):
            counts[i] += 1
        return counts
    
    raise ValueError(f"Unknown distribution method: {method}")

def find_valid_positions_in_bin(pair_positions: List[int], bin_start: int, bin_end: int, 
                               excluded_positions: set) -> List[int]:
    """Find valid positions for 1-back assignment within a temporal bin."""
    valid_positions = []
    for pos in pair_positions:
        if (bin_start <= pos < bin_end and 
            pos not in excluded_positions):
            valid_positions.append(pos)
    return valid_positions

def update_exclusion_zones(selected_positions: List[int], exclusion_radius: int, 
                          sequence_length: int) -> List[int]:
    """Update exclusion zones around selected positions."""
    new_exclusions = []
    for pos in selected_positions:
        for offset in range(-exclusion_radius, exclusion_radius + 1):
            neighbor_pos = pos + offset
            if 0 <= neighbor_pos < sequence_length:
                new_exclusions.append(neighbor_pos)
    return new_exclusions

def assign_element_markers(selected_positions: List[int], config: OneBackConfig) -> Dict[int, Any]:
    """Assign first/second element markers to selected positions."""
    shuffled_positions = selected_positions.copy()
    random.shuffle(shuffled_positions)
    
    markers = {}
    half_point = config.total_onebacks_per_pair // 2
    
    for i, pos in enumerate(shuffled_positions):
        if i < half_point:
            markers[pos] = config.first_element_marker
        else:
            markers[pos] = config.second_element_marker
    
    return markers, shuffled_positions

def validate_assignment_quality(config: OneBackConfig, oneback_array: List[Any], 
                              sequence_indices: List[int]) -> Dict[str, Any]:
    """Validate the quality of 1-back assignments."""
    metrics = {
        'total_onebacks': sum(1 for x in oneback_array if x != 0),
        'pair_distribution': defaultdict(int),
        'temporal_distribution': {},
        'spacing_violations': [],
        'consecutive_violations': 0,
        'success_rate': 0.0
    }
    
    # Count 1-backs per pair
    for i, marker in enumerate(oneback_array):
        if marker != 0:
            pair_id = sequence_indices[i]
            metrics['pair_distribution'][pair_id] += 1
    
    # Check temporal distribution
    num_chunks = len(config.temporal_bins) - 1
    chunk_size = len(oneback_array) // num_chunks if num_chunks > 0 else len(oneback_array)
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(oneback_array))
        chunk_onebacks = sum(1 for i in range(start_idx, end_idx) if oneback_array[i] != 0)
        metrics['temporal_distribution'][f'chunk_{chunk_idx}'] = chunk_onebacks
    
    # Check for consecutive violations
    for i in range(1, len(oneback_array)):
        if oneback_array[i] != 0 and oneback_array[i-1] != 0:
            metrics['consecutive_violations'] += 1
    
    # Check spacing violations
    oneback_positions = [i for i, x in enumerate(oneback_array) if x != 0]
    for i in range(1, len(oneback_positions)):
        spacing = oneback_positions[i] - oneback_positions[i-1]
        if spacing < config.min_spacing:
            metrics['spacing_violations'].append((oneback_positions[i-1], oneback_positions[i]))
    
    # Calculate success rate
    expected_total = config.num_pairs * config.total_onebacks_per_pair
    metrics['success_rate'] = metrics['total_onebacks'] / expected_total if expected_total > 0 else 0.0
    
    return metrics


def validate_1back_assignment(results, config):
    if results.success:
        metrics = results.quality_metrics

        print("📊 Quality Metrics Explained:")
        print(f"   Success Rate: {metrics['success_rate']:.1%}")
        print("     → Percentage of requested 1-backs successfully assigned")
        
        print(f"\n   Total 1-backs: {metrics['total_onebacks']}")
        print("     → Actual number of 1-back trials created")
        
        print(f"\n   Consecutive Violations: {metrics['consecutive_violations']}")
        print("     → Number of 1-backs that appear back-to-back (bad!)")
        
        print(f"\n   Spacing Violations: {len(metrics['spacing_violations'])}")
        print(f"     → Number of 1-backs closer than {config.min_spacing} positions")
        
        print(f"\n   Pair Distribution:")
        for pair_id, count in metrics['pair_distribution'].items():
            balance = "✅" if count == config.total_onebacks_per_pair else "⚠️"
            print(f"     {balance} Pair {pair_id}: {count}/{config.total_onebacks_per_pair}")
        
        print(f"\n   Temporal Distribution:")
        for chunk, count in metrics['temporal_distribution'].items():
            print(f"     {chunk}: {count} 1-backs")
        
        # Overall assessment
        if (metrics['success_rate'] >= 0.95 and 
            metrics['consecutive_violations'] == 0 and 
            len(metrics['spacing_violations']) <= 2):
            print(f"\n🏆 EXCELLENT QUALITY - Ready for experiment!")
        elif metrics['success_rate'] >= 0.8:
            print(f"\n⚠️ ACCEPTABLE QUALITY - Minor issues, but usable")
        else:
            print(f"\n❌ POOR QUALITY - Consider adjusting parameters")
    else:
        print(f"❌ Assignment failed: {results.error_message}")

def assign_1backs(config: OneBackConfig, sequence_indices: List[int], 
                          excluded_positions: List[int], 
                          oneback_array: List[Any] = None) -> OneBackResult:
    """
    Improved 1-back assignment with configuration objects and quality validation.
    
    Args:
        config: Configuration object with all assignment parameters
        sequence_indices: Sequence indicating which pair appears at each position
        excluded_positions: Positions that cannot be used for 1-backs
        oneback_array: Array to mark 1-back positions (created if None)
        
    Returns:
        OneBackResult with assignment results and quality metrics
    """
    
    if oneback_array is None:
        oneback_array = [0] * len(sequence_indices)
    
    excluded_set = set(excluded_positions)
    assignment_order = []
    all_selected_positions = []
    
    print(f"🎯 Starting improved 1-back assignment for {config.label}")
    print(f"   {config.total_onebacks_per_pair} per pair across {len(config.temporal_bins)-1} bins")
    
    try:
        for pair_id in range(config.num_pairs):
            # Find positions for this pair
            pair_positions = find_pair_positions(sequence_indices, pair_id)
            
            if not pair_positions:
                print(f"⚠️ No positions found for pair {pair_id}")
                continue
            
            # Distribute across bins
            num_bins = len(config.temporal_bins) - 1
            bin_counts = distribute_across_bins(config.total_onebacks_per_pair, num_bins)
            
            pair_selected_positions = []
            
            # Process each bin
            for bin_idx in range(num_bins):
                required_count = bin_counts[bin_idx]
                if required_count == 0:
                    continue
                
                bin_start = config.temporal_bins[bin_idx]
                bin_end = config.temporal_bins[bin_idx + 1]
                
                valid_positions = find_valid_positions_in_bin(
                    pair_positions, bin_start, bin_end, excluded_set
                )
                
                if len(valid_positions) < required_count:
                    error_msg = (f"Insufficient positions for {config.label}, pair {pair_id}, bin {bin_idx}: "
                               f"need {required_count}, have {len(valid_positions)}")
                    return OneBackResult(
                        oneback_array=oneback_array,
                        assignment_order=assignment_order,
                        excluded_positions=list(excluded_set),
                        quality_metrics={},
                        success=False,
                        error_message=error_msg
                    )
                
                # Select positions for this bin
                bin_selections = sample(valid_positions, required_count)
                pair_selected_positions.extend(bin_selections)
                
                print(f"   Pair {pair_id}, Bin {bin_idx}: {required_count}/{len(valid_positions)} positions")
            
            # Update exclusion zones
            new_exclusions = update_exclusion_zones(
                pair_selected_positions, config.exclusion_radius, len(sequence_indices)
            )
            excluded_set.update(new_exclusions)
            
            # Assign markers
            markers, shuffled_positions = assign_element_markers(pair_selected_positions, config)
            
            for pos, marker in markers.items():
                oneback_array[pos] = marker
            
            assignment_order.extend(shuffled_positions)
            all_selected_positions.extend(pair_selected_positions)
        
        # Validate assignment quality
        quality_metrics = validate_assignment_quality(config, oneback_array, sequence_indices)
        
        print(f"✅ Assignment completed successfully!")
        print(f"   Total 1-backs: {quality_metrics['total_onebacks']}")
        print(f"   Success rate: {quality_metrics['success_rate']:.1%}")
        print(f"   Consecutive violations: {quality_metrics['consecutive_violations']}")
        print(f"   Spacing violations: {len(quality_metrics['spacing_violations'])}")
        
        return OneBackResult(
            oneback_array=oneback_array,
            assignment_order=assignment_order,
            excluded_positions=list(excluded_set),
            quality_metrics=quality_metrics,
            success=True
        )
        
    except Exception as e:
        return OneBackResult(
            oneback_array=oneback_array,
            assignment_order=assignment_order,
            excluded_positions=list(excluded_set),
            quality_metrics={},
            success=False,
            error_message=str(e)
        )
            
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
    
    # If distribution_bins is 1 or not specified properly, send error
    if distribution_bins <= 1:
        raise ValueError("distribution_bins must be greater than 1")
    
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
