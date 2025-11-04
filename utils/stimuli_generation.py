# stimuli_generation.py
"""
Core stimulus handling utilities for the SL-Obj-Inference task.
This module contains shared functions for loading stimuli and generating ABCD group assignments.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
from numpy.random import randint, choice as randchoice


def load_stimuli(rank_file):
    """Load and categorize stimuli from the folder."""
    rank_dict = pd.read_csv(rank_file).set_index('image')['size_rank'].to_dict()
    stim_set = rank_dict.keys()   #[f for f in os.listdir(stim_folder) if f.endswith(".png")]
    unique_stim = [f for f in stim_set if "unique" in f]
    linking_stim = [f for f in stim_set if "set" in f]

    # separate linking stimuli by set
    linking_stim_by_set = {}
    for f in np.sort(linking_stim):
        set_id = f.split("_")[0]
        linking_stim_by_set.setdefault(set_id, []).append(f)

    return unique_stim, linking_stim_by_set, rank_dict


def get_set_id(filename):
    """Extract set ID from filename. Returns individual IDs for unique stimuli."""
    if filename.startswith("set-"):
        return filename.split("_")[0]  # e.g., 'set-01'
    # For unique stimuli, treat each as its own set to allow pairing
    return filename  # e.g., 'unique_img-01.png'


def generate_pairs(exp_info, unique_stim, linking_stim_by_set):
    """
    Generate AB, BC, CD stimulus groups for the SL-Obj-Inference task.
    
    Returns:
        abcd_groups (dict): dict with A, B, C, D groups (stimulus assignments)
    """
    # Seed random generator per subject for reproducibility
    np.random.seed(int(exp_info["subject"]))
    random.seed(int(exp_info["subject"]))
    
    # even subjects: pair-cb1 (counterbalance 1), odd subjects: pair-cb2 (counterbalance 2)
    cb_num = 1 if int(exp_info["subject"]) % 2 == 0 else 2
    
    # exclude stimuli not for subject's counterbalance
    # e.g., for set 1 (set-01_pair-cb1.jpg, set-01_pair-cb2.jpg, set-01_pair-lap.jpg), if cb_num=1, exclude pair-cb2
    def filter_stimuli(stim_list, cb_num):
        filtered = []
        for stim in stim_list:
            if f"pair-cb{cb_num}" in stim or "pair-lap" in stim:
                filtered.append(stim)
        return filtered

    # apply filtering
    for set_id, stim_list in linking_stim_by_set.items():
        linking_stim_by_set[set_id] = filter_stimuli(stim_list, cb_num)

    print(f"Filtered linking stimuli for counterbalance {cb_num}: {linking_stim_by_set}")
    # --- Generate B-C pairs ---
    b_stim, c_stim = {}, {}
    pair_counter = 0

    for set_id, imgs in linking_stim_by_set.items():
        # BC pair: pair-lap + pair-cb[cb_num]
        item_1 = imgs[1] # pair-lap
        item_2 = imgs[0] # pair-cb[cb_num]
        
        if np.random.randint(0, 2) == 0:
            b_stim[f"B{pair_counter+1}"], c_stim[f"C{pair_counter+1}"] = item_1, item_2
        else:
            b_stim[f"B{pair_counter+1}"], c_stim[f"C{pair_counter+1}"] = item_2, item_1
        pair_counter += 1
        
    # --- Generate A and D stimuli ---
    a_stim, d_stim, rank_num = {}, {}, {}
    bc_pairs = list(zip(b_stim.values(), c_stim.values()))
    ad_stim_choices = unique_stim.copy()
    
    for i, (b_img, c_img) in enumerate(bc_pairs):
        a_img =  np.random.choice(ad_stim_choices)
        ad_stim_choices.remove(a_img)
        d_img =  np.random.choice(ad_stim_choices)
        ad_stim_choices.remove(d_img)
        a_stim[f"A{i+1}"] = a_img
        d_stim[f"D{i+1}"] = d_img
        
        
    # shuffle the 6 rows of each stimulus assignment
    row_indices = np.arange(len(a_stim))
    np.random.shuffle(row_indices)
    # reorder stimulus dicts
    a_stim = {f"A{i+1}": a_stim[f"A{row_indices[i]+1}"] for i in range(len(a_stim))}
    b_stim = {f"B{i+1}": b_stim[f"B{row_indices[i]+1}"] for i in range(len(b_stim))}
    c_stim = {f"C{i+1}": c_stim[f"C{row_indices[i]+1}"] for i in range(len(c_stim))}
    d_stim = {f"D{i+1}": d_stim[f"D{row_indices[i]+1}"] for i in range(len(d_stim))}
    
    abcd_groups = {
        "A": a_stim,
        "B": b_stim,
        "C": c_stim,
        "D": d_stim
    }
    
    # --- Save assignments to CSV ---
    df_rows = []
    num_pairs = len(abcd_groups["A"])   # should be 6
    for i in range(num_pairs):
        row = {
            "subject": exp_info["subject"],
            "session": exp_info["session"],
            "cb_num": cb_num,
            "group_num": i + 1,   
            "A": list(abcd_groups["A"].values())[i],
            "B": list(abcd_groups["B"].values())[i],
            "C": list(abcd_groups["C"].values())[i],
            "D": list(abcd_groups["D"].values())[i]
        }
        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{exp_info['file_prefix']}_stim-assignments.csv")
    df.to_csv(save_path, index=False)

    return abcd_groups

def plot_stimuli(exp_info, abcd_groups, stim_folder):
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
    
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{exp_info['file_prefix']}_stim-groups.png")
    plt.savefig(save_path)
    plt.close()