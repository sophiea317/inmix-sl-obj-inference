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
    a_stim, d_stim, rank_num = {}, {}, {}
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
            set_a = get_set_id(candidate_a)
            set_d = get_set_id(candidate_d)
            forbidden_sets = {get_set_id(b_img), get_set_id(c_img)}
            
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
        row = {
            "subject": exp_info["subject"],
            "session": exp_info["session"],
            "group_num": pair_idx + 1,   # 1-based
            "A": list(abcd_groups["A"].values())[pair_idx],
            "B": list(abcd_groups["B"].values())[pair_idx],
            "C": list(abcd_groups["C"].values())[pair_idx],
            "D": list(abcd_groups["D"].values())[pair_idx]
        }
        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{exp_info['file_prefix']}_stim-assignments.csv"), index=False)

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