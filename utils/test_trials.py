# test_trials.py
"""
Test trial generation for the SL-Obj-Inference task.
This module handles the creation of test trials with target/foil pairs.
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal
from itertools import combinations
import pandas as pd


def generate_test_trials(
    target_pairs: List[Tuple[str, str]], 
    n_repetitions: int, 
    seed: Optional[int] = None,
    abcd_groups: Optional[Dict[str, Dict[str, str]]] = None
) -> List[Dict]:
    """
    Generate test trials for a memory experiment with fixed foil sets across repetitions.

    Args:
        target_pairs: List of (stim1, stim2) tuples for target pairs.
        n_repetitions: Number of test repetitions to generate.
        seed: Optional random seed for deterministic behavior.
        abcd_groups: Optional mapping from group letters (A-D) to stimuli dicts.

    Returns:
        List of dictionaries, each representing a test trial with
        target/foil pairs and sequence assignments.
    """
    if not target_pairs:
        return []

    rng = random.Random(seed)
    target_pairs = target_pairs[:]  # avoid modifying input
    rng.shuffle(target_pairs)

    # === Step 1: Generate fixed foil pairs ===
    first_stimuli = [a for a, _ in target_pairs]
    second_stimuli = [b for _, b in target_pairs]
    foil_second_stimuli = second_stimuli[:]

    max_attempts = 5000
    for attempt in range(max_attempts):
        rng.shuffle(foil_second_stimuli)
        if all(t2 != f2 for (_, t2), f2 in zip(target_pairs, foil_second_stimuli)):
            break
    else:
        # Manual fallback: force swap any self-matching foils
        for i, ((_, t2), f2) in enumerate(zip(target_pairs, foil_second_stimuli)):
            if t2 == f2:
                for j in range(len(foil_second_stimuli)):
                    if i != j and target_pairs[i][1] != foil_second_stimuli[j]:
                        foil_second_stimuli[i], foil_second_stimuli[j] = (
                            foil_second_stimuli[j], foil_second_stimuli[i]
                        )
                        break

    fixed_foil_pairs = [(a, f2) for (a, _), f2 in zip(target_pairs, foil_second_stimuli)]

    # === Step 2: Initialize foil index assignment ===
    foil_indices = list(range(len(fixed_foil_pairs)))
    rng.shuffle(foil_indices)

    for _ in range(5000):
        conflicts = [i for i, idx in enumerate(foil_indices) if idx == i]
        if not conflicts:
            break
        for i in conflicts:
            swap_idx = next(
                (j for j in range(len(foil_indices)) 
                 if j != i and foil_indices[j] != i and foil_indices[i] != j),
                None
            )
            if swap_idx is not None:
                foil_indices[i], foil_indices[swap_idx] = (
                    foil_indices[swap_idx], foil_indices[i]
                )

    # === Step 3: Counterbalanced sequence assignment ===
    n_targets = len(target_pairs)
    seq1_assignments_rep1 = [True] * (n_targets // 2) + [False] * (n_targets - n_targets // 2)
    rng.shuffle(seq1_assignments_rep1)
    seq1_assignments_rep2 = [not x for x in seq1_assignments_rep1]

    # === Step 4: Trial generation across repetitions ===
    trials = []
    for rep in range(n_repetitions):
        rep_rng = random.Random((seed or 0) + rep)
        seq1_assignments = seq1_assignments_rep1 if rep == 0 else seq1_assignments_rep2

        targ_shuffle_idx = list(range(len(target_pairs)))
        foil_shuffle_idx = list(range(len(foil_indices)))
        rep_rng.shuffle(targ_shuffle_idx)
        rep_rng.shuffle(foil_shuffle_idx)
        print("Target shuffle indices:", targ_shuffle_idx)
        target_pairs_shuffled = [target_pairs[i] for i in targ_shuffle_idx]
        seq1_assignments_shuffled = [seq1_assignments[i] for i in targ_shuffle_idx]
        foil_indices_shuffled = [foil_indices[i] for i in foil_shuffle_idx]

        for target_idx, ((t1, t2), foil_idx, in_seq1) in enumerate(
            zip(target_pairs_shuffled, foil_indices_shuffled, seq1_assignments_shuffled)
        ):
            f1, f2 = fixed_foil_pairs[foil_idx]

            if in_seq1:
                seq1, seq2, correct_seq = (t1, t2), (f1, f2), "seq1"
            else:
                seq1, seq2, correct_seq = (f1, f2), (t1, t2), "seq2"

            targ1_label, targ1_grp_num, targ1_label_grp_num = get_stimulus_label(t1, abcd_groups)
            targ2_label, targ2_grp_num, targ2_label_grp_num = get_stimulus_label(t2, abcd_groups)
            foil1_label, foil1_grp_num, foil1_label_grp_num = get_stimulus_label(f1, abcd_groups)
            foil2_label, foil2_grp_num, foil2_label_grp_num = get_stimulus_label(f2, abcd_groups)
            
            # Determine target group number for record
            targ_grp_num = targ1_grp_num if targ1_grp_num == targ2_grp_num else "MIXED"

            trial = {
                "rep": rep,
                "block_num": 0,
                "targ_grp_num": targ_grp_num,
                "targ_stim1": t1,
                "targ_stim2": t2,
                "targ1_label": targ1_label,
                "targ2_label": targ2_label,
                "targ1_grp_num": targ1_grp_num,
                "targ2_grp_num": targ2_grp_num,
                "targ1_label_grp_num": targ1_label_grp_num,
                "targ2_label_grp_num": targ2_label_grp_num,
                "foil_stim1": f1,
                "foil_stim2": f2,
                "foil1_label": foil1_label,
                "foil2_label": foil2_label,
                "foil1_grp_num": foil1_grp_num,
                "foil2_grp_num": foil2_grp_num,
                "foil1_label_grp_num": foil1_label_grp_num,
                "foil2_label_grp_num": foil2_label_grp_num,
                "seq1_stim1": seq1[0],
                "seq1_stim2": seq1[1],
                "seq2_stim1": seq2[0],
                "seq2_stim2": seq2[1],
                "correct_seq": correct_seq,
                "correct_resp": "f" if correct_seq == "seq1" else "j", 
                "target_idx": target_idx,
                "foil_idx": foil_idx,
            }

            trials.append(trial)

    return trials


def get_direct_pairs(abcd_groups: Dict[str, Dict[str, str]]) -> List[Tuple[str, str, str]]:
    """Generate all direct pairs (AB, BC, CD) from the stimulus groups."""
    pairs = []
    for key in abcd_groups["A"]:
        num = key[1:]
        pairs.append(("AB", abcd_groups["A"][key], abcd_groups["B"][f"B{num}"]))
    for key in abcd_groups["B"]:
        num = key[1:]
        pairs.append(("BC", abcd_groups["B"][key], abcd_groups["C"][f"C{num}"]))
    for key in abcd_groups["C"]:
        num = key[1:]
        pairs.append(("CD", abcd_groups["C"][key], abcd_groups["D"][f"D{num}"]))
    return pairs


def get_indirect_pairs(abcd_groups: Dict[str, Dict[str, str]], step: Literal["1-step", "2-step"]) -> List[Tuple[str, str, str]]:
    """Generate indirect pairs (1-step: AC, BD; 2-step: AD)."""
    pairs = []
    if step == "1-step":
        for key in abcd_groups["A"]:
            num = key[1:]
            pairs.append(("AC", abcd_groups["A"][key], abcd_groups["C"][f"C{num}"]))
        for key in abcd_groups["B"]:
            num = key[1:]
            pairs.append(("BD", abcd_groups["B"][key], abcd_groups["D"][f"D{num}"]))
    elif step == "2-step":
        for key in abcd_groups["A"]:
            num = key[1:]
            pairs.append(("AD", abcd_groups["A"][key], abcd_groups["D"][f"D{num}"]))
    return pairs


def get_stimulus_label(stimulus: str, abcd_groups: Dict[str, Dict[str, str]]) -> str:
    """Return the stimulus label (e.g., 'A1') for a given stimulus filename."""
    for label, items in abcd_groups.items():
        # label = 'A', 'B', 'C', or 'D', items = dict of labels to filenames
        for label_grp_num, fname in items.items():
            # group = 'A1', 'B2', etc., fname = filename
            if fname == stimulus:
                grp_num = label_grp_num[1:]  # e.g., '1' from 'A1'
                return label, grp_num, label_grp_num
    return "UNKNOWN", "UNKNOWN", "UNKNOWN"


def get_pair_type_from_stimuli(stim1: str, stim2: str, abcd_groups: Dict[str, Dict[str, str]]) -> str:
    """Infer pair type (AB, AC, BD, etc.) based on two stimuli."""
    g1 = next((k for k, v in abcd_groups.items() if stim1 in v.values()), None)
    g2 = next((k for k, v in abcd_groups.items() if stim2 in v.values()), None)
    return f"{g1}{g2}" if g1 and g2 else "UNKNOWN"


def generate_all_test_trials(exp_info: Dict, abcd_groups: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Generate and combine all indirect and direct test trials."""
    seed = int(exp_info.get("subject", 1234))
    test_type = exp_info.get("test", "2-step")
    indirect_type = "2-step" if test_type == "2-step" else "1-step"

    print(f"\nGenerating {indirect_type} test trials...")

    # === Indirect block ===
    indirect_pairs = get_indirect_pairs(abcd_groups, indirect_type)
    indirect_targets = [(a, b) for _, a, b in indirect_pairs]
    indirect_trials = generate_test_trials(indirect_targets, 2, seed, abcd_groups)
    indirect_df = pd.DataFrame(indirect_trials)
    indirect_df["block_num"] = 0
    indirect_df["test_type"] = indirect_type
    indirect_df["pair_type"] = "AD" if indirect_type == "2-step" else indirect_df.apply(
        lambda r: get_pair_type_from_stimuli(r["target_stim1"], r["target_stim2"], abcd_groups),
        axis=1,
    )

    # === Direct block ===
    direct_pairs = get_direct_pairs(abcd_groups)
    direct_dfs = []
    direct_tests_blocks = ["AB", "BC", "CD"]
    all_combos = list(combinations(direct_tests_blocks, len(direct_tests_blocks)))
    # get modulo index to cycle through combinations based on subject
    cb_num = int(exp_info['subject']) % len(all_combos)
    test_order = all_combos[cb_num]
    print(f"Direct test order for subject {exp_info['subject']}: {test_order}")
    for ptype in test_order:
        p_pairs = [(a, b) for t, a, b in direct_pairs if t == ptype]
        if not p_pairs:
            continue
        trials = generate_test_trials(p_pairs, 2, seed + hash(ptype), abcd_groups)
        df = pd.DataFrame(trials)
        df["pair_type"] = ptype
        df["block_num"] = test_order.index(ptype) + 1
        direct_dfs.append(df)

    direct_df = pd.concat(direct_dfs, ignore_index=True)
    direct_df["test_type"] = "direct"

    combined = pd.concat([indirect_df, direct_df], ignore_index=True)
    # Add task and trial information to beginning
    
    combined["test_cb"] = cb_num
    combined["task"] = "test"
    combined["block_tnum"] = combined.groupby("block_num").cumcount()
    combined["trial_num"] = range(len(combined))
    combined["rep_tnum"] = combined.groupby(["block_num", "rep"]).cumcount()
    combined["block"] = combined["test_type"].map(
        {"1-step": "indirect", "2-step": "indirect", "direct": "direct"}
    )

    cols = ["task", "block", "test_type", "pair_type", "trial_num", "block_num", "block_tnum", "rep", "rep_tnum",
            "targ_grp_num", "targ_stim1", "targ_stim2", "targ1_label", "targ2_label", "targ1_grp_num",
            "targ2_grp_num", "targ1_label_grp_num", "targ2_label_grp_num",
            "foil_stim1", "foil_stim2", "foil1_label", "foil2_label", "foil1_grp_num", "foil2_grp_num",
            "foil1_label_grp_num", "foil2_label_grp_num",
            "seq1_stim1", "seq1_stim2", "seq2_stim1", "seq2_stim2",
            "correct_seq", "correct_resp", "target_idx", "foil_idx"]
    combined = combined[cols]   
    # === Save ===
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    outfile = out_dir / f"{exp_info['file_prefix']}_test-trials.csv"
    combined.to_csv(outfile, index=False)

    print(f"Saved test trials → {outfile}")
    print(f"Total: {len(combined)} | Indirect: {len(indirect_df)} | Direct: {len(direct_df)}")
    return combined
