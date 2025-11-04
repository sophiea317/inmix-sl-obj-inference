"""
Debug script for the separated stimuli generation modules.
Demonstrates targeted imports from the new modular structure.
"""

import os
from numpy.random import randint
from IPython.display import Image, display
from psychopy import data

# Import specific functions from the new separated modules
from utils.stimuli_generation2 import load_stimuli, generate_pairs, plot_stimuli
from utils.exposure_trials import generate_obj_stream, create_image_trials
from utils.test_trials import generate_test_trials, generate_all_test_trials


def get_set_id(filename: str) -> str:
    """Extract set ID from a filename. Returns 'unique' if not in a set."""
    if filename.startswith("set-"):
        return filename.split("_")[0]  # e.g., 'set-01'
    return "unique"

plotting = True  # set to True to visualize stimulus group assignments
printing = True  # set to True to print debugging info
# ----------------------
# set up experiment info
# ----------------------
exp_name = "sl-obj-inference"
psychopy_ver = "2025.1.0"

exp_info = {
    "subject": "6310", #f"{randint(1000, 9999):04.0f}",  # random 4-digit subject ID
    "session": "001",
    "exposure": "retrospective",
    "test": "2-step",
    "date|hid": "2025-0830", #data.getDateStr(format="%Y%m%d-%H%M"),
    "exp_name|hid": exp_name,
    "psychopyVersion|hid": psychopy_ver,
    "file_prefix|hid": "",  # will be set below
    "num_stim|hid": 24,
    "num_grps|hid": 6,
    "num_reps|hid": 40
}
exp_info['file_prefix'] = u'sub-%s_%s_expo-%s_test-%s_%s' % (exp_info['subject'], exp_info['exp_name|hid'], exp_info['exposure'], exp_info['test'].replace("-", ""), exp_info['date|hid'])

print("=== Debugging stimuli_generation ===")
print(f"Subject: {exp_info['subject']}, Session: {exp_info['session']}")

# ----------------------
# Run stimulus pipeline
# ----------------------
stim_folder = "stimuli"
image_folder = os.path.join(stim_folder, "images")
rank_file = os.path.join(stim_folder, "stimulus_size-rank.csv")
try:
    unique_stim, linking_stim_by_set, rank_dict = load_stimuli(rank_file=rank_file)
    abcd_groups = generate_pairs(exp_info, unique_stim, linking_stim_by_set)
    obj_stream_data = generate_obj_stream(exp_info, abcd_groups, rank_dict)
    test_trials_df = generate_all_test_trials(exp_info, abcd_groups)
    if plotting := True:
        plot_stimuli(exp_info, abcd_groups, image_folder)
except Exception as e:
    print("Error during stimulus generation:", e)
    raise


# ---------------------------------
# Print debugging info with set IDs
# ---------------------------------
if printing:
    print("\n--- A-B-C-D assignments ---")
    a_items = list(abcd_groups["A"].items())
    b_items = list(abcd_groups["B"].items())
    c_items = list(abcd_groups["C"].items())
    d_items = list(abcd_groups["D"].items())

    for i, ((a_label, a), (b_label, b), (c_label, c), (d_label, d)) in enumerate(zip(a_items, b_items, c_items, d_items), start=1):
        print(f"Pair {i}: A = {get_set_id(a)} {a.split('_')[-1]} \t B = {get_set_id(b)} {b.split('_')[-1]} \t C = {get_set_id(c)} {c.split('_')[-1]} \t D = {get_set_id(d)} {d.split('_')[-1]}")

    print("\n--- Quick checks ---")
    for i in range(len(abcd_groups["A"])):
        a = list(abcd_groups["A"].values())[i]
        b = list(abcd_groups["B"].values())[i]
        c = list(abcd_groups["C"].values())[i]
        d = list(abcd_groups["D"].values())[i]

        a_set, b_set, c_set, d_set = get_set_id(a), get_set_id(b), get_set_id(c), get_set_id(d)

        
        if a_set in {b_set, c_set} or d_set in {b_set, c_set}:
            print(f"Pair {i+1} check: A={a_set}, B={b_set}, C={c_set}, D={d_set} \t ❌ ERROR: A or D overlaps with B–C set!")
        else:
            print(f"Pair {i+1} check: A={a_set}, B={b_set}, C={c_set}, D={d_set} \t ✅ OK")
            
# ----------------------
# Test trial generation
# ----------------------
print("\n=== Testing Test Trial Generation ===")

try:

    print(f"✅ Generated {len(test_trials_df)} total test trials")
    
    # Show trials per pair_type
    print("\n📊 Trials per pair_type:")
    pair_counts = test_trials_df['pair_type'].value_counts().sort_index()
    for pair_type, count in pair_counts.items():
        print(f"  {pair_type}: {count} trials")
    
    # Verify structure
    expected_per_pair = 12  # 6 groups × 2 repetitions each
    success = all(count == expected_per_pair for count in pair_counts.values)
    print(f"\n✅ All pair_types have exactly {expected_per_pair} trials: {success}")
    
    if success:
        print("🎉 Perfect! Each group tested twice per pair_type.")
    
    # Show detailed structure for first few trials
    print(f"\n📋 First 8 trials structure:")
    display_cols = ['phase', 'block_num', 'block_tnum', 'pair_type', 'rep_num', 'rep_tnum',
                   'target_stim1', 'target_stim2', 'foil_stim1', 'foil_stim2', 
                   'test_type', 'correct_seq']
    
    if all(col in test_trials_df.columns for col in display_cols):
        print(test_trials_df[display_cols].head(8).to_string(index=False, max_colwidth=12))
    else:
        # Show available columns
        available_cols = [col for col in display_cols if col in test_trials_df.columns]
        print(f"Available columns: {available_cols}")
        print(test_trials_df[available_cols].head(8).to_string(index=False, max_colwidth=12))
    
    # Show repetition structure
    print(f"\n📊 Repetition structure:")
    if 'rep_num' in test_trials_df.columns:
        rep_structure = test_trials_df.groupby(['pair_type', 'rep_num']).size().unstack(fill_value=0)
        print(rep_structure)
    
    # Show sequence balance
    if 'correct_seq' in test_trials_df.columns:
        print(f"\n✅ Sequence balance:")
        seq_counts = test_trials_df['correct_seq'].value_counts()
        print(seq_counts)
        
        balance_ratio = seq_counts.min() / seq_counts.max() if seq_counts.max() > 0 else 0
        print(f"Balance ratio: {balance_ratio:.2f} (1.0 = perfect balance)")
    
    # Save test file for inspection
    test_filename = f"data/{exp_info['file_prefix']}_test-trials.csv"
    test_trials_df.to_csv(test_filename, index=False)
    print(f"\n💾 Test trials saved to: {test_filename}")
    
except Exception as e:
    print(f"❌ Error during test trial generation: {e}")
    import traceback
    traceback.print_exc()

# ----------------------
# Test new generate_test_trials function
# ----------------------
print("\n=== Testing New generate_test_trials Function ===")

# try:
#     # Extract some target pairs from the ABCD groups for testing
#     print("🧪 Testing with extracted target pairs from ABCD groups...")
    
#     # Create simple target pairs from AB pairs (use all 6 for better demonstration)
#     ab_targets = []
#     for i, (a_key, a_stim) in enumerate(abcd_groups["A"].items()):
#         pair_num = a_key[1:]  # Extract number (e.g., "1" from "A1")
#         b_stim = abcd_groups["B"][f"B{pair_num}"]
#         ab_targets.append((a_stim, b_stim))
    
#     print(f"Target pairs: {ab_targets}")
    
#     # Test the new function with multiple seeds to show different outcomes
#     print("\n--- Test 1: With seed=42 ---")
#     simple_trials = generate_test_trials(ab_targets, n_repetitions=2, seed=42)
    
#     print(f"✅ Generated {len(simple_trials)} trials with new function")
#     print(f"Expected: {len(ab_targets)} targets × 2 repetitions = {len(ab_targets) * 2}")
    
#     # Show detailed structure
#     print("\n📋 New function trial structure:")
#     for i, trial in enumerate(simple_trials):
#         target = f"{trial['target_stim1'].split('_')[-1]}-{trial['target_stim2'].split('_')[-1]}"
#         foil = f"{trial['foil_stim1'].split('_')[-1]}-{trial['foil_stim2'].split('_')[-1]}"
#         print(f"  Trial {i+1}: Rep {trial['repetition']}, "
#               f"Target: {target}, Foil: {foil}, "
#               f"Correct: {trial['correct_seq']}")
    
#     # Verify foil are not the same across repetitions
#     print("\nFoil rep check:")
#     foil_map = {}
#     for trial in simple_trials:
#         target_key = (trial['target_stim1'], trial['target_stim2'])
#         foil_key = (trial['foil_stim1'], trial['foil_stim2'])
#         rep = trial['repetition']
        
#         if target_key not in foil_map:
#             foil_map[target_key] = {}
#         foil_map[target_key][rep] = foil_key
    
#     all_same = True
#     for target, reps in foil_map.items():
#         foils_same = len(set(reps.values())) == 1
#         target_short = f"{target[0].split('_')[-1]}-{target[1].split('_')[-1]}"
#         foil_reps = {rep: f"{foil[0].split('_')[-1]}-{foil[1].split('_')[-1]}" 
#                     for rep, foil in reps.items()}
#         status = "✅" if not foils_same else "❌"
#         print(f"  Target {target_short}: {status} Foils: {foil_reps}")
#         if foils_same:
#             all_same = False

#     if not all_same:
#         print("🎉 Perfect! All targets use consistent foils across repetitions")
#     else:
#         print("⚠️  Some targets have inconsistent foils across repetitions")
    
#     # Check sequence balance
#     seq1_count = sum(1 for t in simple_trials if t['correct_seq'] == 'seq1')
#     seq2_count = sum(1 for t in simple_trials if t['correct_seq'] == 'seq2')
#     print(f"\n📊 Sequence balance: seq1={seq1_count}, seq2={seq2_count}")
    
#     # Test foil reassignment across repetitions
#     print("\n🔄 Foil reassignment verification:")
#     # rep1_assignments = [(t['target_idx'], t['foil_idx']) for t in simple_trials if t['rep'] == 0]
#     # rep2_assignments = [(t['target_idx'], t['foil_idx']) for t in simple_trials if t['rep'] == 1]

#     same_assignments = sum(1 for r1, r2 in zip(rep1_assignments, rep2_assignments) if r1 == r2)
#     total_assignments = len(rep1_assignments)
    
#     print(f"  Same assignments across reps: {same_assignments}/{total_assignments}")
#     print(f"  Different assignments: {total_assignments - same_assignments}/{total_assignments}")
    
#     if same_assignments < total_assignments:
#         print("✅ Good! Foils are reassigned between repetitions")
#     else:
#         print("⚠️  All foil assignments are the same across repetitions")
    
#     # Test with different seed to show variability
#     print("\n--- Test 2: With seed=123 (different randomization) ---")
#     simple_trials2 = generate_test_trials(ab_targets, n_repetitions=2, seed=123)
    
#     seq1_count2 = sum(1 for t in simple_trials2 if t['correct_seq'] == 'seq1')
#     seq2_count2 = sum(1 for t in simple_trials2 if t['correct_seq'] == 'seq2')
#     print(f"Sequence balance: seq1={seq1_count2}, seq2={seq2_count2}")
    
#     # Check foil reassignment for this test
#     rep1_assignments2 = [(t['target_idx'], t['foil_idx']) for t in simple_trials2 if t['repetition'] == 1]
#     rep2_assignments2 = [(t['target_idx'], t['foil_idx']) for t in simple_trials2 if t['repetition'] == 2]
#     same_assignments2 = sum(1 for r1, r2 in zip(rep1_assignments2, rep2_assignments2) if r1 == r2)
    
#     print(f"Foil reassignment: {len(rep1_assignments2) - same_assignments2}/{len(rep1_assignments2)} different")
    
#     if same_assignments2 < len(rep1_assignments2):
#         print("✅ Good! Foils are reassigned between repetitions")
#     else:
#         print("⚠️  All foil assignments are the same across repetitions")
    
#     # Show a few example trials from the second test
#     print("\nExample trials from Test 2:")
#     for i, trial in enumerate(simple_trials2[:4]):
#         target = f"{trial['target_stim1'].split('_')[-1]}-{trial['target_stim2'].split('_')[-1]}"
#         foil = f"{trial['foil_stim1'].split('_')[-1]}-{trial['foil_stim2'].split('_')[-1]}"
#         print(f"  Trial {i+1}: Rep {trial['repetition']}, "
#               f"Target: {target}, Foil: {foil}, "
#               f"Correct: {trial['correct_seq']}")

# except Exception as e:
#     print(f"❌ Error testing new generate_test_trials function: {e}")
#     import traceback
#     traceback.print_exc()


print("abcd_groups: ", abcd_groups['A']['A1'])
