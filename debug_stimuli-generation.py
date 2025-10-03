"""
Debug script for stimuli_generation.py
"""

import os
from numpy.random import randint
from psychopy import data
import stimuli_generation as stimgen


def get_set_id(filename: str) -> str:
    """Extract set ID from a filename. Returns 'unique' if not in a set."""
    if filename.startswith("set-"):
        return filename.split("_")[0]  # e.g., 'set-01'
    return "unique"

plotting = True  # set to True to visualize stimulus group assignments
# ----------------------
# set up experiment info
# ----------------------
exp_name = "sl-obj-inference"
psychopy_ver = "2025.1.0"

exp_info = {
    "subject": f"{randint(1000, 9999):04.0f}",  # random 4-digit subject ID
    "session": "001",
    "exposure": "retrospective",
    "test": "2-step",
    "date|hid": data.getDateStr(format="%Y%m%d-%H%M"),
    "exp_name|hid": exp_name,
    "psychopyVersion|hid": psychopy_ver,
    "file_prefix|hid": "",  # will be set below
    "num_stim|hid": 24,
    "num_grps|hid": 6,
    "num_reps|hid": 40,
    "prob_1back|hid": 0.1,
}
exp_info['file_prefix'] = u'sub-%s_%s_expo-%s_test-%s_%s' % (exp_info['subject'], exp_info['exp_name|hid'], exp_info['exposure'], exp_info['test'].replace("-", ""), exp_info['date|hid'])

print("=== Debugging stimuli_generation ===")
print(f"Subject: {exp_info['subject']}, Session: {exp_info['session']}")

# ----------------------
# Run stimulus pipeline
# ----------------------
try:
    unique_stim, linking_stim_by_set = stimgen.load_stimuli("assets")
    abcd_groups = stimgen.generate_pairs(exp_info, unique_stim, linking_stim_by_set)
    if plotting := True:
        stimgen.plot_stimuli(exp_info, abcd_groups)
except Exception as e:
    print("Error during stimulus generation:", e)
    raise

print("\nStimulus generation successful!\n")

print("abcd_groups:", abcd_groups)

# ----------------------
# Print debugging info with set IDs
# ----------------------
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

