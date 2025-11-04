# test_stim_gen.py

"""
Debug script for the separated stimuli generation modules.
Demonstrates targeted imports from the new modular structure.
"""

import os
from numpy.random import randint
from IPython.display import Image, display
from psychopy import data

# Import specific functions from the new separated modules
from utils.stimuli_generation import load_stimuli, generate_pairs, plot_stimuli

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

unique_stim, linking_stim_by_set, rank_dict = load_stimuli(rank_file=rank_file)
abcd_groups = generate_pairs(exp_info, unique_stim, linking_stim_by_set)
if plotting:
    plot_stimuli(exp_info, abcd_groups, image_folder)