# run_sl-obj-inference.py
#
# Object Statistical Learning Experiment with Intermixed Exposure and Inference Test
# ------------------------------------------------------------------------------


from psychopy import visual, core, event, data, gui, logging
from psychopy import __version__ as psychopy_ver
import pandas as pd
import os
import numpy as np
from numpy.random import random, randint, normal, shuffle, choice as randchoice

import stimuli_generation as stimgen # custom module

# -----------------------------
# set debug mode
# -----------------------------
DEBUG = True
if DEBUG:
    fullscr = False
    monitor = "debugMonitor"
    screen = 0
    win_size = [800, 600]
else:
    fullscr = True
    monitor = "testMonitor"
    screen = 1
    win_size = [1920, 1080]
    
# ------------------------
# Parameters
# ------------------------
data_folder = "data"
stim_folder = "assets"
practice_folder = os.path.join(stim_folder, "practice")
num_stim = 24
num_grps = 6
num_reps = 40
prob_1back = 0.1
img_dur = 0.5
isi_dur = 0.5
test_l_key = "f"
test_r_key = "j"

# -----------------------------
# Experiment setup
# -----------------------------
exp_name = "inmix-sl-obj-inference"
exp_info = {
    'subject': f"{randint(1000, 9999):04.0f}", 
    'session': "001",
    'exposure': ["retrospective","transitive"],
    'test': ["2-step","1-step"],
    'date|hid': data.getDateStr(format="%Y%m%d-%H%M"),
    'exp_name|hid': exp_name,
    'psychopyVersion|hid': psychopy_ver,
    'file_prefix|hid': "",  # will be set below
    'num_stim|hid': num_stim,
    'num_grps|hid': num_grps,
    'num_reps|hid': num_reps,
    'prob_1back|hid': prob_1back,
    }
dlg = gui.DlgFromDict(exp_info, title="Intermixed SL Object Inference") 
if not dlg.OK:
    core.quit()

# create data folder if needed
if not os.path.isdir(data_folder):
    os.makedirs(data_folder)

# define output filename
exp_info['file_prefix'] = u'sub-%s_%s_expo-%s_test-%s_%s' % (exp_info['subject'], exp_info['exp_name|hid'], exp_info['exposure'], exp_info['test'].replace("-", ""), exp_info['date|hid'])

# setup experiment handler
this_exp = data.ExperimentHandler(
    name="Inmix-SL-Obj-Inference",
    extraInfo=exp_info,
    dataFileName=os.path.join(data_folder, exp_info['file_prefix'])
)

# csv files
practice_exposure_csv = "practice_exposure-trials.csv"
exposure_trials_csv = os.path.join(data_folder, exp_info['file_prefix'] + "_exposure-trials.csv")
test_trials_csv = os.path.join(data_folder, exp_info['file_prefix'] + "_test-trials.csv")

# ------------------------------
# initialize stimuli
# ------------------------------
unique_stim, linking_stim_by_set = stimgen.load_stimuli(stim_folder=stim_folder)
abcd_groups = stimgen.generate_pairs(exp_info, unique_stim, linking_stim_by_set)
obj_stream_data = stimgen.generate_obj_stream(exp_info, abcd_groups)


# -----------------------------
# Window setup
# -----------------------------
win = visual.Window(size=win_size, fullscr=fullscr, screen=screen, monitor=monitor, 
                    color=[0, 0, 0], colorSpace='rgb', units='pix'
                    )

fixation = visual.Circle(win, radius=5, fillColor="white", lineColor="black", pos=(0,0))
instr_text = visual.TextStim(win, text="", color="white", height=26, wrapWidth=1000)

def show_instructions(text):
    """Show instructions."""
    instr_text.text = text
    instr_text.draw()
    win.flip()
    event.waitKeys()
    
# ------------------------
# Helper Functions
# ------------------------
def preload_images(images):
    cache = {}
    for img in set(images):
        path = os.path.join(stim_folder, img)
        if not os.path.exists(path):
            logging.error(f"Missing image: {path}")
            core.quit()
        cache[img] = visual.ImageStim(win, image=path, size=(200, 200))
    return cache

def present_trial(img_name, img_cache, duration, isi, trial_num, extra_info=None):
    """Show one image with fixation and log timing."""
    stim = img_cache[img_name]
    stim.draw()
    fixation.draw()
    onset_time = win.flip()
    clock = core.Clock()
    while clock.getTime() < duration:
        if "escape" in event.getKeys():
            win.close()
            core.quit()
        core.wait(0.001)
    # ISI: fixation only
    fixation.draw()
    win.flip()
    core.wait(isi)

    record = {
        "trial_num": trial_num,
        "image": img_name,
        "onset_time": onset_time,
        "duration": duration
    }
    if extra_info:
        record.update(extra_info)
    return record
def run_stream(csv_path, label="stream"):
    """Run a continuous stream from CSV."""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping {label}.")
        return []
    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        print(f"Error: {csv_path} must contain an 'image' column.")
        core.quit()
    cache = preload_images(df["image"])
    records = []
    for i, row in df.iterrows():
        img_name = row["image"]
        extra = {c: row[c] for c in df.columns if c != "image"}
        record = present_trial(img_name, cache, img_dur, isi_dur, i+1, extra)
        record["phase"] = label
        records.append(record)
    return records

def run_test():
    """Optional 2AFC test phase."""
    if not os.path.exists(test_trials_csv):
        print("No test phase found. Skipping.")
        return []
    df = pd.read_csv(test_trials_csv)
    if not {"left", "right", "correct_side"}.issubset(df.columns):
        print("test_pairs.csv must have 'left','right','correct_side' columns.")
        core.quit()

    # Preload images
    all_imgs = pd.unique(df[["left","right"]].values.ravel())
    cache = preload_images(all_imgs)
    results = []

    show_instructions(f"Test phase:\n\nPress '{test_l_key.upper()}' for LEFT\nPress '{test_r_key.upper()}' for RIGHT\n\nPress any key to begin.")

    left_stim = visual.ImageStim(win, size=(500,500), pos=(-300,0))
    right_stim = visual.ImageStim(win, size=(500,500), pos=(300,0))

    for t, row in df.iterrows():
        left_stim.image = os.path.join(stim_folder, row["left"])
        right_stim.image = os.path.join(stim_folder, row["right"])
        left_stim.draw()
        right_stim.draw()
        fixation.draw()
        win.flip()

        timer = core.Clock()
        key, rt = event.waitKeys(keyList=[test_l_key, test_r_key, "escape"], timeStamped=timer)[0]
        if key == "escape":
            win.close()
            core.quit()
        choice = "left" if key == test_l_key else "right"
        correct = int(choice == row["correct_side"])
        results.append({
            "trial_num": t+1,
            "left": row["left"],
            "right": row["right"],
            "correct_side": row["correct_side"],
            "choice": choice,
            "accuracy": correct,
            "rt": rt
        })

        # Feedback
        fb_color = "green" if correct else "red"
        fb = visual.TextStim(win, text="Correct" if correct else "Incorrect", color=fb_color, height=40)
        fb.draw()
        fixation.draw()
        win.flip()
        core.wait(0.5)
    return results

# ------------------------
# Experiment Flow
# ------------------------
show_instructions("Welcome!\n\nYou will see a sequence of images.\nKeep your eyes on the central dot at all times.\n\nPress any key to begin practice.")

# Practice phase
practice_data = run_stream(practice_exposure_csv, label="practice")

show_instructions("Great job!\nNow the real task will begin.\nAgain, keep your eyes on the central dot.\n\nPress any key to start.")

# Exposure phase
main_data = run_stream(exposure_trials_csv, label="exposure")

# Optional test
test_data = run_test()

# ------------------------
# Save Data
# ------------------------
all_data = practice_data + main_data
df_all = pd.DataFrame(all_data)
test_df = pd.DataFrame(test_data)

df_all.to_csv(os.path.join(data_folder, f"{exp_info['file_prefix']}_exposure-data.csv"), index=False)
if len(test_data) > 0:
    test_df.to_csv(os.path.join(data_folder, f"{exp_info['file_prefix']}_test-data.csv"), index=False)

# -----------------------------
# Wrap up
# -----------------------------
show_instructions("Thank you for participating!\n\nThis concludes the experiment.")
event.clearEvents()
win.close()
core.quit()
