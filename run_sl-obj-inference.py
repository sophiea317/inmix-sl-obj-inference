# run_sl-obj-inference.py
#
# Object Statistical Learning Experiment with Intermixed Exposure and Inference Test
# -----------------------------------------------------------------------------------
# How to run with defaults:
# python run_sl-obj-inference.py --sub 1234 --test 2-step
# How to run with all args specified:
# python run_sl-obj-inference.py --sub 1234 --ses 001 --expo retrospective --test 2-step --debug f --practice t --expoRun t --testRun t
# python run_sl-obj-inference.py --sub 1234 --ses 001 --expo retrospective --test 2-step --debug t --practice f


from psychopy import visual, core, event, data, gui, logging
from psychopy import __version__ as psychopy_ver
import pandas as pd
import os
import numpy as np
import sys
import argparse

# Import specific functions from custom modules
from utils.stimuli_generation import load_stimuli, generate_pairs, plot_stimuli
from utils.exposure_trials import generate_obj_stream
from utils.test_trials import generate_all_test_trials
from utils.onscreen_text import *

# -----------------------------
# Argument parsing
# -----------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1", "y")
parser = argparse.ArgumentParser(description="Run Inmix SL Object Inference experiment.")
# Optional flagged args 
parser.add_argument("--sub", type=str, default=None, help="Subject number (e.g., 1234)")
parser.add_argument("--ses", type=str, default="001", help="Session number (e.g., 001)")
parser.add_argument("--expo", type=str, default="retrospective", help="Exposure type (e.g., retrospective or transitive)")
parser.add_argument("--test", type=str, default=None, help="Test type (e.g., 1-step or 2-step)")
parser.add_argument("--debug", type=str2bool, default=False, help="Run in debug mode (True/False)")
parser.add_argument("--practice", type=str2bool, default=True, help="Run practice phase (True/False)")
parser.add_argument("--expoRun", type=str2bool, default=True, help="Run exposure phase (True/False)")
parser.add_argument("--testRun", type=str2bool, default=True, help="Run test phase (True/False)")
args = parser.parse_args()

# if no args provided, will prompt with dialog
skip_dialog = all([args.sub is not None, args.test is not None])

# -----------------------------
# Configuration / constants
# -----------------------------
# Phase toggles
RUN_PRACTICE = args.practice
RUN_EXPOSURE = args.expoRun
RUN_TESTS = args.testRun

if args.debug:
    FULLSCREEN = False
    MONITOR = "debugMonitor"
    SCREEN = 0
    WIN_SIZE = [800, 600]
    bug = 0.5
else:
    FULLSCREEN = True
    MONITOR = "testMonitor"
    SCREEN = 1
    WIN_SIZE = [1920, 1080]
    bug = 1
    
# Experiment folders and constants
exp_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(exp_folder, "data")
stim_folder = "stimuli"
imgs_folder = os.path.join(exp_folder, stim_folder, "images")
practice_folder = os.path.join(exp_folder, "practice")
practice_stim_folder = os.path.join(practice_folder, stim_folder, "images")
num_stim = 24
num_grps = 6
num_reps = 40

# Timing + keys
IMG_DUR = 1.0*bug                       # 1000 ms per image in stream
ISI_DUR = 0.5*bug                       # 500 ms between images in stream
PAUSE_DUR = (IMG_DUR*2)*bug             # 2500 ms pause between seqs in test
FBCK_DUR = 0.12*bug                     # 150 ms feedback flash on response
BREAK_DUR = 15                          # seconds for break countdown
EXPO_BIGGER_KEY = "f"
EXPO_SMALLER_KEY = "j"
TEST_SEQ1_KEY = "f"
TEST_SEQ2_KEY = "j"

# experiment constants
FIX_LINE = "white"                      # fixation line color
FIX_FILL = "black"                      # fixation fill color


# -----------------------------
# Experiment setup
# -----------------------------
exp_name = "inmix-sl-obj-inf"
default_sub = f"{np.random.randint(1000, 9999):04d}" if args.sub is None else args.sub
exp_info = {
    'subject': default_sub,
    'session': "001" if args.ses is None else args.ses,
    'exposure': "retrospective" if args.expo is None else args.expo,
    'test': "2-step" if args.test is None else args.test,
    'date|hid': data.getDateStr(format="%Y%m%d-%H%M"),
    'exp_name|hid': exp_name,
    'psychopyVersion|hid': psychopy_ver,
    'file_prefix|hid': "",
    'num_stim|hid': num_stim,
    'num_grps|hid': num_grps,
    'num_reps|hid': num_reps,
}

# Show dialog unless CLI args were provided
if not skip_dialog:
    dlg = gui.DlgFromDict(exp_info, title="Intermixed SL Object Inference")
    if not dlg.OK:
        core.quit()

# Create data folder if necessary
os.makedirs(data_folder, exist_ok=True)

# File prefix
exp_info['file_prefix'] = (
    #f"sub-{exp_info['subject']}_{exp_info['exp_name|hid']}_test-{exp_info['test'].replace('-', '')}_{exp_info['date|hid']}"
    f"sub-{exp_info['subject']}_{exp_info['exp_name|hid']}_test-{exp_info['test'].replace('-', '')}_{exp_info['date|hid']}_id-{exp_info['session'].lower()}"
)

this_exp = data.ExperimentHandler(
    name="Inmix-SL-Obj-Inference",
    extraInfo=exp_info,
    dataFileName=os.path.join(data_folder, f"{exp_info['file_prefix']}_raw-data"), 
    savePickle=True, saveWideText=True,
)

# CSV paths
exposure_trials_csv = os.path.join(data_folder, exp_info['file_prefix'] + "_exposure-trials.csv")
test_trials_csv = os.path.join(data_folder, exp_info['file_prefix'] + "_test-trials.csv")
rank_file = os.path.join(stim_folder, "stimulus_size-rank.csv")
practice_exposure_csv = os.path.join(practice_folder, "practice_exposure-trials.csv")
practice_test_csv = os.path.join(practice_folder, "practice_test-trials.csv")

# ------------------------------
# Initialize stimuli
# ------------------------------
unique_stim, linking_stim_by_set, rank_dict, obj_dict = load_stimuli(rank_file=rank_file)
abcd_groups = generate_pairs(exp_info, unique_stim, linking_stim_by_set)
obj_stream_data = generate_obj_stream(exp_info, abcd_groups, rank_dict, obj_dict)
test_trials_df = generate_all_test_trials(exp_info, abcd_groups)

# -----------------------------
# Window set-up
# -----------------------------
win = visual.Window(
    size=WIN_SIZE, fullscr=FULLSCREEN, screen=SCREEN, monitor=MONITOR,
    color=[0, 0, 0], colorSpace='rgb', units='pix'
)

# size calcs
win_size = win.size
wrap_wth = win_size[0]*0.65         # 65% of window width for text wrapping
txt_sz = win_size[1]*0.04           # 4% of window height for text size
res = int(win_size[1]*0.4)          # 40% of window height for image size
img_sz = (res, res)                 # image size
expo_txt_pos = -win_size[1]*0.25   # 25% down from center for response text
small_txt_sz = txt_sz*0.75          # smaller text for exposure text

print(f"Window size: {win_size}\n Text wrap width: {wrap_wth}\n Text size: {txt_sz}\n Image size: {img_sz}\n Small text size: {small_txt_sz}\n Exposure text Y pos: {expo_txt_pos}")
logging.info(f"Window size: {win_size}\n Text wrap width: {wrap_wth}\n Text size: {txt_sz}\n Image size: {img_sz}\n Small text size: {small_txt_sz}\n Exposure text Y pos: {expo_txt_pos}") 
# fixation constants
radius=(res*0.012)*bug          # 1% of image size for fixation radius
lineWidth=1.75*bug

# central fixation, instruction text, and response prompt
fixation = visual.Circle(win, radius=radius, lineWidth=lineWidth, fillColor=FIX_FILL, lineColor=FIX_LINE, pos=(0, 0)) # outline fixation thickness*bug
instr_text = visual.TextStim(win, text="", color="white", height=txt_sz, wrapWidth=wrap_wth)
response_text_stim = visual.TextStim(win, text="", color="white", height=small_txt_sz, pos=(0, expo_txt_pos))

# frame-rate and logging
refresh_rate = win.getActualFrameRate()
if refresh_rate is None:
    refresh_rate = 60.0  # assume 60Hz if can't measure
logging.info(f"Monitor refresh rate: {refresh_rate:.1f} Hz")

# ------------------------
# helper class & functions
# ------------------------

class FixationFlashHandler:
    """Handles frame-based fixation flashing without blocking.

    This instance can be reused across trials by calling set_normal() at trial start.
    """
    def __init__(self, fixation_stim, normal_fill, normal_line, flash_fill, flash_line, flash_duration_sec, refresh_rate):
        self.fixation = fixation_stim
        self.normal_fill = normal_fill
        self.normal_line = normal_line
        self.flash_fill = flash_fill
        self.flash_line = flash_line
        self.flash_duration_frames = int(flash_duration_sec * refresh_rate)
        self.set_normal()

    def set_normal(self):
        self.fixation.fillColor = self.normal_fill
        self.fixation.lineColor = self.normal_line
        self.state = "normal"
        self.flash_frame_counter = 0

    def start_flash(self):
        self.state = "flashing"
        self.flash_frame_counter = 0
        self.fixation.fillColor = self.flash_fill
        self.fixation.lineColor = self.flash_line

    def update(self):
        if self.state == "flashing":
            self.flash_frame_counter += 1
            if self.flash_frame_counter >= self.flash_duration_frames:
                self.set_normal()

    def is_flashing(self):
        return self.state == "flashing"

# Create one flash handler and reuse it across trials
_flash_handler = FixationFlashHandler(
    fixation_stim=fixation,
    normal_fill=FIX_FILL,
    normal_line=FIX_LINE,
    flash_fill=FIX_LINE,  # inverted
    flash_line=FIX_FILL,
    flash_duration_sec=FBCK_DUR,
    refresh_rate=refresh_rate
)

def show_instructions(text: str) -> None:
    """Show instructions and wait for keypress."""
    instr_text.text = text
    instr_text.draw()
    win.flip()
    event.waitKeys()

def preload_images(images, practice=False):
    """Preload a set of images into a cache of ImageStim objects."""
    cache = {}
    # Filter out NaN values and convert to set
    unique_images = set(images.dropna())
    # If images are empty, return empty cache
    if not unique_images:
        return cache

    # Check existence once
    missing = []
    for img in unique_images:
        if practice:
            path = img
        else:
            path = os.path.join(imgs_folder, img)
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        logging.error(f"Missing image(s): {missing}")
        core.quit()

    for img in unique_images:
        if practice:
            path = img
        else:
            path = os.path.join(imgs_folder, img)
        cache[img] = visual.ImageStim(win, image=path, size=img_sz)
    return cache

def get_correct_key(extra_info):
    """Return correct response key from possible column names."""
    if not extra_info:
        return None
    return extra_info.get("correct_resp") or extra_info.get("correct_ans")

def break_trial():
    """Present a break screen that counts down from BREAK_DUR seconds or until keypress."""
    break_start_time = core.getTime()
    last_update_time = break_start_time
    
    # Initial display
    remaining = BREAK_DUR
    instr_text.text = (
        f"Take a short break!\n\n"
        f"Time remaining: {BREAK_DUR} seconds...\n\n"
        f"Press any key to continue to the experiment.\n"
        f"Otherwise, the experiment will start again immediately after the countdown."
    )
    instr_text.draw()
    break_display_onset = win.flip()
    
    # Track all screen updates
    update_times = [break_display_onset]
    
    # update text each second (NOT frame-based)
    while True:
        current_time = core.getTime()
        elapsed = current_time - break_start_time
        
        # Update display every second
        if current_time - last_update_time >= 1.0:
            remaining = max(0, BREAK_DUR - int(elapsed))
            instr_text.text = (
                f"Take a short break!\n\n"
                f"Time remaining: {remaining} seconds...\n\n"
                f"Press any key to continue to the experiment.\n"
                f"Otherwise, the experiment will start again immediately after the countdown."
            )
            instr_text.draw()
            update_onset = win.flip()
            update_times.append(update_onset)
            last_update_time = current_time
        
        keys = event.getKeys()
        if keys:
            rt = elapsed
            break_end_reason = "keypress"
            break
        if elapsed >= BREAK_DUR:
            rt = elapsed
            break_end_reason = "timeout"
            break
    
    break_end_time = core.getTime()
    
    record = {
        "break_start_time": break_start_time,
        "break_end_time": break_end_time,
        "break_display_onset": break_display_onset,
        "break_duration_actual": break_end_time - break_start_time,
        "break_duration_planned": BREAK_DUR,
        "break_end_reason": break_end_reason,
        "break_num_updates": len(update_times),
        "break_update_times": update_times,
        "rt": rt
    }
    
    # Add break trial data to ExperimentHandler
    this_exp.addData("trial_type", "break")
    this_exp.addData("break_duration", rt)
    this_exp.addData("break_completed", rt < BREAK_DUR)  # True if ended early with keypress
    this_exp.addData("break_start_time", break_start_time)
    this_exp.addData("break_end_time", break_end_time)
    this_exp.addData("break_end_reason", break_end_reason)
    this_exp.nextEntry()
    
    return record  # return time spent on break, in case needed for logging
    

def present_trial(img_name, img_cache, duration, isi, trial_num, extra_info=None):
    """Show one image with fixation and log timing; merged stim+ISI loop."""
    # Ensure the image exists in cache
    if img_name not in img_cache:
        logging.error(f"Image {img_name} not found in cache.")
        core.quit()
    stim = img_cache[img_name]

    # Prepare response text 
    response_text_stim.text = (
        f"{EXPO_BIGGER_KEY.upper()} if BIGGER than last object\n"
        f"{EXPO_SMALLER_KEY.upper()} if SMALLER than last object"
    )
    
    if args.debug:
        response_text_stim.text += f"\n[DEBUG MODE: Trial {trial_num}]"

    # Reset flash handler to normal for this trial
    _flash_handler.set_normal()
    trial_clock = core.Clock()
    responses = []

    # Draw initial frame and flip to get stimulus onset_time
    stim.draw()
    fixation.draw()
    response_text_stim.draw()
    stim_onset_time = win.flip()
    stim_offset_time = None
    isi_onset_time = None
    isi_offset_time = None

    total_duration = duration + isi
    # Frame-based loop covering both stimulus presentation and ISI
    while trial_clock.getTime() < total_duration:
        t = trial_clock.getTime()
        in_stim_phase = (t < duration)

        # Collect keys (timestamped relative to trial onset)
        keys = event.getKeys(timeStamped=trial_clock)
        for k, rt in keys:
            if k == "escape":
                win.close()
                core.quit()
            elif k == EXPO_BIGGER_KEY or k == EXPO_SMALLER_KEY:
                _flash_handler.start_flash()
                responses.append({"key": k, "rt": rt})

        # Update flash handler (frame bookkeeping)
        _flash_handler.update()

        # Draw appropriate content and capture timing transitions
        if in_stim_phase:
            stim.draw()
        else:
            # Transitioning to ISI - record stimulus offset and ISI onset
            if stim_offset_time is None:
                stim_offset_time = core.getTime()
            if isi_onset_time is None:
                isi_onset_time = core.getTime()
        # Always draw fixation and response text
        fixation.draw()
        response_text_stim.draw()
        win.flip()

    # Record final ISI offset time
    isi_offset_time = core.getTime()

    # Calculate durations
    actual_stim_duration = stim_offset_time - stim_onset_time if stim_offset_time else duration
    actual_isi_duration = isi_offset_time - isi_onset_time if isi_onset_time else 0
    total_trial_duration = isi_offset_time - stim_onset_time

    # Post-trial record creation with comprehensive timing
    record = {
        "trial_num": trial_num,
        "image": img_name,
        "stim_onset_time": stim_onset_time,
        "stim_offset_time": stim_offset_time,
        "stim_duration_planned": duration,
        "stim_duration_actual": actual_stim_duration,
        "isi_onset_time": isi_onset_time,
        "isi_offset_time": isi_offset_time,
        "isi_duration_planned": isi,
        "isi_duration_actual": actual_isi_duration,
        "trial_duration_total": total_trial_duration,
        "responses": responses,
        # Keep legacy fields for backward compatibility
        "onset_time": stim_onset_time,
        "duration": duration,
    }
    if extra_info:
        record.update(extra_info)

    # Compute response summary
    if responses:
        record["num_responses"] = len(responses)
        record["first_rt"] = responses[0]["rt"]
        record["last_rt"] = responses[-1]["rt"]
        record["last_key"] = responses[-1]["key"]
        record["mean_rt"] = float(np.mean([r["rt"] for r in responses]))
        correct_key = get_correct_key(extra_info)
        if correct_key and correct_key != "none":
            record["acc"] = int(responses and responses[-1]["key"] == correct_key)
            record["rt"] = responses[-1]["rt"] if responses else None
        else:
            record["acc"] = None
            record["rt"] = None
    else:
        record["num_responses"] = 0
        record["first_rt"] = None
        record["last_rt"] = None
        record["last_key"] = None
        record["mean_rt"] = None
        correct_key = get_correct_key(extra_info)
        if correct_key and correct_key != "none":
            record["acc"] = 0
        else:
            record["acc"] = None
        record["rt"] = None

    # Add trial-level data to ExperimentHandler
    for key, value in record.items():
        if key != "responses":  # Skip nested response data for main logging
            this_exp.addData(key, value)
    
    # Add detailed response data if needed
    if responses:
        for i, resp in enumerate(responses):
            this_exp.addData(f"response_{i+1}_key", resp["key"])
            this_exp.addData(f"response_{i+1}_rt", resp["rt"])
    
    this_exp.nextEntry()

    return record

def calculate_accuracy(trial_data):
    """Calculate accuracy from trial data."""
    if isinstance(trial_data, list):
        trial_data = pd.DataFrame(trial_data)

    if 'acc' not in trial_data.columns:
        return np.nan, "Accuracy: N/A (no 'acc' column)"

    valid_acc = trial_data['acc'].dropna()
    accuracy = valid_acc.mean() if not valid_acc.empty else np.nan
    missed = int(trial_data['last_key'].isna().sum()) if 'last_key' in trial_data.columns else 0
    total = len(trial_data)
    valid_trials = len(valid_acc)
    summary = (f"Accuracy: {accuracy:.1%} ({valid_trials} trials), Missed: {missed}/{total}")
    return accuracy, summary

def run_stream(csv_path, label="stream"):
    """Run a continuous stream of images defined in a CSV."""
    if not os.path.exists(csv_path):
        logging.warning(f"{csv_path} not found, skipping {label}.")
        return []
    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        logging.error(f"{csv_path} must contain an 'image' column.")
        core.quit()

    practice_flag = (csv_path == practice_exposure_csv or csv_path == practice_test_csv)
    cache = preload_images(df["image"], practice=practice_flag)

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        img_name = row["image"]
        trial_task = row["task"]
        extra = {c: row[c] for c in df.columns if c != "image"}
        if trial_task == "break":      
            record = break_trial()
            record["phase"] = "break"
        else:
            record = present_trial(img_name, cache, IMG_DUR, ISI_DUR, i + 1, extra)
            record["phase"] = label
        records.append(record)
    return records

def run_test(test_csv, label="test", practice=False):
    """Sequential test phase that can handle both practice and real test trials."""
    if not os.path.exists(test_csv):
        logging.info(f"No {label} phase found. Skipping.")
        return []
    df = pd.read_csv(test_csv)

    # Required columns check
    required_cols = {"seq1_stim1", "seq1_stim2", "seq2_stim1", "seq2_stim2", "correct_seq"}
    if not required_cols.issubset(df.columns):
        logging.error(f"{test_csv} must have columns: {required_cols}. Found: {list(df.columns)}")
        core.quit()

    # Gather images for preloading
    image_cols = ["seq1_stim1", "seq1_stim2", "seq2_stim1", "seq2_stim2"]
    all_imgs = df[image_cols[0]].copy()

    for col in image_cols[1:]:
        all_imgs = pd.concat([all_imgs, df[col]])

    # Handle practice vs. real test image loading
    if practice:
        # Practice images have full paths in CSV
        unique_imgs = set(all_imgs.dropna())
        cache = {}
        for img in unique_imgs:
            if not os.path.exists(img):
                logging.error(f"Missing practice test image: {img}")
                core.quit()
            cache[img] = visual.ImageStim(win, image=img, size=img_sz)
    else:
        # Real test images use preload_images function
        cache = preload_images(all_imgs)

    results = []

    # For each test row, present sequence 1, pause, sequence 2, then collect response
    for t, row in df.iterrows():
        result = present_test_trial(row, t, cache, label, practice)
        results.append(result)
        
    return results

def present_test_trial(row, trial_num, img_cache, label, practice=False):
    """Present a test trial with comprehensive timing recording."""
    
    # Initialize timing variables
    timing_log = {}
    trial_start_time = core.getTime()
    
    # Sequence 1: first image
    img_cache[row["seq1_stim1"]].draw()
    fixation.draw()
    seq1_img1_onset = win.flip()
    timing_log["seq1_img1_onset"] = seq1_img1_onset
    core.wait(IMG_DUR)
    seq1_img1_offset = core.getTime()
    timing_log["seq1_img1_offset"] = seq1_img1_offset
    timing_log["seq1_img1_duration"] = seq1_img1_offset - seq1_img1_onset

    # ISI after seq1_img1
    fixation.draw()
    isi1_onset = win.flip()
    timing_log["isi1_onset"] = isi1_onset
    core.wait(ISI_DUR)
    isi1_offset = core.getTime()
    timing_log["isi1_offset"] = isi1_offset
    timing_log["isi1_duration"] = isi1_offset - isi1_onset

    # Sequence 1: second image
    img_cache[row["seq1_stim2"]].draw()
    fixation.draw()
    seq1_img2_onset = win.flip()
    timing_log["seq1_img2_onset"] = seq1_img2_onset
    core.wait(IMG_DUR)
    seq1_img2_offset = core.getTime()
    timing_log["seq1_img2_offset"] = seq1_img2_offset
    timing_log["seq1_img2_duration"] = seq1_img2_offset - seq1_img2_onset

    # Pause between sequences
    fixation.draw()
    inter_seq_pause_onset = win.flip()
    timing_log["inter_seq_pause_onset"] = inter_seq_pause_onset
    core.wait(PAUSE_DUR)
    inter_seq_pause_offset = core.getTime()
    timing_log["inter_seq_pause_offset"] = inter_seq_pause_offset
    timing_log["inter_seq_pause_duration"] = inter_seq_pause_offset - inter_seq_pause_onset

    # Sequence 2: first image
    img_cache[row["seq2_stim1"]].draw()
    fixation.draw()
    seq2_img1_onset = win.flip()
    timing_log["seq2_img1_onset"] = seq2_img1_onset
    core.wait(IMG_DUR)
    seq2_img1_offset = core.getTime()
    timing_log["seq2_img1_offset"] = seq2_img1_offset
    timing_log["seq2_img1_duration"] = seq2_img1_offset - seq2_img1_onset

    # ISI after seq2_img1
    fixation.draw()
    isi2_onset = win.flip()
    timing_log["isi2_onset"] = isi2_onset
    core.wait(ISI_DUR)
    isi2_offset = core.getTime()
    timing_log["isi2_offset"] = isi2_offset
    timing_log["isi2_duration"] = isi2_offset - isi2_onset

    # Sequence 2: second image
    img_cache[row["seq2_stim2"]].draw()
    fixation.draw()
    seq2_img2_onset = win.flip()
    timing_log["seq2_img2_onset"] = seq2_img2_onset
    core.wait(IMG_DUR)
    seq2_img2_offset = core.getTime()
    timing_log["seq2_img2_offset"] = seq2_img2_offset
    timing_log["seq2_img2_duration"] = seq2_img2_offset - seq2_img2_onset

    # Response prompt
    response_prompt = TEST_RESP_PROMPT.format(
        seq1_key=TEST_SEQ1_KEY.upper(),
        seq2_key=TEST_SEQ2_KEY.upper()
    )
    response_text = visual.TextStim(win, text=response_prompt, color="white", height=small_txt_sz, pos=(0, 0))
    response_text.draw()
    response_onset = win.flip()
    timing_log["response_onset"] = response_onset

    timer = core.Clock()
    key, rt = event.waitKeys(keyList=[TEST_SEQ1_KEY, TEST_SEQ2_KEY, "escape"], timeStamped=timer)[0]
    if key == "escape":
        win.close()
        core.quit()
    
    response_offset = core.getTime()
    timing_log["response_offset"] = response_offset
    timing_log["response_duration"] = response_offset - response_onset

    correct_key = TEST_SEQ1_KEY if row["correct_seq"] == "seq1" else TEST_SEQ2_KEY
    choice_seq = "seq1" if key == TEST_SEQ1_KEY else "seq2"
    choice_key = key
    correct = (choice_seq == row["correct_seq"])

  
    fixation.draw()
    post_isi_onset = win.flip()
    timing_log["post_isi_onset"] = post_isi_onset
    core.wait(0.5)
    post_isi_offset = core.getTime()
    timing_log["post_isi_offset"] = post_isi_offset
    timing_log["post_isi_duration"] = post_isi_offset - post_isi_onset

    # Calculate total trial duration
    trial_end_time = core.getTime()
    timing_log["trial_start_time"] = trial_start_time
    timing_log["trial_end_time"] = trial_end_time
    timing_log["trial_total_duration"] = trial_end_time - trial_start_time

    # Compile comprehensive result record
    result = {
        "trial_num": trial_num + 1,
        "pair_type": row.get("pair_type", "unknown"),
        "test_type": row.get("test_type", "unknown"),
        "block": row.get("block", "unknown"),
        "targ_grp_num": row.get("targ_grp_num", "unknown"),
        "target_idx": row.get("target_idx", "unknown"),
        "foil_idx": row.get("foil_idx", "unknown"),
        "seq1_stim1": row["seq1_stim1"],
        "seq1_stim2": row["seq1_stim2"],
        "seq2_stim1": row["seq2_stim1"],
        "seq2_stim2": row["seq2_stim2"],
        "correct_seq": row["correct_seq"],
        "correct_key": correct_key,
        "choice_seq": choice_seq,
        "choice_key": choice_key,
        "accuracy": correct,
        "rt": rt,
        # Add all timing information
        **timing_log
    }
    
    # Add test trial data to ExperimentHandler
    for key_name, value in result.items():
        this_exp.addData(key_name, value)
    this_exp.addData("phase", label)
    this_exp.nextEntry()
    
    return result

# ------------------------
# Experiment Flow
# ------------------------
all_practice_data = []
practice_test_data = []
accuracy = 0.0
summary = ""
practice_attempt = 0

# Initialize data lists
main_data = []
test_data = []

if RUN_EXPOSURE:
    # Run exposure practice before main exposure phase
    if RUN_PRACTICE:
        # Welcome text
        show_instructions(EXPO_1_INSTRUCT.format(
            bigger_key=EXPO_BIGGER_KEY.upper(),
            smaller_key=EXPO_SMALLER_KEY.upper()
        ))
        # Practice instructions
        show_instructions(EXPO_2_PRACTICE)
        show_instructions(EXPO_KEY_REMINDER.format(
            bigger_key=EXPO_BIGGER_KEY.upper(),
            smaller_key=EXPO_SMALLER_KEY.upper()
        ))
        practice_attempt = 1
        max_practice_attempts = 20
        all_practice_data = []
        min_accuracy = 0.0 if args.debug else 1.0 # require 100% accuracy to pass practice unless in debug mode

        while practice_attempt <= max_practice_attempts:
            logging.info(f"Practice attempt {practice_attempt}")
            practice_data = run_stream(practice_exposure_csv, label=f"practice_attempt_{practice_attempt}")
            all_practice_data.extend(practice_data)

            accuracy, summary = calculate_accuracy(practice_data)
            logging.info(f"Practice accuracy: {accuracy:.1%} - {summary}")

            if accuracy >= min_accuracy:
                show_instructions(EXPO_4_PRACT_DONE.format(accuracy=accuracy))
                break
            else:
                if practice_attempt >= max_practice_attempts:
                    show_instructions(EXPO_4_PRACT_DONE.format(
                        accuracy=accuracy
                    ))
                else:
                    show_instructions(EXPO_3_PRACT_RETRY.format(
                        accuracy=accuracy,
                        bigger_key=EXPO_BIGGER_KEY.upper(),
                        smaller_key=EXPO_SMALLER_KEY.upper()
                    ))
                practice_attempt += 1

        show_instructions(EXPO_MAIN_TEXT)
        show_instructions(EXPO_KEY_REMINDER.format(
            bigger_key=EXPO_BIGGER_KEY.upper(),
            smaller_key=EXPO_SMALLER_KEY.upper()
        ))
    else:
        show_instructions(EXPO_NO_PRACT.format(
            bigger_key=EXPO_BIGGER_KEY.upper(),
            smaller_key=EXPO_SMALLER_KEY.upper()
        ))
    main_data = run_stream(exposure_trials_csv, label="exposure")

if RUN_TESTS:
    # Run test practice phase before main test
    if RUN_PRACTICE:
        show_instructions(TEST_1_INSTRUCT)
        show_instructions(TEST_2_INSTRUCT.format(
            seq1_key=TEST_SEQ1_KEY.upper(),
            seq2_key=TEST_SEQ2_KEY.upper()
        ))
        show_instructions(TEST_3_PRACTICE)
        show_instructions(TEST_KEY_REMINDER.format(
            seq1_key=TEST_SEQ1_KEY.upper(),
            seq2_key=TEST_SEQ2_KEY.upper()
        ))

        # Practice test retry loop
        practice_test_attempt = 1
        max_practice_test_attempts = 10
        all_practice_test_data = []
        
        while practice_test_attempt <= max_practice_test_attempts:
            logging.info(f"Practice test attempt {practice_test_attempt}")
            current_practice_test_data = run_test(practice_test_csv, label=f"practice_test_attempt_{practice_test_attempt}", practice=True)
            all_practice_test_data.extend(current_practice_test_data)
            
            if current_practice_test_data:
                test_accuracy, test_summary = calculate_accuracy(current_practice_test_data)
                
                # Ask if they want to continue or retry
                while True:
                    retry_text = TEST_6_PRACT_RETRY.format(
                        seq1_key=TEST_SEQ1_KEY.upper(),
                        seq2_key=TEST_SEQ2_KEY.upper()
                    )
                    instr_text.text = retry_text
                    instr_text.draw()
                    win.flip()

                    keys = event.waitKeys(keyList=['c', 'r', 'escape'])
                    if keys[0] in ['escape']:
                        show_instructions(EXPERIMENT_CANCELLED_TEXT)
                        win.close()
                        core.quit()
                    elif keys[0] == 'c':
                        # Continue to main test
                        practice_test_data = all_practice_test_data
                        show_instructions(TEST_5_PRACT_CONT)
                        break
                    elif keys[0] == 'r':
                        # Retry practice test
                        if practice_test_attempt >= max_practice_test_attempts:
                            show_instructions(TEST_7_PRACT_MAX.format(
                                max_attempts=max_practice_test_attempts
                            ))
                            practice_test_data = all_practice_test_data
                            break
                        else:
                            show_instructions(TEST_6_PRACT_RETRY.format(
                                seq1_key=TEST_SEQ1_KEY.upper(),
                                seq2_key=TEST_SEQ2_KEY.upper()
                            ))
                            practice_test_attempt += 1
                            break
                
                # Break out of retry loop if continuing to main test
                if keys[0] == 'c' or practice_test_attempt > max_practice_test_attempts:
                    break
            else:
                show_instructions(TEST_5_PRACT_CONT)
                practice_test_data = []
                break
        # If practice was run, show transition message
        show_instructions(TEST_MAIN_TEXT)
        show_instructions(TEST_KEY_REMINDER.format(
            seq1_key=TEST_SEQ1_KEY.upper(),
            seq2_key=TEST_SEQ2_KEY.upper()
        ))
        
    else:
        # No practice, initialize empty practice test data
        practice_test_data = []
        show_instructions(TEST_NO_PRACTICE_TEXT.format(
            seq1_key=TEST_SEQ1_KEY.upper(),
            seq2_key=TEST_SEQ2_KEY.upper()
        ))
    
    test_data = run_test(test_trials_csv, label="test", practice=False)

# ------------------------
# Save Data 
# ------------------------
all_data = all_practice_data + main_data
df_all  = pd.DataFrame(all_data)
test_df = pd.DataFrame(test_data)

practice_summary = {
    'subject': exp_info['subject'],
    'total_practice_attempts': practice_attempt,
    'achieved_100_percent': accuracy >= 1.0,
    'final_accuracy': accuracy,
    'summary': summary
}
practice_summary_df = pd.DataFrame([practice_summary])

practice_summary_df.to_csv(os.path.join(data_folder, f"{exp_info['file_prefix']}_practice-summary.csv"), index=False)
df_all.to_csv(os.path.join(data_folder, f"{exp_info['file_prefix']}_exposure-data.csv"), index=False)
if len(test_data) > 0:
    test_df.to_csv(os.path.join(data_folder, f"{exp_info['file_prefix']}_test-data.csv"), index=False)
if len(practice_test_data) > 0:
    practice_test_df = pd.DataFrame(practice_test_data)
    practice_test_df.to_csv(os.path.join(data_folder, f"{exp_info['file_prefix']}_practice-test-data.csv"), index=False)

# -----------------------------
# Wrap up
# -----------------------------
#saveData(results=results)
show_instructions(EXPERIMENT_COMPLETE_TEXT)
event.clearEvents()
win.close()
core.quit()

