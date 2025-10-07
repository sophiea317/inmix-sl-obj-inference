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
DEBUG = False
practice = False
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
# experiment parameters
# ------------------------
data_folder = "data"
stim_folder = "assets"
practice_folder = os.path.join(stim_folder, "practice")
num_stim = 24
num_grps = 6
num_reps = 40
prob_1back = 0.1

# parameters for stimulus presentation
img_dur = 0.5
isi_dur = 0.5
fbck_dur = 0.15
img_size = (300, 300)
test_l_key = "f"
test_r_key = "j"
response_window_ms = 2000  # Response window for 1-back detection (milliseconds)

fix_line = "black"
fix_fill = "white"

# -----------------------------
# experiment setup
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
exp_info['file_prefix'] = (
    f"sub-{exp_info['subject']}_{exp_info['exp_name|hid']}_"
    f"test-{exp_info['test'].replace('-', '')}_{exp_info['date|hid']}"
)

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

# window variables
fixation = visual.Circle(win, radius=5, fillColor=fix_fill, lineColor=fix_line, pos=(0,0))
instr_text = visual.TextStim(win, text="", color="white", height=26, wrapWidth=1000)

# Frame rate info for precise timing
refresh_rate = win.getActualFrameRate()
if refresh_rate is None:
    refresh_rate = 60.0  # assume 60Hz if can't measure
print(f"Monitor refresh rate: {refresh_rate:.1f} Hz")

# ------------------------
# Helper Classes and Functions
# ------------------------

class FixationFlashHandler:
    """Handles frame-based fixation flashing without blocking."""
    
    def __init__(self, fixation_stim, normal_fill, normal_line, flash_fill, flash_line, flash_duration_sec, refresh_rate):
        self.fixation = fixation_stim
        self.normal_fill = normal_fill
        self.normal_line = normal_line
        self.flash_fill = flash_fill
        self.flash_line = flash_line
        self.flash_duration_frames = int(flash_duration_sec * refresh_rate)
        
        # State variables
        self.state = "normal"  # "normal", "flashing"
        self.flash_frame_counter = 0
        
        # Set initial state
        self.set_normal()
    
    def set_normal(self):
        """Set fixation to normal appearance."""
        self.fixation.fillColor = self.normal_fill
        self.fixation.lineColor = self.normal_line
        self.state = "normal"
        self.flash_frame_counter = 0
    
    def start_flash(self):
        """Start a fixation flash."""
        self.state = "flashing"
        self.flash_frame_counter = 0
        self.fixation.fillColor = self.flash_fill
        self.fixation.lineColor = self.flash_line
    
    def update(self):
        """Update flash state - call once per frame."""
        if self.state == "flashing":
            self.flash_frame_counter += 1
            if self.flash_frame_counter >= self.flash_duration_frames:
                self.set_normal()
    
    def is_flashing(self):
        """Check if currently flashing."""
        return self.state == "flashing"
def show_instructions(text):
    """Show instructions."""
    instr_text.text = text
    instr_text.draw()
    win.flip()
    event.waitKeys()

def preload_images(images):
    cache = {}
    for img in set(images):
        path = os.path.join(stim_folder, img)
        if not os.path.exists(path):
            logging.error(f"Missing image: {path}")
            core.quit()
        cache[img] = visual.ImageStim(win, image=path, size=img_size)
    return cache

def present_trial(img_name, img_cache, duration, isi, trial_num, extra_info=None):
    """Show one image with fixation and log timing + allow multiple responses using frame-based approach."""
    stim = img_cache[img_name]
    
    # Initialize flash handler
    flash_handler = FixationFlashHandler(
        fixation_stim=fixation,
        normal_fill=fix_fill,
        normal_line=fix_line,
        flash_fill=fix_line,  # inverted colors for flash
        flash_line=fix_fill,
        flash_duration_sec=fbck_dur,
        refresh_rate=refresh_rate
    )
    
    # Initialize timing variables
    trial_clock = core.Clock()
    responses = []  # store all keypresses (time + key)
    
    # Draw initial frame
    stim.draw()
    fixation.draw()
    onset_time = win.flip()
    
    # Main stimulus presentation loop (frame-based)
    while trial_clock.getTime() < duration:
        # Check for keypresses
        keys = event.getKeys(timeStamped=trial_clock)
        for k, t in keys:
            if k == "escape":
                win.close()
                core.quit()
            elif k == "space":
                # Start fixation flash
                flash_handler.start_flash()
                # Record keypress
                responses.append({"key": k, "rt": t})
        
        # Update flash state
        flash_handler.update()
        
        # Draw current frame
        stim.draw()
        fixation.draw()
        win.flip()
    
    # ISI period (frame-based)
    isi_clock = core.Clock()
    
    while isi_clock.getTime() < isi:
        # Continue checking for keypresses during ISI
        keys = event.getKeys(timeStamped=trial_clock)  # Note: still using trial_clock for RT relative to trial onset
        for k, t in keys:
            if k == "escape":
                win.close()
                core.quit()
            elif k == "space":
                # Start fixation flash during ISI
                flash_handler.start_flash()
                # Record keypress
                responses.append({"key": k, "rt": t})
        
        # Update flash state
        flash_handler.update()
        
        # Draw ISI frame (fixation only)
        fixation.draw()
        win.flip()
    
    # Create trial record
    record = {
        "trial_num": trial_num,
        "image": img_name,
        "onset_time": onset_time,
        "duration": duration,
        "responses": responses,  # list of all keypresses this trial
    }
    
    if extra_info:
        record.update(extra_info)
    
    # For convenience, compute quick response summary
    if responses:
        record["num_responses"] = len(responses)
        record["first_rt"] = responses[0]["rt"]
        record["last_rt"] = responses[-1]["rt"]
        record["mean_rt"] = np.mean([r["rt"] for r in responses])
    else:
        record["num_responses"] = 0
        record["first_rt"] = None
        record["last_rt"] = None
        record["mean_rt"] = None
    
    return record


def calculate_1back_accuracy(trial_data, response_window_ms=2000):
    """Calculate 1-back detection accuracy with proper post-onset response window."""
    if not trial_data:
        return 0.0, 0, 0, "No trials found"
    
    df = pd.DataFrame(trial_data)
    
    # Check if we have the required columns
    if 'is_1back' not in df.columns:
        return 0.0, 0, 0, "Missing 'is_1back' column in data"
    
    # Convert response window to seconds
    response_window_sec = response_window_ms / 1000.0
    
    # Sort trials by trial number to process in sequence
    df_sorted = df.sort_values('trial_num').reset_index(drop=True)
    
    n_oneback_trials = len(df_sorted[df_sorted['is_1back'] == 1])
    n_normal_trials = len(df_sorted[df_sorted['is_1back'] == 0])
    
    if n_oneback_trials == 0:
        return 0.0, 0, 0, "No 1-back trials found"
    
    # Track which 1-back trials have been detected
    oneback_detected = set()
    false_alarms = []
    late_responses = 0
    
    # Get trial duration for timing calculations
    trial_duration = img_dur + isi_dur
    
    # Process each trial and look for valid responses to 1-back trials
    for i, trial in df_sorted.iterrows():
        if trial['is_1back'] == 1:  # This is a 1-back trial
            # Look for responses from this trial and subsequent trials within the response window
            trial_detected = False
            
            # Check responses during the 1-back trial itself
            if trial['num_responses'] > 0 and trial['responses']:
                for resp in trial['responses']:
                    # Response must be after trial onset (rt > 0) and within window
                    if 0 < resp['rt'] <= response_window_sec:
                        trial_detected = True
                        break
            
            # Check responses in subsequent trials within the response window
            if not trial_detected:
                for j in range(i + 1, len(df_sorted)):
                    later_trial = df_sorted.iloc[j]
                    if later_trial['num_responses'] > 0 and later_trial['responses']:
                        # Calculate when this later trial started relative to the 1-back trial
                        trials_later = j - i
                        later_trial_start_time = trials_later * trial_duration
                        
                        # If this later trial started after our response window, stop looking
                        if later_trial_start_time > response_window_sec:
                            break
                            
                        # Check each response in the later trial
                        for resp in later_trial['responses']:
                            # Time from 1-back onset to this response
                            response_time_from_oneback = later_trial_start_time + resp['rt']
                            
                            # Valid if within response window after 1-back onset
                            if 0 < response_time_from_oneback <= response_window_sec:
                                trial_detected = True
                                break
                        
                        if trial_detected:
                            break
            
            if trial_detected:
                oneback_detected.add(i)
    
    # Now identify false alarms - responses that weren't attributed to any 1-back
    for i, trial in df_sorted.iterrows():
        if trial['num_responses'] > 0 and trial['responses']:
            for resp in trial['responses']:
                response_attributed = False
                
                if trial['is_1back'] == 1:
                    # Response during 1-back trial
                    if 0 < resp['rt'] <= response_window_sec and i in oneback_detected:
                        response_attributed = True
                    elif resp['rt'] > response_window_sec:
                        # Late response on 1-back trial
                        late_responses += 1
                        continue
                else:
                    # Response during normal trial - check if it can be attributed to a previous 1-back
                    for j in range(max(0, i-10), i):  # Look back at recent trials
                        prev_trial = df_sorted.iloc[j]
                        if prev_trial['is_1back'] == 1:
                            # Calculate time from previous 1-back onset to this response
                            trials_later = i - j
                            current_trial_start_time = trials_later * trial_duration
                            response_time_from_oneback = current_trial_start_time + resp['rt']
                            
                            # Check if this response is within the window for that 1-back
                            if 0 < response_time_from_oneback <= response_window_sec and j in oneback_detected:
                                response_attributed = True
                                break
                
                # If response wasn't attributed to any 1-back, it's a false alarm
                if not response_attributed:
                    false_alarms.append((i, resp['rt']))
    
    # Calculate final metrics
    hits = len(oneback_detected)
    missed_onebacks = n_oneback_trials - hits
    false_alarms_count = len(false_alarms) + late_responses
    
    # Calculate accuracy
    correct_detections = hits
    correct_rejections = n_normal_trials - len([fa for fa in false_alarms if df_sorted.iloc[fa[0]]['is_1back'] == 0])
    total_trials = n_oneback_trials + n_normal_trials
    accuracy = (correct_detections + correct_rejections) / total_trials
    
    # Calculate rates
    hit_rate = hits / n_oneback_trials if n_oneback_trials > 0 else 0
    fa_rate = false_alarms_count / (n_normal_trials + n_oneback_trials) if (n_normal_trials + n_oneback_trials) > 0 else 0
    
    summary = (f"Hits: {hits}/{n_oneback_trials} ({hit_rate:.1%}), "
              f"Misses: {missed_onebacks}, Late: {late_responses}, "
              f"FA: {false_alarms_count} ({fa_rate:.1%})")
    
    return accuracy, hits, false_alarms_count, summary

def run_stream(csv_path, label="stream"):
    """Run a continuous stream of images defined in a CSV."""
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

    left_stim = visual.ImageStim(win, size=img_size, pos=(-300,0))
    right_stim = visual.ImageStim(win, size=img_size, pos=(300,0))

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
if practice:
    show_instructions(
        "Welcome!\n\nYou will see a sequence of images.\nKeep your eyes on the central dot at all times.\n\nPress SPACE whenever you see an image repeat.\n\nYou must achieve 100% accuracy on practice trials to continue.\n\nPress any key to begin practice."
    )

    # Practice loop - require 100% accuracy
    practice_attempt = 1
    max_practice_attempts = 10  # Prevent infinite loops
    all_practice_data = []

    while practice_attempt <= max_practice_attempts:
        print(f"Practice attempt {practice_attempt}")
        
        # Run practice trials
        practice_data = run_stream(practice_exposure_csv, label=f"practice_attempt_{practice_attempt}")
        all_practice_data.extend(practice_data)
        
        # Calculate accuracy
        accuracy, hits, false_alarms, summary = calculate_1back_accuracy(practice_data, response_window_ms)
        
        print(f"Practice accuracy: {accuracy:.1%} - {summary}")
        
        if accuracy >= 1.0:  # 100% accuracy required
            show_instructions(
                f"Excellent! You achieved 100% accuracy!\n\n{summary}\n\nYou're ready for the main experiment.\n\nPress any key to continue."
            )
            break
        else:
            if practice_attempt >= max_practice_attempts:
                show_instructions(
                    f"Practice complete after {max_practice_attempts} attempts.\n\nResponse accuracy: {accuracy:.1%}\n\nYou will now proceed to the main experiment.\n\nPress any key to continue."
                )
            else:
                show_instructions(
                    f"Practice accuracy: {accuracy:.1%}\nYou need 100% accuracy to continue.\nLet's try the practice again.\n\nRemember:\n- Press SPACE only AFTER you see an image repeat\n\nPress any key to retry."
                )
            practice_attempt += 1

    show_instructions(
        "Now the real experiment will begin.\nAgain, keep your eyes on the central dot.\n\nPress SPACE whenever an image repeats.\n\nPress any key to start."
    )
else: 
    show_instructions(
        "Welcome to the experiment!\n\nYou will see a sequence of images.\nKeep your eyes on the central dot at all times.\n\nPress SPACE whenever you see an image repeat.\n\nPress any key to begin."
    )

main_data = run_stream(exposure_trials_csv, label="exposure")

# Optional test
test_data = run_test()

# ------------------------
# Save Data
# ------------------------
all_data = all_practice_data + main_data
df_all = pd.DataFrame(all_data)
test_df = pd.DataFrame(test_data)

# Also save practice performance summary
practice_summary = {
    'subject': exp_info['subject'],
    'total_practice_attempts': practice_attempt,
    'achieved_100_percent': accuracy >= 1.0,
    'final_accuracy': accuracy,
    'final_hits': hits,
    'final_false_alarms': false_alarms,
    'summary': summary
}
practice_summary_df = pd.DataFrame([practice_summary])
practice_summary_df.to_csv(os.path.join(data_folder, f"{exp_info['file_prefix']}_practice-summary.csv"), index=False)

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
