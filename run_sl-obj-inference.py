# run_sl-obj-inference.py

# --- Import packages ---
from psychopy import __version__ as psychopy_ver
from psychopy import visual, core, event, data, gui

import numpy as np
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import pandas as pd
import os

# -----------------------------
# set debug mode
# -----------------------------
DEBUG = True
if DEBUG:
    fullscr = False
    monitor = "debugMonitor"
    screen = 0
else:
    fullscr = True
    monitor = "testMonitor"
    screen = 1

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
    'psychopyVersion|hid': psychopy_ver
    }
dlg = gui.DlgFromDict(exp_info, title="Intermixed SL Object Inference") 
if not dlg.OK:
    core.quit()

# create data folder if needed
if not os.path.isdir("data"):
    os.makedirs("data")

# define output filename
filename = u'data/sub-%s_%s_expo-%s_test-%s_%s' % (exp_info['subject'], exp_info['exp_name|hid'], exp_info['exposure'], exp_info['test'].replace("-", ""), exp_info['date|hid'])

# setup experiment handler
this_exp = data.ExperimentHandler(
    name="Inmix-SL-Obj-Inference",
    extraInfo=exp_info,
    dataFileName=filename
)

# -----------------------------
# Window setup
# -----------------------------
win = visual.Window(size=[1920, 1080], 
                    fullscr=fullscr, screen=screen, monitor=monitor, 
                    color=[0, 0, 0], colorSpace='rgb', units='height'
                    )


# -----------------------------
# Stimuli & trial setup
# -----------------------------

# Create a text stimulus
message = visual.TextStim(win, text="Press Space to continue", color='white')

# Create a clock to manage timing
timer = core.Clock()

# Display the message
message.draw()
win.flip() # Update the window to show the stimulus

# Wait for a key press
event.waitKeys(keyList=['space'])

# -----------------------------
# Wrap up
# -----------------------------
event.clearEvents()
win.close()
core.quit()
