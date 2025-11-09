# test_onscreen_text.py
#
# Test script for onscreen text display
# Uses same window variables as run_sl-obj-inference.py and displays text from onscreen_text.py
# -----------------------------------------------------------------------------------
# Run with: python test_onscreen_text.py --debug True
# or: python test_onscreen_text.py --debug False (for full screen test)

from psychopy import visual, core, event
import argparse
import sys
import os

# Add utils folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from onscreen_text import *

# -----------------------------
# Argument parsing
# -----------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1", "y")

parser = argparse.ArgumentParser(description="Test onscreen text display.")
parser.add_argument("--debug", type=str2bool, default=False, help="Run in debug mode (True/False)")
args = parser.parse_args()

# -----------------------------
# Configuration (matching run_sl-obj-inference.py)
# -----------------------------
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

# Fixation colors (matching main script)
FIX_LINE = "white"
FIX_FILL = "black"

# -----------------------------
# Window setup (matching main script)
# -----------------------------
win = visual.Window(
    size=WIN_SIZE, 
    fullscr=FULLSCREEN, 
    screen=SCREEN, 
    monitor=MONITOR,
    color=[0, 0, 0], 
    colorSpace='rgb', 
    units='pix'
)

# size calculations
win_size = win.size
win_w = win_size[0]
win_h = win_size[1]
wrap_wth = int(win_w * 0.65)    # 65% of window width for text wrapping
lg_txt = int(win_h * 0.038)     # 4% of window height for text size
sm_txt= int(lg_txt * 0.75)      # smaller text for exposure text
expo_txt_pos = -win_h * 0.25    # 25% down from center for response text
res = int(win_h * 0.4)          # 40% of window height for image size
img_sz = (res, res)             # image size


# Print size info
print(f"\n{'='*60}")
print(f"ONSCREEN TEXT TEST")
print(f"{'='*60}")
print(f"Mode: {'DEBUG' if args.debug else 'FULL SCREEN'}")
print(f"Window size: {win_size}")
print(f"Text wrap width: {wrap_wth:.1f} px")
print(f"Large text size: {lg_txt:.1f} px")
print(f"Small text size: {sm_txt:.1f} px")
print(f"Image size: {img_sz}")
print(f"Exposure text Y position: {expo_txt_pos:.1f} px")
print(f"{'='*60}\n")

# Fixation setup (matching main script)
radius = (res * 0.012) * bug          # 1% of image size for fixation radius
lineWidth = 1.75 * bug

# Create text stimuli (matching main script)
fixation = visual.Circle(
    win, 
    radius=radius, 
    lineWidth=lineWidth, 
    fillColor=FIX_FILL, 
    lineColor=FIX_LINE, 
    pos=(0, 0)
)
instr_text = visual.TextStim(
    win, 
    text="", 
    color="white", 
    height=lg_txt, 
    wrapWidth=wrap_wth,
)
response_text_stim = visual.TextStim(
    win, 
    text="", 
    color="white", 
    height=sm_txt, 
    pos=(0, expo_txt_pos)
)

# -----------------------------
# Test sequences
# -----------------------------

# Define keys for formatting
LARGER_KEY = "F"
SMALLER_KEY = "J"
SEQ1_KEY = "F"
SEQ2_KEY = "J"

def show_screen(main_text, response_text="", show_fixation=True, duration=None):
    """Display text with optional fixation and wait for keypress or duration."""
    instr_text.text = main_text
    response_text_stim.text = response_text
    
    # Draw elements
    instr_text.draw()
    if response_text:
        response_text_stim.draw()
    if show_fixation:
        fixation.draw()
    
    win.flip()
    
    # Wait for keypress or duration
    if duration:
        core.wait(duration)
    else:
        event.waitKeys()

print("\nStarting text display tests...")
print("Press any key to advance through screens.\n")

# ===========================
# SIZE INFO SCREEN (FIRST)
# ===========================

# Create temporary text stims at different sizes to show actual heights
large_text_demo = visual.TextStim(win, text="Large text size", color="white", height=lg_txt)
small_text_demo = visual.TextStim(win, text="Small text size", color="white", height=sm_txt)

# Main info text
instr_text.text = (
    f"window size:\t{win_size[0]} x {win_size[1]}\n"
    f"wrap width:\t{wrap_wth:.1f} px\n"
    f"image size:\t{img_sz[0]} x {img_sz[1]}\n"
    f"resp Y-pos:\t{expo_txt_pos:.1f} px\n"
    f"fix radius:\t{radius:.2f} px\n"
)
large_text_demo.pos = (0, -(win_size[1]/2)+300)
large_text_demo.text = f"Large text: {lg_txt:.1f} px"
small_text_demo.pos = (0, -(win_size[1]/2)+350)
small_text_demo.text = f"Small text: {sm_txt:.1f} px"

# Draw all elements
instr_text.draw()
large_text_demo.draw()
small_text_demo.draw()
win.flip()
event.waitKeys()

# ===========================
# IMAGE SIZE DEMO (SECOND)
# ===========================

# Load a sample image
sample_image_path = os.path.join(os.path.dirname(__file__), "practice", "stimuli", "images", "1-A.png")
if os.path.exists(sample_image_path):
    demo_image = visual.ImageStim(win, image=sample_image_path, size=img_sz)
    
    instr_text.text = (
        f"IMAGE DISPLAY TEST\n"
        f"Image resolution: {img_sz[0]} x {img_sz[1]} px"
    )
    # position text above image
    instr_text.pos = (0, img_sz[1]/2 + 50)
    # Position image center
    demo_image.pos = (0, 0)
    
    # Draw all elements
    instr_text.draw()
    demo_image.draw()
    fixation.draw()
    win.flip()
    event.waitKeys()
else:
    # If image not found, skip this screen
    print(f"Sample image not found at {sample_image_path}, skipping image demo.")

# reset instr_text position
instr_text.pos = (0, 0)
# ===========================
# EXPOSURE PHASE TEXTS
# ===========================

# # Test 1: Initial exposure instructions
# show_screen(
#     EXPO_1_INSTRUCT.format(larger_key=LARGER_KEY, smaller_key=SMALLER_KEY),
#     show_fixation=False
# )

# # Test 2: Practice introduction
# show_screen(
#     EXPO_2_PRACTICE,
#     show_fixation=False
# )

# # Test 3: Practice retry message
# show_screen(
#     EXPO_3_PRACT_RETRY.format(
#         accuracy=0.75,
#         larger_key=LARGER_KEY,
#         smaller_key=SMALLER_KEY
#     ),
#     show_fixation=False
# )

# # Test 4: Practice done
# show_screen(
#     EXPO_4_PRACT_DONE.format(accuracy=1.0),
#     show_fixation=False
# )

# # Test 5: Main exposure text
# show_screen(
#     EXPO_MAIN_TEXT,
#     show_fixation=False
# )

# # Test 6: Key reminder
# show_screen(
#     EXPO_KEY_REMINDER.format(larger_key=LARGER_KEY, smaller_key=SMALLER_KEY),
#     show_fixation=False
# )

# # Test 7: No practice version
# show_screen(
#     EXPO_NO_PRACT.format(larger_key=LARGER_KEY, smaller_key=SMALLER_KEY),
#     show_fixation=False
# )

# # Test 8: Exposure response prompt (with fixation)
# show_screen(
#     "",
#     response_text=EXPO_RESP_PROMPT.format(larger_key=LARGER_KEY, smaller_key=SMALLER_KEY),
#     show_fixation=True
# )

# ===========================
# TEST PHASE TEXTS
# ===========================

# Test 9: Test phase instruction 1
show_screen(
    TEST_1_INSTRUCT,
    show_fixation=False
)

# Test 10: Test phase instruction 2
show_screen(
    TEST_2_INSTRUCT.format(seq1_key=SEQ1_KEY, seq2_key=SEQ2_KEY),
    show_fixation=False
)

# Test 11: Test practice introduction
show_screen(
    TEST_3_PRACTICE,
    show_fixation=False
)

# Test 12: Practice done with options
show_screen(
    TEST_4_PRACT_DONE,
    show_fixation=False
)

# Test 13: Practice continue
show_screen(
    TEST_5_PRACT_CONT,
    show_fixation=False
)

# Test 14: Practice retry
show_screen(
    TEST_6_PRACT_RETRY.format(seq1_key=SEQ1_KEY, seq2_key=SEQ2_KEY),
    show_fixation=False
)

# Test 15: Max attempts reached
show_screen(
    TEST_7_PRACT_MAX.format(max_attempts=5),
    show_fixation=False
)

# Test 16: No practice test text
show_screen(
    TEST_NO_PRACTICE_TEXT.format(seq1_key=SEQ1_KEY, seq2_key=SEQ2_KEY),
    show_fixation=False
)

# Test 17: Main test text
show_screen(
    TEST_MAIN_TEXT,
    show_fixation=False
)

# Test 18: Test key reminder
show_screen(
    TEST_KEY_REMINDER.format(seq1_key=SEQ1_KEY, seq2_key=SEQ2_KEY),
    show_fixation=False
)

# Test 19: Test response prompt (with fixation)
show_screen(
    TEST_RESP_PROMPT.format(seq1_key=SEQ1_KEY, seq2_key=SEQ2_KEY),
    show_fixation=True,
    duration=3
)

# ===========================
# GENERAL MESSAGES
# ===========================

# Test 20: Experiment cancelled
show_screen(
    EXPERIMENT_CANCELLED_TEXT,
    show_fixation=False
)

# Test 21: Experiment complete (final screen)
show_screen(
    EXPERIMENT_COMPLETE_TEXT,
    show_fixation=False
)

# Summary screen
show_screen(
    f"All Text Variables Displayed!\n\n"
    f"Total screens shown: 21\n\n"
    f"Window configuration:\n"
    f"Size: {win_size}\n"
    f"Fullscreen: {FULLSCREEN}\n"
    f"Monitor: {MONITOR}\n"
    f"Large text size: {lg_txt:.1f} px\n"
    f"Small text size: {sm_txt:.1f} px\n\n"
    f"Press any key to exit."
)

# Cleanup
print("\n" + "="*60)
print("All text variables from onscreen_text.py have been displayed!")
print("="*60)
print("\nTest complete!")
win.close()
core.quit()
