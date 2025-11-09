# onscreen_text.py
#
# Text variables for all instruction screens in the Inmix SL Object Inference experiment
# These texts are used throughout the experiment to provide consistent messaging


#################   
# EXPOSURE TEXT
#################
EXPO_1_INSTRUCT = (
    "Welcome!\n\n"
    "You will see images of objects in a stream. Your task is to decide whether each object is bigger or smaller than the one before it.\n"
    "All objects are shown as the same size on the screen. Base your answer on their real-life size.\n\n"
    "Press '{bigger_key}' if the current object is bigger than the previous one.\n"
    "Press '{smaller_key}' if the current object is smaller than the previous one.\n\n"
)

EXPO_2_PRACTICE = (
    "Before you begin, you will complete a few practice trials to get used to the format and speed of the task.\n\n"
    "You need to get a 100% accuracy on the practice trials to proceed to the actual experiment. The images are on screen for a very short time, so be sure to pay close attention and respond as quickly and accurately as possible.\n\n"
    "Practice trials will use simple object icons colored black, but the actual experiment will use images of real-world objects.\n\n"
    "Any questions?"
)

EXPO_3_PRACT_RETRY = (
    "Accuracy: {accuracy:.1%}\n"
    "You need 100% accuracy to continue.\n"
    "Let's try the practice again.\n\n"
    "Remember:\n"
    "- Press '{bigger_key}' if bigger than previous object\n"
    "- Press '{smaller_key}' if smaller than previous object\n\n"
    "Press the spacebar to retry."
)

EXPO_4_PRACT_DONE = (
    "You got a {accuracy:.1%} accuracy!\n\n"
    "Please wait for the experimenter to provide further instructions..."
)

EXPO_MAIN_TEXT = (
    "Now that you have completed the practice trials, you will proceed to the actual experiment.\n\n"
    "Everything will be the same as in practice, but you will see images of real-world objects instead of simple icons.\n"
    "The object images are shown as the same size on the screen, but remember to base your answer on their real-life size.\n\n"
    "You will also have breaks throughout the experiment where you can rest your eyes. After the end of each break, the first object shown will not have a previous object to compare to, so simply wait for the next object to appear before responding.\n\n"
    "Any questions?"
)

EXPO_KEY_REMINDER = (
    "Reminder:\n\n"
    "Press '{bigger_key}' if the current object is bigger than the previous one.\n"
    "Press '{smaller_key}' if the current object is smaller than the previous one.\n\n"
    "Press the spacebar to begin."
)

EXPO_NO_PRACT = (
    "You will see images of objects in a stream. Your task is to decide whether each object is bigger or smaller than the one before it.\n"
    "All objects are shown as the same size on the screen. Base your answer on their real-life size.\n\n"
    "Press '{bigger_key}' if the current object is bigger than the previous one.\n"
    "Press '{smaller_key}' if the current object is smaller than the previous one.\n\n"
    "Press the spacebar to begin the experiment."
)

#################
# TEST PHASE TEXT
#################

# Test phase instructions
TEST_1_INSTRUCT = (
    "You are now moving on to the test phase of the experiment.\n\n\n"
    "Please wait for the experimenter to provide further instructions..."
)

TEST_2_INSTRUCT = (
    "During the bigger/smaller task, you may have noticed some objects seem to belong together based on their proximity in the sequence or associations with other objects.\n\n"
    "In this part of the experiment, you will complete several tests. Each test will show two pairs of objects, one after the other, with a brief pause in between.\n\n"
    "Your task is to decide which pair feels more familiar based on what you saw earlier in the experiment.\n\n"
    "Press '{seq1_key}' if the FIRST pair feels more familiar.\n"
    "Press '{seq2_key}' if the SECOND pair feels more familiar."
)

TEST_3_PRACTICE = (
    "Before you begin, you will start with a few practice tests.\n\n"
    "Practice tests will use the same simple object icons colored black as in the first practice you completed, but there are no right or wrong answers during practice. The goal is to help you get used to the format.\n\n"
    "Any questions?"
)

TEST_4_PRACT_DONE = (
    "Practice Complete!\n\n"
    "You can choose to RESTART the practice if you feel you need more practice, or CONTINUE to the experiment if you understand the test format.\n\n"
    "Press 'C' to CONTINUE to the experiment\n"
    "Press 'R' to RESTART the practice"
)

TEST_5_PRACT_CONT = (
    "Great! Now you understand the format of the next phase.\n\n"
    "Please wait for the experimenter to provide further instructions..."
)

TEST_6_PRACT_RETRY = (
    "Let's try the practice again.\n\n"
    "Remember:\n"
    "Press '{seq1_key}' for FIRST sequence\n"
    "Press '{seq2_key}' for SECOND sequence\n\n"
    "Press the spacebar to continue the practice."
)

TEST_7_PRACT_MAX = (
    "Maximum practice attempts ({max_attempts}) reached.\n\n"
    "You will now proceed to the actual experiment.\n\n"
    "Please wait for the experimenter to provide further instructions..."
)

TEST_NO_PRACTICE_TEXT = (
    "You will see two sequences of image pairs.\n"
    "Choose which sequence went together during the exposure phase.\n\n"
    "Press '{seq1_key}' for FIRST sequence\n"
    "Press '{seq2_key}' for SECOND sequence\n\n"
    "Press any key to begin the tests."
)

TEST_MAIN_TEXT = (
    "Now that you have completed the practice tests, you will proceed to the actual experiment.\n\n"
    "Everything will be the same as in practice tests, but you will see the images of real-world objects from the bigger/smaller task instead of simple icons.\n\n"
    "Again, each test will show two pairs of objects, one after the other, with a brief pause in between.\n"
    "You should decide which pair feels more familiar to you based on what you saw in the bigger/smaller task.\n"
    "You can only provide your answer after seeing both pairs.\n\n"
    "Any questions?"
)

TEST_KEY_REMINDER = (
    "Reminder:\n\n"
    "Press '{seq1_key}' for FIRST sequence\n"
    "Press '{seq2_key}' for SECOND sequence\n\n"
    "Press the spacebar to begin."
)

TEST_RESP_PROMPT = (
    "Which pair felt more familiar?\n\n"
    "Press '{seq1_key}' for FIRST sequence\n"
    "Press '{seq2_key}' for SECOND sequence"
)

# General messages
EXPERIMENT_CANCELLED_TEXT = "Experiment cancelled.\n\nThank you for your time."

EXPERIMENT_COMPLETE_TEXT = "Thank you for participating!\n\nThis concludes the experiment."