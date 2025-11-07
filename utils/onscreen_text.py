# onscreen_text.py
#
# Text variables for all instruction screens in the Inmix SL Object Inference experiment
# These texts are used throughout the experiment to provide consistent messaging



# Practice exposure instructions
WELCOME_PRACTICE_TEXT = (
    "Welcome!\n\n"
    "You will see images of objects in a stream. Your task is to decide whether each object is bigger or smaller than the one before it.\n"
    "All objects are shown as the same size on the screen. Base your answer on their real-life size.\n\n"
    "Press '{bigger_key}' if the current object is bigger than the previous one.\n"
    "Press '{smaller_key}' if the current object is smaller than the previous one.\n\n"
    "You must achieve 100% accuracy on practice trials to continue.\n\n"
    "Press any key to begin practice."
)

PRACTICE_SUCCESS_TEXT = (
    "Excellent! You achieved 100% accuracy!\n\n"
    "You're ready to proceed to the actual experiment.\n\n"
    "Press any key to continue."
)

PRACTICE_MAX_ATTEMPTS_TEXT = (
    "Practice complete after {max_attempts} attempts.\n\n"
    "Accuracy: {accuracy:.1%}\n\n"
    "You will now proceed to the actual experiment.\n\n"
    "Press any key to continue."
)

PRACTICE_RETRY_TEXT = (
    "Practice accuracy: {accuracy:.1%}\n"
    "You need 100% accuracy to continue.\n"
    "Let's try the practice again.\n\n"
    "Remember:\n"
    "- Press '{bigger_key}' if bigger than previous object\n"
    "- Press '{smaller_key}' if smaller than previous object\n\n"
    "Press any key to retry."
)

# Main exposure instructions
MAIN_EXPOSURE_START_TEXT = (
    "Great! Now the first part of the experiment will begin.\n\n"
    "Press '{bigger_key}' if bigger than previous object\n"
    "Press '{smaller_key}' if smaller than previous object\n\n"
    "Press any key to start the experiment."
)

EXPOSURE_NO_PRACTICE_TEXT = (
    "You will see images of objects in a stream. Your task is to decide whether each object is bigger or smaller than the one before it.\n"
    "All objects are shown as the same size on the screen. Base your answer on their real-life size.\n\n"
    "Press '{bigger_key}' if the current object is bigger than the previous one.\n"
    "Press '{smaller_key}' if the current object is smaller than the previous one.\n\n"
    "Press any key to begin the experiment."
)

# Test phase instructions
TEST_PRACTICE_INTRO_TEXT = (
    "During the bigger/smaller task, you may have noticed some objects seem to belong together based on their proximity in the sequence or associations with other objects.\n"
    "In this part of the experiment, you will complete several tests. Each test will show two pairs of objects, one after the other, with a brief pause in between.\n"
    "Your task is to decide which pair feels more familiar based on what you saw earlier in the experiment.\n\n"
    "Press '{seq1_key}' if the FIRST pair feels more familiar.\n"
    "Press '{seq2_key}' if the SECOND pair feels more familiar.\n\n"
    "You will start with a few practice tests. There are no right or wrong answers during practice. The goal is to help you get used to the format.\n\n"
    "Press any key to begin the practice."
)

PRACTICE_TEST_COMPLETE_RETRY_TEXT = (
    "Practice Complete!\n\n"
    "You can choose to RESTART the practice if you feel you need more practice, or CONTINUE to the experiment if you understand the test format.\n\n"
    "Press 'C' to CONTINUE to the experiment\n"
    "Press 'R' to RESTART the practice"
)

PRACTICE_TEST_CONTINUE_TEXT = (
    "Great! Now you understand the format of the next phase.\n"
    "The actual tests will use images of real-world objects that you saw during the bigger/smaller task.\n\n"
    "Press any key to begin the actual experiment."
)

PRACTICE_TEST_MAX_ATTEMPTS_TEXT = (
    "Maximum practice attempts ({max_attempts}) reached.\n\n"
    "You will now proceed to the actual experiment.\n\n"
    "Press any key to continue."
)

PRACTICE_TEST_RETRY_TEXT = (
    "Let's try the practice again.\n\n"
    "Remember:\n"
    "- Press '{seq1_key}' for FIRST sequence\n"
    "- Press '{seq2_key}' for SECOND sequence\n\n"
    "Press any key to continue the practice."
)

PRACTICE_TEST_NO_DATA_TEXT = (
    "Test practice complete.\n\n"
    "Now the real test will begin.\n\n"
    "Press any key to continue."
)

TEST_NO_PRACTICE_TEXT = (
    "You will see two sequences of image pairs.\n"
    "Choose which sequence went together during the exposure phase.\n\n"
    "Press '{seq1_key}' for FIRST sequence\n"
    "Press '{seq2_key}' for SECOND sequence\n\n"
    "Press any key to begin the tests."
)

MAIN_TEST_PHASE_TEXT = (
    "Again, during the bigger/smaller task, you may have noticed some objects seem to belong together based on their proximity in the sequence or associations with other objects.\n"
    "In this part of the experiment, you will complete several tests. Each test will show two pairs of objects, one after the other, with a brief pause in between.\n"
    "Your task is to decide which pair feels more familiar based on what you saw earlier in the experiment.\n\n"
    "Press '{seq1_key}' if the FIRST pair feels more familiar.\n"
    "Press '{seq2_key}' if the SECOND pair feels more familiar.\n\n"
    "You will start with a few practice tests. There are no right or wrong answers during practice. The goal is to help you get used to the format.\n\n"
    "Press any key to begin the practice."
)

TEST_RESPONSE_PROMPT = (
    "Which pair feels more familiar?\n\n"
    "Press '{seq1_key}' for FIRST sequence\n"
    "Press '{seq2_key}' for SECOND sequence"
)

# General messages
EXPERIMENT_CANCELLED_TEXT = "Experiment cancelled.\n\nThank you for your time."

EXPERIMENT_COMPLETE_TEXT = "Thank you for participating!\n\nThis concludes the experiment."