# onscreen_text.py
#
# Text variables for all instruction screens in the Inmix SL Object Inference experiment
# These texts are used throughout the experiment to provide consistent messaging


#################   
# EXPOSURE TEXT
#################
EXPO_1_INSTRUCT = (
    "Welcome!\n\n"
    "You will see a series of object images appear one after another.\n"
    "Your task is to decide whether each object is LARGER or SMALLER than the one shown just before it.\n\n"
    "All images will appear as the same size on the screen.\n\n"
    "Base your decisions on each object's real-life size.\n\n"
    "Press '{larger_key}' if the current object is LARGER than the previous one.\n"
    "Press '{smaller_key}' if the current object is SMALLER than the previous one."
)

EXPO_2_PRACTICE = (
    "Before starting the larger/smaller task, you will complete a few practice trials to get used to the format and timing.\n\n"
    "You must achieve 100% accuracy on the practice trials to proceed to the actual task.\n"
    "Images appear for only a few moments, so pay close attention and respond as quickly and accurately as you can.\n\n"
    "Practice trials will use simple black icons of objects.\n\n"
    "Any questions before we begin?"
)

EXPO_3_PRACT_RETRY = (
    "Accuracy: {accuracy:.1%}\n"
    "You need 100% accuracy to continue.\n\n"
    "Let's try the practice again.\n\n"
    "REMEMBER:\n"
    "Press '{larger_key}' if the current object is LARGER than the previous one.\n"
    "Press '{smaller_key}' if the current object is SMALLER than the previous one.\n\n"
    "Press the SPACEBAR to retry."
)

EXPO_4_PRACT_DONE = (
    "You achieved {accuracy:.1%} accuracy.\n\n"
    "Please wait for the experimenter to provide further instructions..."
)

EXPO_MAIN_TEXT = (
    "Now that you have completed the practice trials, you will begin the actual task.\n\n"
    "Everything will work the same way as in the practice, except now you will see real-world object images instead of icons.\n\n"
    "All images will appear as the same size on the screen.\n"
    "Base your answers on each object's real-life size.\n\n"
    "You will have a few short breaks during the task.\n"
    "After each break, the first object shown will NOT have a previous object to compare to, so you should simply wait for the second image to appear before responding.\n\n"
    "Any questions before you continue?"
)

EXPO_KEY_REMINDER = (
    "Reminder:\n\n"
    "Press '{larger_key}' if the current object is LARGER than the previous one.\n"
    "Press '{smaller_key}' if the current object is SMALLER than the previous one.\n\n"
    "Press the SPACEBAR when you are ready to begin."
)

EXPO_NO_PRACT = (
    "You will see images of objects in a stream. Your task is to decide whether each object is larger or smaller than the object before it.\n"
    "All objects are shown as the same size on the screen. Base your answer on their real-life size.\n\n"
    "Press '{larger_key}' if the current object is LARGER than the previous one.\n"
    "Press '{smaller_key}' if the current object is SMALLER than the previous one.\n\n"
    "Press the SPACEBAR to begin the experiment."
)

EXPO_RESP_PROMPT = (
    "'{larger_key}' - LARGER than last object\n"
    "'{smaller_key}' - SMALLER than last object"
)

#################
# TEST PHASE TEXT
#################

# Test phase instructions
TEST_1_INSTRUCT = (
    "You have completed the first part of the experiment and will now move on to the second part of the experiment.\n\n\n"
    "Please wait for the experimenter to provide further instructions..."
)

TEST_2_INSTRUCT = (
    "During the larger/smaller task, you may have noticed that some objects seemed to appear close together.\n\n"
    "In this part of the experiment, you will complete several tests. Each test will show two pairs of objects, one pair after the other, with a short pause in between.\n\n"
    "Your task is to decide which pair feels MORE FAMILIAR based on what you saw during the larger/smaller task earlier in the experiment.\n\n"
    "Press '{seq1_key}' if the FIRST pair feels more familiar.\n"
    "Press '{seq2_key}' if the SECOND pair feels more familiar.\n\n"
    "Respond only after you have seen both pairs."
)

TEST_3_PRACTICE = (
    "Before you begin the actual test, you will complete a few practice trials to get used to the format.\n\n"
    "Practice tests will use simple black icons, like the practice for the larger/smaller task.\n\n"
    "There are no right or wrong answers for the practice test. Instead, the goal is for you to get comfortable with the format of the tests.\n\n"
    "Any questions before we begin?"
)

TEST_4_PRACT_DONE = (
    "Practice complete!\n\n"
    "You may RESTART the practice if you’d like more practice,\n"
    "or CONTINUE to the actual tests if you understand the format.\n\n"
    "Press 'C' to CONTINUE to the experiment.\n"
    "Press 'R' to RESTART the practice tests."
)

TEST_5_PRACT_CONT = (
    "Great! You now understand the test format.\n\n"
    "Please wait for the experimenter to provide further instructions..."
)

TEST_6_PRACT_RETRY = (
    "Let's try the practice again.\n\n"
    "REMEMBER:\n"
    "Press '{seq1_key}' if the FIRST pair feels more familiar.\n"
    "Press '{seq2_key}' if the SECOND pair feels more familiar.\n\n"
    "Press the SPACEBAR to continue the practice."
)

TEST_7_PRACT_MAX = (
    "Maximum number of practice attempts reached ({max_attempts}).\n\n"
    "You will now proceed to the ACTUAL experiment.\n\n"
    "Please wait for the experimenter to provide further instructions..."
)

TEST_NO_PRACTICE_TEXT = (
    "You will see two image pairs.\n"
    "Choose which pair went together during the exposure phase.\n\n"
    "Press '{seq1_key}' for FIRST pair\n"
    "Press '{seq2_key}' for SECOND pair\n\n"
    "Press any key to begin the tests."
)

TEST_MAIN_TEXT = (
    "Now that you have completed the practice tests, you will begin the actual tests.\n\n"
    "Everything will be the same as in practice, except you will see real-world object images from the larger/smaller task instead of icons.\n\n"
    "Each test will show two pairs of objects, one pair after the other, with a short pause in between.\n\n"
    "After both pairs have been shown, decide which pair feels MORE FAMILIAR based on what you saw in the larger/smaller task.\n\n"
    "Any questions before you continue?"
)

TEST_KEY_REMINDER = (
    "Reminder:\n\n"
    "Press '{seq1_key}' for the FIRST pair.\n"
    "Press '{seq2_key}' for the SECOND pair.\n\n"
    "Press the SPACEBAR when you are ready to begin."
)


TEST_RESP_PROMPT = (
    "Which pair felt MORE FAMILIAR?\n\n"
    "Press '{seq1_key}' for the FIRST pair.\n"
    "Press '{seq2_key}' for the SECOND pair."
)

# General messages
EXPERIMENT_CANCELLED_TEXT = "Experiment cancelled.\n\nThank you for your time."

EXPERIMENT_COMPLETE_TEXT = "Thank you for participating!\n\nThis concludes the experiment."