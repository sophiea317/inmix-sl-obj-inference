# onscreen_text.py
#
# Text variables for all instruction screens in the Inmix SL Object Inference experiment
# These texts are used throughout the experiment to provide consistent messaging


#################   
# EXPOSURE TEXT
#################
EXPO_1_INSTRUCT = (
    "Welcome!\n\n"
    "In this part of the experiment, you will see a sequence of object images, with each image appearing one at a time.\n\n"
    "YOUR TASK:\n"
    "Compare each object to the one shown immediately before it, and decide whether the current object is LARGER or SMALLER than the previous.\n\n"
    "All images will be displayed as the same size on the screen. You must judge objects by their real-world size, NOT their size on the screen.\n\n"
    "HOW TO RESPOND:\n"
    "Press '{larger_key}' if the current object is LARGER than the previous object.\n"
    "Press '{smaller_key}' if the current object is SMALLER than the previous object.\n\n"
    "Each image will appear for only a few seconds, so pay close attention and respond as quickly and accurately as you can.\n\n"
    "Note: You will not need to respond to the first image, since there is nothing to compare it to."
)

EXPO_2_PRACTICE = (
    "Practice Round\n\n"
    "Before starting the experiment, you will complete a practice round of the larger/smaller task using simple black icons of objects.\n\n"
    "To get you comfortable with the task format, timing, and response keys, you must achieve 100% accuracy to proceed to the experiment.\n\n"
    "The practice will be repeated until you reach 100%.\n\n"
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
    "Now that you have completed the practice round, you will now proceed to the actual task.\n\n"
    "Everything will work the same way as in the practice, except now you will see real-world object images instead of icons.\n\n"
    "Remember to judge objects by their real-world size, NOT their size as it appears on the screen.\n\n"
    "You will have a few short breaks spread throughout the task.\n"
    "After each break, you will not need to respond to the first image, since there is nothing to compare it to.\n\n"
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
    "You have completed the first part of the experiment and will now move on to the next part of the experiment.\n\n\n"
    "Please wait for the experimenter to provide further instructions..."
)

TEST_2_INSTRUCT = (
    "During the larger/smaller task, you might have noticed that some objects tended to go together because they appeared close to each other in the sequence or seemed related in some way.\n\n"
    "In this next part of the experiment, you will complete several tests based on the patterns you may have noticed.\n\n"
    "YOUR TASK:\n"
    "In each test you will see two pairs of objects. Your task is to decide which pair feels MORE FAMILIAR based on what you observed during the larger/smaller task.\n\n"
    "The first pair of objects will appear on screen, with each object shown one at a time. After a brief pause, the second pair will appear the same way. Then you will be prompted to make your choice.\n\n"
    "HOW TO RESPOND:\n"
    "Press '{seq1_key}' if the FIRST pair feels more familiar.\n"
    "Press '{seq2_key}' if the SECOND pair feels more familiar.\n\n"
    "You can take your time to decide, but you must make a choice to continue to the next test."
)

TEST_3_PRACTICE = (
    "Practice Round\n\n"
    "Before starting the actual tests, you will complete a practice of the task.\n\n" 
    "The practice will use the same simple black icons you saw during the larger/smaller task practice.\n\n"
    "There are no right or wrong answers for the practice trials. Instead, the goal is for you to get comfortable with the format of the test.\n\n"
    "Any questions before we begin?"
)

TEST_4_PRACT_DONE = (
    "Practice Round Complete!\n\n"
    "You may RESTART the practice if you’d like more practice,\n"
    "or CONTINUE to the actual tests if you understand the format.\n\n"
    "If you want to CONTINUE to the experiment, press 'C'.\n"
    "If you want to RESTART the practice tests, press 'R'."
)

TEST_5_PRACT_CONT = (
    "You are now ready to continue to the actual tests.\n\n"
    "Please wait for the experimenter to provide further instructions..."
)

TEST_6_PRACT_RETRY = (
    "Let's try the practice again.\n\n"
    "REMEMBER:\n"
    "Press '{seq1_key}' if the FIRST pair feels more familiar.\n"
    "Press '{seq2_key}' if the SECOND pair feels more familiar.\n\n"
    "Press the SPACEBAR to start the practice."
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
    "Now that you've completed the practice tests, you'll begin the actual tests.\n\n"
    "Everything will be the same as in the practice, but now you’ll see real-world objects from the larger/smaller task instead of icons.\n\n"
    "All of the objects you’ll see were shown during the larger/smaller task, so both pairs may feel somewhat familiar.\n\n"
    "If you’re unsure, trust your gut and choose the pair that feels most familiar based only on what you saw earlier in the experiment. Try not to choose based on your personal preferences or opinions about the objects.\n\n"
    "Any questions before you continue?"
)

TEST_KEY_REMINDER = (
    "Reminder:\n\n"
    "Decide which pair feels MORE FAMILIAR.\n"
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