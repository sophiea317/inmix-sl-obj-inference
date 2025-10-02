# stimuli_generation.py

# import necessary packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy.random import randint, choice as randchoice


# set up experiment parameters
stim_folder = "assets"
stim_set = [f for f in os.listdir(stim_folder) if f.endswith(".png")]
stim_num = len(stim_set)
unique_stim = [f for f in stim_set if "set" not in f]
linking_stim = [f for f in stim_set if "set" in f]

print("Linking stimuli:", linking_stim, "\nTotal:", len(linking_stim))
print("Unique stimuli:", unique_stim, "\nTotal:", len(unique_stim))

# separate linking stimuli by set (sorted to keep order)
linking_stim_by_set = {}
for f in np.sort(linking_stim):
    set_id = f.split("_")[0]
    if set_id not in linking_stim_by_set:
        linking_stim_by_set[set_id] = []
    linking_stim_by_set[set_id].append(f)
num_linking_sets = len(linking_stim_by_set)
print("Number of linking sets:", num_linking_sets)
print("Linking stimuli by set:", linking_stim_by_set)

# assign the 6 BC pairs by first tasking img-02 in each set and randomly choose either img-01 or img-03 as the other item in the pair. 
# then take img-04 in each set and pair it with img-05 or img-03 (if not already used) to make the second BC pair for each set.
b_stim = {}
c_stim = {}
pair_counter = 0
for set_id, imgs in linking_stim_by_set.items():
    # first BC pair
    first_item = imgs[1]  # img-02
    second_item = randchoice([imgs[0], imgs[2]])  # img-01 or img-03
    if randint(0, 2) == 0:
        b_stim[pair_counter] = first_item
        c_stim[pair_counter] = second_item
    else:
        b_stim[pair_counter] = second_item
        c_stim[pair_counter] = first_item
    pair_counter += 1

    # second BC pair
    first_item = imgs[3]  # img-04
    if imgs[2] not in c_stim.values() and imgs[2] not in b_stim.values():
        second_item = imgs[2]
    else:
        second_item = imgs[4]  # img-05
        
    # randomize order within the pair
    if randint(0, 2) == 0:
        b_stim[pair_counter] = first_item
        c_stim[pair_counter] = second_item
    else:
        b_stim[pair_counter] = second_item
        c_stim[pair_counter] = first_item
    pair_counter += 1

    # gather images not in assigned as B or C stim
    used_imgs = list(b_stim.values()) + list(c_stim.values())
    unused_imgs = [img for img in linking_stim if img not in used_imgs]
# print the B and C stimuli to visually check
print("B stimuli:", b_stim)
print("C stimuli:", c_stim)

# combine unused linking stimuli with unique stimuli
remaining_stim = unique_stim + unused_imgs
print("Non-BC stimuli:", remaining_stim, "\nTotal:", len(remaining_stim))

# print the BC pairs to visually check
bc_pairs = list(zip(b_stim.values(), c_stim.values()))
print("BC pairs:")
for i, pair in enumerate(bc_pairs):
    print(f"Pair {i+1}: {pair[0]} & {pair[1]}")

# assign the 6 A and 6 D stimuli by randomly selecting from the remaining stimuli. the images from the same set should not be assigned to A and D.
a_stim = {}
d_stim = {}
used_sets = set()
for i in range(len(bc_pairs)):
    while True:
        candidate_a = randchoice(remaining_stim)
        candidate_d = randchoice(remaining_stim)
        set_a = candidate_a.split("_")[0]
        set_d = candidate_d.split("_")[0]
        if set_a != set_d and set_a not in used_sets and set_d not in used_sets:
            a_stim[f"A{i+1}"] = candidate_a
            d_stim[f"D{i+1}"] = candidate_d
            used_sets.update([set_a, set_d])
            remaining_stim.remove(candidate_a)
            remaining_stim.remove(candidate_d)
            break

abcd_groups = {
    "A": a_stim,
    "B": b_stim,
    "C": c_stim,
    "D": d_stim
}

# print abcd groups to visually check
for group, items in abcd_groups.items():
    print(f"{group} stimuli:")
    for label, img in items.items():
        print(f"  {label}: {img}")
    print()


# plot the stimuli for each group number to visual check
fig, axes = plt.subplots(6, 4, figsize=(18, 18))
for i, (group, items) in enumerate(abcd_groups.items()):
    for j, (label, img) in enumerate(items.items()):
        ax = axes[j, i]
        img_data = mpimg.imread(os.path.join(stim_folder, img))
        ax.imshow(img_data)
        ax.set_title(f"{group}{j+1}")
        ax.axis('off')
plt.tight_layout()
plt.savefig("qa/stimuli_check.png")