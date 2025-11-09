# inmix-sl-obj-inference

# Running from BW computers  
```
cd /media/bw-share/experiments/turk-browne_nbt2/allen_sda37/inmix-sl-obj-inference
```
```
python run_sl-obj-inference.py --sub 1234 --ses 001 --expo retrospective --test 2-step --debug f --practice t --expoRun t --testRun t
```
or simply
```
python run_sl-obj-inference.py --sub 1234 --expo retrospective --test 2-step
```

# Experiment folder set-up
```
inmix-sl-obj-inference/  
└── run_sl-obj-inference.py         # main PsychoPy experiment runner 
└──data
└──practice
└──stimuli       
└──utils  
    └── stimuli_generation.py       # makes stimuli lists  
    └── exposure_trials.py          # generate exposure streams 
    └── test_trials.py              # generate test trials  
    └── onscreen_text.py            # text for instructions
```

### `exposure_trials.py`

**purpose:**
This script generates the **exposure phase trial sequences** for the experiment. It builds temporally structured stimulus streams (e.g., A–B–C–D sets) and assigns **1-back trials** under strict spacing and balancing constraints.

---

#### overview

The code constructs two types of visual exposure streams:

* **Stream 1 (“AB_CD”)** — intermixed stream of A–B and C–D stimulus pairs.
* **Stream 2 (“BC”)** — stream of B–C pairs that link across the pairs in the AB_CD.

Each stream contains **1-back repetition trials** used for the working-memory cover task. The algorithm ensures:

* Balanced distribution of 1-back trials across all pairs.
* Even spacing of 1-backs across each stream using trial bins (e.g., bins=[1, 60, 120, 180, 238]).
* No consecutive or overlapping 1-backs.
* Separate control of 1-back position (repeat of first vs. second pair).
* A–B and C–D pairs within an A–B–C–D set are never shown consecutively in the AB_CD stream.