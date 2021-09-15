# ------------------------------------------------------------------------------ #
# Load various MuJoCo snapshots. Usually we're loading from well-trained policies,
# or those with standard RL w/the usual noise amounts (e.g., 0.1 for TD3). By
# default, we load the LAST saved snapshot from the training run.
# ------------------------------------------------------------------------------ #
# We can actually use this in two ways. One: for any imitation-like settings where
# we add noise to data, or two: use to train noise predictor.
# ------------------------------------------------------------------------------ #

# Distribution of sigmas. Default data sizes are 1M train and 500K valid.
# DIST=uniform_0.01_1.00

# CPUs to expose to the agent.
CPU1=0
CPU2=12

# Pick one.
#NAME=halfcheetah_td3_act0-1
#NAME=halfcheetah_td3_act0-5
#NAME=hopper_td3_act0-1
#NAME=hopper_td3_act0-5
#NAME=walker2d_td3_act0-1
#NAME=walker2d_td3_act0-5

## # Generates 6 datasets. [Daniel: done earlier, Jan 10-ish, also I don't think we need seed 20 if we're not going to use.]
## for DIST in  uniform_0.0_0.25  uniform_0.0_0.50  uniform_0.0_0.75  uniform_0.0_1.00  uniform_0.0_1.25  ; do
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s10/  --noise ${DIST}  -nr
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s20/  --noise ${DIST}  -nr
## done

## # Generates noise-free data. [Daniel: done earlier, Jan 10-ish, for act 0.1, and Jan 17 for act 0.5]
## for DIST in  constant_0.0 ; do
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s10/  --noise ${DIST}  -nr
## done

# Generates uniformeps data, the same as earlier except. The last number means we do noise-free some % of the time.
# Here we're doing eps=0.5 on a per-action basis within each episode. We also have another decision point about
# filtering vs no filtering, so if we include any noisy actions in the data. [Daniel: generating Jan 17 ...]

## for DIST in  uniformeps_0.0_0.25_0.5_nofilt  uniformeps_0.0_0.50_0.5_nofilt  uniformeps_0.0_0.75_0.5_nofilt  uniformeps_0.0_1.00_0.5_nofilt  uniformeps_0.0_1.25_0.5_nofilt  ; do
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s10/  --noise ${DIST}  -nr
## done
##
## for DIST in  uniformeps_0.0_0.25_0.5_filter  uniformeps_0.0_0.50_0.5_filter  uniformeps_0.0_0.75_0.5_filter  uniformeps_0.0_1.00_0.5_filter  uniformeps_0.0_1.25_0.5_filter  ; do
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s10/  --noise ${DIST}  -nr
## done



# -------------------------------------------------------------------------------------------------- #
# Now trying to increase noise, and making it simpler with a constant sigma value. Also,
# we're trying to use teacher seed 40 for HalfCheetah and Hopper, and 50 for Walker2d.
# -------------------------------------------------------------------------------------------------- #
# Since we're still hoping to use the noise predictor, and that was trained on Uniform_0_x data for
# `x in {0.25, 0.5, 0.75, 1.0, 1.25}`, then presumably if we use Constant_x_e where e is the filtering
# value (e=0.5 by default) then that can use noise predictor trained on uniform_0_x since the filtering
# will hopefully mean we get a blend of states that would be from the distribution induced from a fixed
# sigma value, for sigma within 0 and x.
# However, I don't have a Noise Predictor trained on Uniform(0, 1.5) data, but Constanteps_1.5_0.5
# can probably use the NP trained on Uniform(0, 1.25) data.
# -------------------------------------------------------------------------------------------------- #
# [Daniel: started generating Jan 23 on `constanteps` branch]
# -------------------------------------------------------------------------------------------------- #

# NAME=halfcheetah_td3_act0-1
# for DIST in constanteps_0.00_0.0_filter constanteps_0.25_0.5_filter constanteps_0.50_0.5_filter constanteps_1.00_0.5_filter constanteps_1.50_0.5_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
# done
#
# NAME=halfcheetah_td3_act0-5
# for DIST in constanteps_0.00_0.0_filter constanteps_0.25_0.5_filter constanteps_0.50_0.5_filter constanteps_1.00_0.5_filter constanteps_1.50_0.5_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
# done
#
# NAME=hopper_td3_act0-1
# for DIST in constanteps_0.00_0.0_filter constanteps_0.25_0.5_filter constanteps_0.50_0.5_filter constanteps_1.00_0.5_filter constanteps_1.50_0.5_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
# done
#
# NAME=hopper_td3_act0-5
# for DIST in constanteps_0.00_0.0_filter constanteps_0.25_0.5_filter constanteps_0.50_0.5_filter constanteps_1.00_0.5_filter constanteps_1.50_0.5_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
# done
#
# NAME=walker2d_td3_act0-1
# for DIST in constanteps_0.00_0.0_filter constanteps_0.25_0.5_filter constanteps_0.50_0.5_filter constanteps_1.00_0.5_filter constanteps_1.50_0.5_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s50/  --noise ${DIST}  -nr
# done
#
# NAME=walker2d_td3_act0-5
# for DIST in constanteps_0.00_0.0_filter constanteps_0.25_0.5_filter constanteps_0.50_0.5_filter constanteps_1.00_0.5_filter constanteps_1.50_0.5_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s50/  --noise ${DIST}  -nr
# done

# -------------------------------------------------------------------------------------------------- #
# [Daniel: started generating Jan 26 on `constanteps` branch, let's make even noisier]
# And now I actually think we want to go back to the uniformeps case. :( Let's do two cases.
# -------------------------------------------------------------------------------------------------- #

# NAME=halfcheetah_td3_act0-1
# for DIST in uniformeps_0.0_1.50_0.5_filter uniformeps_0.0_1.50_0.75_filter uniformeps_0.0_1.50_0.9_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
# done
#
# NAME=halfcheetah_td3_act0-5
# for DIST in uniformeps_0.0_1.50_0.5_filter uniformeps_0.0_1.50_0.75_filter uniformeps_0.0_1.50_0.9_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
# done
#
# NAME=hopper_td3_act0-1
# for DIST in uniformeps_0.0_1.50_0.5_filter uniformeps_0.0_1.50_0.75_filter uniformeps_0.0_1.50_0.9_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
# done
#
# NAME=hopper_td3_act0-5
# for DIST in uniformeps_0.0_1.50_0.5_filter uniformeps_0.0_1.50_0.75_filter uniformeps_0.0_1.50_0.9_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
# done
#
# NAME=walker2d_td3_act0-1
# for DIST in uniformeps_0.0_1.50_0.5_filter uniformeps_0.0_1.50_0.75_filter uniformeps_0.0_1.50_0.9_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s50/  --noise ${DIST}  -nr
# done
#
# NAME=walker2d_td3_act0-5
# for DIST in uniformeps_0.0_1.50_0.5_filter uniformeps_0.0_1.50_0.75_filter uniformeps_0.0_1.50_0.9_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s50/  --noise ${DIST}  -nr
# done

# -------------------------------------------------------------------------------------------------- #
# [Daniel: generating Feb 03 on `uniform-act-data` branch, let's avoid additive Gaussian noise.]
# So, recall: teachers seeds 40/50, with the filtering applied.
# Now we can do uniformeps as before with 3 filtering applied, except each time we have to add noise,
# it involves random actions. So call it ... ''?
# -------------------------------------------------------------------------------------------------- #

# NAME=halfcheetah_td3_act0-1
# for DIST in randact_0.25_filter randact_0.50_filter randact_0.75_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
# done
#
# NAME=halfcheetah_td3_act0-5
# for DIST in randact_0.25_filter randact_0.50_filter randact_0.75_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
# done
#
# NAME=hopper_td3_act0-1
# for DIST in randact_0.25_filter randact_0.50_filter randact_0.75_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
# done
#
# NAME=hopper_td3_act0-5
# for DIST in randact_0.25_filter randact_0.50_filter randact_0.75_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
# done
#
# NAME=walker2d_td3_act0-1
# for DIST in randact_0.25_filter randact_0.50_filter randact_0.75_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s50/  --noise ${DIST}  -nr
# done
#
# NAME=walker2d_td3_act0-5
# for DIST in randact_0.25_filter randact_0.50_filter randact_0.75_filter ; do
#     taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s50/  --noise ${DIST}  -nr
# done

# -------------------------------------------------------------------------------------------------- #
# [Daniel: generated Feb 10 on master, this time we do this correctly...]
# Call this nonaddunif_0.0_1.0_filter where the 'unif_0.0_1.0' indicates that we draw a random
# action probability from this distribution before each episode. This seems way better, TBH. Really
# should have thought of doing this before. It's like randact_X_filter except the "X is sampled from
# a distribution and fixed for each episode, which will mean more noise-free episodes.
# -------------------------------------------------------------------------------------------------- #

NAME=halfcheetah_td3_act0-1
for DIST in nonaddunif_0.0_0.5_filter nonaddunif_0.0_1.0_filter ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
done

NAME=halfcheetah_td3_act0-5
for DIST in nonaddunif_0.0_0.5_filter nonaddunif_0.0_1.0_filter ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
done

NAME=hopper_td3_act0-1
for DIST in nonaddunif_0.0_0.5_filter nonaddunif_0.0_1.0_filter ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
done

NAME=hopper_td3_act0-5
for DIST in nonaddunif_0.0_0.5_filter nonaddunif_0.0_1.0_filter ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s40/  --noise ${DIST}  -nr
done

NAME=walker2d_td3_act0-1
for DIST in nonaddunif_0.0_0.5_filter nonaddunif_0.0_1.0_filter ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s50/  --noise ${DIST}  -nr
done

NAME=walker2d_td3_act0-5
for DIST in nonaddunif_0.0_0.5_filter nonaddunif_0.0_1.0_filter ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/load_policy.py  ${NAME}/${NAME}_s50/  --noise ${DIST}  -nr
done
