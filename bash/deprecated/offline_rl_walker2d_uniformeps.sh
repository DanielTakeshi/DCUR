# ------------------------------------------------------------------------------ #
# Offline RL, with the noise predictor, add '_np' to the end to NAME.
# ------------------------------------------------------------------------------ #
# This would be easier if we could have a scalable way of referring to the DATA
# and noise predictor files. Alas, for now we put in checks in the script.
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=8

## # Distribution for network.
## DIST_N=uniform_0.0_0.25
## DIST_N=uniform_0.0_0.50
## DIST_N=uniform_0.0_0.75
## DIST_N=uniform_0.0_1.00
## DIST_N=uniform_0.0_1.25
##
## # Distribution for data. [Must match noise]
## DIST_B=uniformeps_0.0_0.25_0.5_filter
## DIST_B=uniformeps_0.0_0.50_0.5_filter
## DIST_B=uniformeps_0.0_0.75_0.5_filter
## DIST_B=uniformeps_0.0_1.00_0.5_filter
## DIST_B=uniformeps_0.0_1.25_0.5_filter
##
##
## # Main new things: N_PATH for noise path, N_ALPHA for alpha strength of reward shaping.
## N_ALPHA=10.0
## ENV=Walker2d-v3
## NAME=walker2d_td3_offline_${DIST_B}_np_${N_ALPHA}
## T_SOURCE=walker2d_td3_act0-1_s10
## B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST_B}-dtype-train.p
## N_PATH=walker2d_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-${DIST_N}-t-1000000-v-0500000.tar
##
## # These seeds are for the STUDENT, with a FIXED teacher.
## for SEED in 10 20 ; do
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##         --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  -np ${N_PATH}  -na ${N_ALPHA}
## done

# ------------------------------------------------------------------------------ #
# JAN 24, 2021: now supporting constanteps. NOTE: noise predictor has 200K valid now.
# Feb 01, 2021: making this a bit more scalable / easy to use on GCP. NOTE: for these
# if using noise predictor, the main decision point is to pick DIST_B and DIST_N, and
# THOSE HAVE TO BE SELECTED TOGETHER. Possible pairings:
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
#DIST_B=constanteps_0.25_0.5_filter
#DIST_N=uniform_0.0_0.25
# ------------------------------------------------------------------------------ #
#DIST_B=constanteps_0.50_0.5_filter
#DIST_N=uniform_0.0_0.50
# ------------------------------------------------------------------------------ #
#DIST_B=constanteps_1.00_0.5_filter
#DIST_N=uniform_0.0_1.00
# ------------------------------------------------------------------------------ #
#DIST_B=constanteps_1.50_0.5_filter
#DIST_N=uniform_0.0_1.50
# ------------------------------------------------------------------------------ #
#DIST_B=uniformeps_0.0_1.50_0.75_filter
#DIST_N=uniform_0.0_1.50
# ------------------------------------------------------------------------------ #
#DIST_B=uniformeps_0.0_1.50_0.9_filter
#DIST_N=uniform_0.0_1.50
# ------------------------------------------------------------------------------ #


# Act 0.1 case with examples of noise levels. Adjust DIST_B, DIST_N.

ENV=Walker2d-v3
T_SOURCE=walker2d_td3_act0-1_s50

DIST_B=uniformeps_0.0_1.50_0.9_filter
DIST_N=uniform_0.0_1.50
for N_ALPHA in 1.0 10.0 50.0 ; do
    NAME=walker2d_td3_offline_${DIST_B}_np_${N_ALPHA}
    B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST_B}-dtype-train.p
    N_PATH=walker2d_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-${DIST_N}-t-1000000-v-0200000.tar
    for SEED in 10 20 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  -np ${N_PATH}  -na ${N_ALPHA}
    done
done


# Act 0.5 case with examples of noise levels. Adjust DIST_B, DIST_N.

ENV=Walker2d-v3
T_SOURCE=walker2d_td3_act0-5_s50

DIST_B=uniformeps_0.0_1.50_0.9_filter
DIST_N=uniform_0.0_1.50
for N_ALPHA in 1.0 10.0 50.0 ; do
    NAME=walker2d_td3_offline_${DIST_B}_np_${N_ALPHA}
    B_PATH=walker2d_td3_act0-5/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST_B}-dtype-train.p
    N_PATH=walker2d_td3_act0-5/${T_SOURCE}/experiments/sigma_predictor-${DIST_N}-t-1000000-v-0200000.tar
    for SEED in 10 20 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  -np ${N_PATH}  -na ${N_ALPHA}
    done
done
