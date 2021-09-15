# ------------------------------------------------------------------------------ #
# Offline RL, with the 'randact' data as of February 03, 2020.
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=8
ENV=HalfCheetah-v3

# ------------------------------------------------------------------------------ #
# [Feb 09, 2021] This is the better way to do things, IMO. It should sample xi/vareps
# from a uniform distribution. So that gets sampled instead of sigma from earlier.
# ------------------------------------------------------------------------------ #

## # Act 0.1
## T_SOURCE=halfcheetah_td3_act0-1_s40
##
## for DIST_B in nonaddunif_0.0_0.5_filter nonaddunif_0.0_1.0_filter ; do
##     NAME=halfcheetah_td3_offline_${DIST_B}
##     B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST_B}-dtype-train.p
##     for SEED in 10 20 ; do
##         taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##             --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}
##     done
## done
##
## # Act 0.5
## T_SOURCE=halfcheetah_td3_act0-5_s40
##
## for DIST_B in nonaddunif_0.0_0.5_filter nonaddunif_0.0_1.0_filter ; do
##     NAME=halfcheetah_td3_offline_${DIST_B}
##     B_PATH=halfcheetah_td3_act0-5/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST_B}-dtype-train.p
##     for SEED in 10 20 ; do
##         taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##             --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}
##     done
## done

# ------------------------------------------------------------------------------ #
# [Feb 12, 2021] Same as above except with reward shaping, this time it should be
# correct reward shaping that directly uses the correct network. Uses seed 01.
# ------------------------------------------------------------------------------ #

T_SOURCE=halfcheetah_td3_act0-1_s40
for DIST_B in nonaddunif_0.0_0.5_filter nonaddunif_0.0_1.0_filter ; do
    B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST_B}-dtype-train.p
    N_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-${DIST_B}-t-1000000-v-0200000-seed-01.tar
    for N_ALPHA in 1.0 10.0 50.0 ; do
        NAME=halfcheetah_td3_offline_${DIST_B}_np_${N_ALPHA}
        for SEED in 10 20 ; do
            taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
                --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  -np ${N_PATH}  -na ${N_ALPHA}
        done
    done
done

T_SOURCE=halfcheetah_td3_act0-5_s40
for DIST_B in nonaddunif_0.0_0.5_filter nonaddunif_0.0_1.0_filter ; do
    B_PATH=halfcheetah_td3_act0-5/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST_B}-dtype-train.p
    N_PATH=halfcheetah_td3_act0-5/${T_SOURCE}/experiments/sigma_predictor-${DIST_B}-t-1000000-v-0200000-seed-01.tar
    for N_ALPHA in 1.0 10.0 50.0 ; do
        NAME=halfcheetah_td3_offline_${DIST_B}_np_${N_ALPHA}
        B_PATH=halfcheetah_td3_act0-5/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST_B}-dtype-train.p
        for SEED in 10 20 ; do
            taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
                --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  -np ${N_PATH}  -na ${N_ALPHA}
        done
    done
done
