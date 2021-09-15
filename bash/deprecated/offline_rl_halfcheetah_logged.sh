# ------------------------------------------------------------------------------ #
# Offline RL -- handle the final buffer and concurrent cases here.
# Using this for s40 (or s50 for Walker) teachers as of January 25, 2021.
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=8
ENV=HalfCheetah-v3

# ------------------------------------------------------------------------------ #
# Jan 25. Simple logged data. These are for final buffer. Add --concurrent if desired.
# ------------------------------------------------------------------------------ #

## NAME=halfcheetah_td3_offline_finalb
## T_SOURCE=halfcheetah_td3_act0-1_s40
## B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
## for SEED in 10 20 ; do
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##         --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}
## done
##
## NAME=halfcheetah_td3_offline_finalb
## T_SOURCE=halfcheetah_td3_act0-5_s40
## B_PATH=halfcheetah_td3_act0-5/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.5.p
## for SEED in 10 20 ; do
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##         --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}
## done

# ------------------------------------------------------------------------------ #
# Feb 07. Now using a ground truth reward shaper. Add --concurrent if desired,
# but I think we want to stick with 'final buffer'. [Update: will want to re-run with
# the `nalpha` term for n=10 now, I didn't have it in the args before.]
# ------------------------------------------------------------------------------ #

## T_SOURCE=halfcheetah_td3_act0-1_s40
## B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
## for N_ALPHA in 1.0 10.0 ; do
##     NAME=halfcheetah_td3_offline_finalb_gtrewshape_${N_ALPHA}
##     for SEED in 10 20 ; do
##         taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##             --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  -na ${N_ALPHA}  --gt_shaping_logged
##     done
## done
##
## T_SOURCE=halfcheetah_td3_act0-5_s40
## B_PATH=halfcheetah_td3_act0-5/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.5.p
## for N_ALPHA in 1.0 10.0 ; do
##     NAME=halfcheetah_td3_offline_finalb_gtrewshape_${N_ALPHA}
##     for SEED in 10 20 ; do
##         taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##             --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  -na ${N_ALPHA}  --gt_shaping_logged
##     done
## done

# ------------------------------------------------------------------------------ #
# Feb 12. Now use the learned reward predictor instead of 'ground truth'.
# Add --concurrent if desired, but I think we want to stick with 'final buffer'.
# ------------------------------------------------------------------------------ #

T_SOURCE=halfcheetah_td3_act0-1_s40
B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
T_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-01.tar
for N_ALPHA in 1.0 10.0 ; do
    NAME=halfcheetah_td3_offline_finalb_np_${N_ALPHA}
    for SEED in 10 20 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  -tp ${T_PATH}  -na ${N_ALPHA}
    done
    NAME=halfcheetah_td3_offline_concurrent_np_${N_ALPHA}
    for SEED in 10 20 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  -tp ${T_PATH}  -na ${N_ALPHA}  --concurrent
    done
done

T_SOURCE=halfcheetah_td3_act0-5_s40
B_PATH=halfcheetah_td3_act0-5/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.5.p
T_PATH=halfcheetah_td3_act0-5/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-01.tar
for N_ALPHA in 1.0 10.0 ; do
    NAME=halfcheetah_td3_offline_finalb_np_${N_ALPHA}
    for SEED in 10 20 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  -tp ${T_PATH}  -na ${N_ALPHA}
    done
    NAME=halfcheetah_td3_offline_concurrent_np_${N_ALPHA}
    for SEED in 10 20 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  -tp ${T_PATH}  -na ${N_ALPHA}  --concurrent
    done
done
