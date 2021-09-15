# ------------------------------------------------------------------------------ #
# Offline RL, large scale experiments. Time predictor reward shaping (seed 02).
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=12
ENV=Walker2d-v3
TYPE=logged
T_SOURCE=walker2d_td3_act0-1_s50
B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
T_PATH=walker2d_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-02.tar
EPOCHS=2500

# ------------------------------------------------------------------------------ #
# March 10, 2021. Final buffer, more epochs. (Re-doing April 14 w/correct PREV)
# ------------------------------------------------------------------------------ #
PREV=50000000
NEXT=1000000

NAME=walker2d_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}
for SEED in 30 31 32 33 34 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT}  --epochs ${EPOCHS}
done

## for N_ALPHA in 1 10 30 50 100 ; do
##     NAME=walker2d_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}_tp_${N_ALPHA}
##     for SEED in 30 31 32 33 34 ; do
##         taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##             --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT} \
##             --n_alpha ${N_ALPHA}  -tp ${T_PATH}  --epochs ${EPOCHS}
##     done
## done

## # ------------------------------------------------------------------------------ #
## # March 10, 2021. Concurrent, more epochs.
## # ------------------------------------------------------------------------------ #
## C_SCALE=1.00
##
## NAME=walker2d_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_scale_${C_SCALE}t
## for SEED in 30 31 32 33 34 ; do
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##         --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_scale ${C_SCALE}  --epochs ${EPOCHS}
## done
##
## for N_ALPHA in 1 10 30 50 100 ; do
##     NAME=walker2d_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_scale_${C_SCALE}t_tp_${N_ALPHA}
##     for SEED in 30 31 32 33 34 ; do
##         taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##             --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_scale ${C_SCALE} \
##             --n_alpha ${N_ALPHA}  -tp ${T_PATH}  --epochs ${EPOCHS}
##     done
## done
##
## # ------------------------------------------------------------------------------ #
## # March 14, 2021. More epochs, with new time predictor.
## # ------------------------------------------------------------------------------ #
## T_PATH=walker2d_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-02_data-aug.tar
##
## for N_ALPHA in 0 ; do
##     NAME=walker2d_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}_data-aug_tp_${N_ALPHA}
##     for SEED in 60 ; do
##         taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##             --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT} \
##             --n_alpha ${N_ALPHA}  -tp ${T_PATH}  --epochs ${EPOCHS}
##     done
## done
##
## # ------------------------------------------------------------------------------ #
## # April 14, 2021. Using special seed for saving student data. Use NEXT=0 for concurrent.
## # ------------------------------------------------------------------------------ #
## PREV=50000000
## NEXT=1000000
## for N_ALPHA in 0 ; do
##     NAME=walker2d_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}_tp_${N_ALPHA}
##     for SEED in 60 ; do
##         taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##             --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT} \
##             --n_alpha ${N_ALPHA}  -tp ${T_PATH}  --epochs ${EPOCHS}
##     done
## done
## T_PATH=walker2d_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-02_data-aug.tar
## for N_ALPHA in 0 ; do
##     NAME=walker2d_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}_data-aug_tp_${N_ALPHA}
##     for SEED in 60 ; do
##         taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##             --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT} \
##             --n_alpha ${N_ALPHA}  -tp ${T_PATH}  --epochs ${EPOCHS}
##     done
## done
