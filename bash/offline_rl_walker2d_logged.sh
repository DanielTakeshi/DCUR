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

# ------------------------------------------------------------------------------ #
# March 03, 2021. Final buffer.
# ------------------------------------------------------------------------------ #
PREV=1000000
NEXT=1000000

NAME=walker2d_td3_offline_curriculum_${TYPE}_p_${PREV}_n_${NEXT}
for SEED in 30 31 32 33 34 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT}
done

for N_ALPHA in 1 10 30 50 100 ; do
    NAME=walker2d_td3_offline_curriculum_${TYPE}_p_${PREV}_n_${NEXT}_tp_${N_ALPHA}
    for SEED in 30 31 32 33 34 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT} \
            --n_alpha ${N_ALPHA}  -tp ${T_PATH}
    done
done

# ------------------------------------------------------------------------------ #
# March 05, 2021. Concurrent.
# ------------------------------------------------------------------------------ #
C_SCALE=1.00

NAME=walker2d_td3_offline_curriculum_${TYPE}_scale_${C_SCALE}t
for SEED in 30 31 32 33 34 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_scale ${C_SCALE}
done

for N_ALPHA in 1 10 30 50 100 ; do
    NAME=walker2d_td3_offline_curriculum_${TYPE}_scale_${C_SCALE}t_tp_${N_ALPHA}
    for SEED in 30 31 32 33 34 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_scale ${C_SCALE} \
            --n_alpha ${N_ALPHA}  -tp ${T_PATH}
    done
done

# ------------------------------------------------------------------------------ #
# March 14, 2021. New time predictor.
# ------------------------------------------------------------------------------ #
T_PATH=walker2d_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-02_data-aug.tar

for N_ALPHA in 1 10 30 50 100 ; do
    NAME=walker2d_td3_offline_curriculum_${TYPE}_p_${PREV}_n_${NEXT}_data-aug_tp_${N_ALPHA}
    for SEED in 30 31 32 33 34 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT} \
            --n_alpha ${N_ALPHA}  -tp ${T_PATH}
    done
done
