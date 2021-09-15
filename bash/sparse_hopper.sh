# --------------------------------------------------------------------------------------------- #
# Now with the sparse reward case with ONLY the time predictor there.
# Can do this with both online and offline samples -- just have to be careful not to use any of
# the real environment reward. We CAN do nalpha=0 but that'd be for checking horrible performance.
# --------------------------------------------------------------------------------------------- #
CPU1=0
CPU2=12
ENV=Hopper-v3
TYPE=logged
T_SOURCE=hopper_td3_act0-1_s40
B_PATH=hopper_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
T_PATH=hopper_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-02.tar

# If using this, should use `_data-aug_tp` in the experiment name.
#T_PATH=hopper_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-02_data-aug.tar

# 500 epochs and 100K total means it's getting 50K online for a 'normal' training run.
EPOCHS=500
STUDENT_SIZE=100000

# --------------------------------------------------------------------------------------------- #
# April 2021. Final buffer.
# --------------------------------------------------------------------------------------------- #
PREV=50000000
NEXT=1000000

for N_ALPHA in 1 ; do
    NAME=hopper_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}_overlap_sparse_tp_${N_ALPHA}
    for SEED in 30 31 32 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT} \
            --n_alpha ${N_ALPHA}  -tp ${T_PATH}  --epochs ${EPOCHS}  --overlap  --sparse
    done

    NAME=hopper_td3_online_stud_total_${STUDENT_SIZE}_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}_overlap_sparse_tp_${N_ALPHA}
    for SEED in 30 31 32 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/online_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT} \
            --n_alpha ${N_ALPHA}  -tp ${T_PATH}  --epochs ${EPOCHS}  --overlap  --sparse  --student_size ${STUDENT_SIZE}
    done
done

# --------------------------------------------------------------------------------------------- #
# April 2021. Concurrent
# --------------------------------------------------------------------------------------------- #
C_SCALE=1.00

for N_ALPHA in 1 ; do
    NAME=hopper_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_scale_${C_SCALE}t_overlap_sparse_tp_${N_ALPHA}
    for SEED in 30 31 32 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_scale ${C_SCALE} \
            --n_alpha ${N_ALPHA}  -tp ${T_PATH}  --epochs ${EPOCHS}  --overlap  --sparse
    done

    NAME=hopper_td3_online_stud_total_${STUDENT_SIZE}_curriculum_ep_${EPOCHS}_${TYPE}_scale_${C_SCALE}t_overlap_sparse_tp_${N_ALPHA}
    for SEED in 30 31 32 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/online_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_scale ${C_SCALE} \
            --n_alpha ${N_ALPHA}  -tp ${T_PATH}  --epochs ${EPOCHS}  --overlap  --sparse  --student_size ${STUDENT_SIZE}
    done
done
