# --------------------------------------------------------------------------------------------- #
# Now with online student samples. We can still try the time predictor reward shaping (seed 02).
# Older results in https://github.com/CannyLab/spinningup/pull/30 but I want this in the paper.
# --------------------------------------------------------------------------------------------- #
CPU1=0
CPU2=9
ENV=Walker2d-v3
TYPE=logged
T_SOURCE=walker2d_td3_act0-1_s50
B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p

# If adjusting this, need to adjust NAME to include `data-aug_tp`
#T_PATH=walker2d_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-02.tar
#T_PATH=walker2d_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-02_data-aug.tar

# 250 epochs and 50K total means we're getting 1/20-th of the samples for a normal training run.
EPOCHS=250
STUDENT_SIZE=50000

# --------------------------------------------------------------------------------------------- #
# Final buffer.
# --------------------------------------------------------------------------------------------- #
PREV=1000000
NEXT=1000000

NAME=walker2d_td3_online_stud_total_${STUDENT_SIZE}_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}_overlap_data
for SEED in 90 91 92 93 94 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/online_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE} \
        --c_prev ${PREV}  --c_next ${NEXT}  --epochs ${EPOCHS}  --student_size ${STUDENT_SIZE}  --overlap
done

# --------------------------------------------------------------------------------------------- #
# Concurrent.
# --------------------------------------------------------------------------------------------- #
C_SCALE=1.00

NAME=walker2d_td3_online_stud_total_${STUDENT_SIZE}_curriculum_ep_${EPOCHS}_${TYPE}_scale_${C_SCALE}t_overlap
for SEED in 90 91 92 93 94 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/online_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE} \
        --c_scale ${C_SCALE}  --epochs ${EPOCHS}  --student_size ${STUDENT_SIZE}  --overlap
done
