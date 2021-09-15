# ------------------------------------------------------------------------------ #
# Offline RL, test overlap. Now add --overlap_analysis.
# https://github.com/CannyLab/spinningup/pull/26
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=12
ENV=Hopper-v3
TYPE=logged
T_SOURCE=hopper_td3_act0-1_s40
B_PATH=hopper_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
T_PATH=hopper_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-02.tar
EPOCHS=500

# ------------------------------------------------------------------------------ #
# Final buffer.
# ------------------------------------------------------------------------------ #
PREV=50000000
NEXT=1000000

NAME=hopper_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}_overlap
for SEED in 30 31 32 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT} \
        --epochs ${EPOCHS}  --overlap
done

# ------------------------------------------------------------------------------ #
# Concurrent.
# ------------------------------------------------------------------------------ #
C_SCALE=1.00

NAME=hopper_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_scale_${C_SCALE}t_overlap
for SEED in 30 31 32 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_scale ${C_SCALE} \
        --epochs ${EPOCHS}  --overlap
done
