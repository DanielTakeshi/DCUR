# ------------------------------------------------------------------------------ #
# Offline RL, for curriculum experiments. CoRL 2021 plan.
# https://github.com/CannyLab/spinningup/pull/32
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=9
ENV=HalfCheetah-v3
TYPE=logged
T_SOURCE=halfcheetah_td3_act0-1_s40
B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
T_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-02.tar
EPOCHS=2500

# ------------------------------------------------------------------------------ #
# (May 30) Final buffer, for more epochs. No overlap. PREV=50M, NEXT=1M.
# ------------------------------------------------------------------------------ #

PREV=50000000
NEXT=1000000
NAME=halfcheetah_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}
for SEED in 90 91 92 93 94 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE} \
        --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT}  --epochs ${EPOCHS}
done

# ------------------------------------------------------------------------------ #
# (May 30) Concurrent, for more epochs. No overlap.
# ------------------------------------------------------------------------------ #

C_SCALE=1.00
NAME=halfcheetah_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_scale_${C_SCALE}t
for SEED in 90 91 92 93 94 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE} \
        --curriculum ${TYPE}  --c_scale ${C_SCALE}  --epochs ${EPOCHS}
done
