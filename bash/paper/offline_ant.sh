# ------------------------------------------------------------------------------ #
# Offline RL, for curriculum experiments. CoRL 2021 plan.
# https://github.com/CannyLab/spinningup/pull/32
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=9
ENV=Ant-v3
TYPE=logged
T_SOURCE=ant_td3_act0-1_s50
B_PATH=ant_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
T_PATH=ant_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-02.tar
EPOCHS=250

# ------------------------------------------------------------------------------ #
# (May 28) Final buffer. Different choices for the additive curricula, going up to final buffer.
# NOTE: don't have the ones that decrease PREV (yet), want to find best ones here first.
# ------------------------------------------------------------------------------ #

PREV=1000000
for NEXT in 50000 100000 200000 500000 1000000 ; do
    NAME=ant_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}_overlap
    for SEED in 90 91 92 93 94 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE} \
            --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT}  --epochs ${EPOCHS}  --overlap
    done
done

# ------------------------------------------------------------------------------ #
# (May 28) Different scaling curricula choices, including the concurrent case.
# ------------------------------------------------------------------------------ #

for C_SCALE in 0.50 0.75 1.00 1.10 ; do
    NAME=ant_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_scale_${C_SCALE}t_overlap
    for SEED in 90 91 92 93 94 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE} \
            --curriculum ${TYPE}  --c_scale ${C_SCALE}  --epochs ${EPOCHS}  --overlap
    done
done
