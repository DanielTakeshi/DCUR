# ------------------------------------------------------------------------------ #
# Offline RL, for curriculum experiments. CoRL 2021 plan. SAC <--> TD3 now!
# https://github.com/CannyLab/spinningup/pull/32
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=9
ENV=Walker2d-v3
TYPE=logged
EPOCHS=250

## # ------------------------------------------------------------------------------ #
## # SAC to TD3. Final Buffer and Concurrent.
## # ------------------------------------------------------------------------------ #
## T_SOURCE=walker2d_sac_alpha0-2_fix_alpha_s50
## B_PATH=walker2d_sac_alpha0-2_fix_alpha/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0-alpha-0.2-updateAlpha-False.p
##
## PREV=1000000
## NEXT=1000000
## NAME=walker2d_sac_to_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}
## for SEED in 90 91 92 93 94 ; do
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##         --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE} \
##         --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT}  --epochs ${EPOCHS}
## done
##
## C_SCALE=1.00
## NAME=walker2d_sac_to_td3_offline_curriculum_ep_${EPOCHS}_${TYPE}_scale_${C_SCALE}t
## for SEED in 90 91 92 93 94 ; do
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
##         --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE} \
##         --curriculum ${TYPE}  --c_scale ${C_SCALE}  --epochs ${EPOCHS}
## done

# ------------------------------------------------------------------------------ #
# TD3 to SAC. Final Buffer and Concurrent.
# ------------------------------------------------------------------------------ #
T_SOURCE=walker2d_td3_act0-1_s50
B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p

PREV=1000000
NEXT=1000000
NAME=walker2d_td3_to_sac_offline_curriculum_ep_${EPOCHS}_${TYPE}_p_${PREV}_n_${NEXT}
for SEED in 90 91 92 93 94 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  --algorithm sac \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE} \
        --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT}  --epochs ${EPOCHS}
done

C_SCALE=1.00
NAME=walker2d_td3_to_sac_offline_curriculum_ep_${EPOCHS}_${TYPE}_scale_${C_SCALE}t
for SEED in 90 91 92 93 94 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  --algorithm sac \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE} \
        --curriculum ${TYPE}  --c_scale ${C_SCALE}  --epochs ${EPOCHS}
done
