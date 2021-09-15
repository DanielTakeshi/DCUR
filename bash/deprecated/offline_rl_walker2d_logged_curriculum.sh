# ------------------------------------------------------------------------------ #
# Offline RL -- handle the _curriculum_ case.
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=8
ENV=Walker2d-v3
TYPE=logged

# ------------------------------------------------------------------------------ #
# [19 Feb 2021]. Curriculum case where we limit samples in one teacher based on an
# amount before and after the current time cursor `t`, bounded by [0, 1M) of course.
# In the limiting case, PREV=1e6 and NEXT=0 is the same as `--concurrent`.
# ------------------------------------------------------------------------------ #
PREV=1000000
NEXT=50000
NAME=walker2d_td3_offline_curriculum_${TYPE}_p_${PREV}_n_${NEXT}
T_SOURCE=walker2d_td3_act0-1_s50
B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p

for SEED in 10 20 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT}
done

# ------------------------------------------------------------------------------ #
# [24 Feb 2021]. Curriculum case where we instead use a time scaling value, c_scale,
# to determine what to sample. Assume range is [0, t*c_scale] for now, fixed during
# training (later, vary it).
# ------------------------------------------------------------------------------ #

for C_SCALE in 0.50 0.75 ; do
    NAME=walker2d_td3_offline_curriculum_${TYPE}_scale_${C_SCALE}t
    T_SOURCE=walker2d_td3_act0-1_s50
    B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
    for SEED in 10 20 21 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_scale ${C_SCALE}
    done
done
