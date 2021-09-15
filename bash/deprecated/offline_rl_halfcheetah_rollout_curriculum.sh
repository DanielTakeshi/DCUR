# ------------------------------------------------------------------------------ #
# Offline RL -- handle the _curriculum_ case, with rolled out data from s40 teacher for HalfCheetah.
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=8
ENV=HalfCheetah-v3

# ------------------------------------------------------------------------------ #
# [25 Feb 2021]. Curriculum case where we limit samples in one teacher based on an
# amount before and after the current time cursor `t`, bounded by [0, 1M) of course.
# In the limiting case, PREV=1e6 and NEXT=0 is the same as `--concurrent`.
# ------------------------------------------------------------------------------ #
PREV=1000000
NEXT=0
TYPE=noise_rollout
T_SOURCE=halfcheetah_td3_act0-1_s40

for DIST_B in uniform_0.0_1.50 uniformeps_0.0_1.50_0.9_filter nonaddunif_0.0_0.5_filter ; do
    B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST_B}-dtype-train.p
    NAME=halfcheetah_td3_offline_${DIST_B}_curriculum_${TYPE}_p_${PREV}_n_${NEXT}
    for SEED in 10 20 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT}
    done
done
