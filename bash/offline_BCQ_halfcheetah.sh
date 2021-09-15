# ------------------------------------------------------------------------------ #
# Offline RL! This is the 'naive' version where we just run a standard off the
# shelf DeepRL algorithm like TD3. No reward shaping involved.
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=10

# HC, act_noise=0.1. Both final buffer and then an example uniform one.
# ENV=HalfCheetah-v3
# NAME=halfcheetah_td3_offline_finalb
# T_SOURCE=halfcheetah_td3_act0-1_s10
# B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p

# For the uniform noise case, pick one:
DIST=uniform_0.0_0.25
DIST=uniform_0.0_0.50
DIST=uniform_0.0_0.75
DIST=uniform_0.0_1.00
DIST=uniform_0.0_1.25

DIST=uniformeps_0.0_0.25_0.5_filter
DIST=uniformeps_0.0_0.50_0.5_filter
DIST=uniformeps_0.0_0.75_0.5_filter
DIST=uniformeps_0.0_1.00_0.5_filter
DIST=uniformeps_0.0_1.25_0.5_filter

DIST=uniformeps_0.0_0.25_0.5_nofilt
DIST=uniformeps_0.0_0.50_0.5_nofilt
DIST=uniformeps_0.0_0.75_0.5_nofilt
DIST=uniformeps_0.0_1.00_0.5_nofilt
DIST=uniformeps_0.0_1.25_0.5_nofilt


# Remember, here there's no noise predictor. See the scripts with `_np` in their names.
ENV=HalfCheetah-v3
T_SOURCE=halfcheetah_td3_act0-1_s10

# These seeds are for the STUDENT, with a FIXED teacher.
for DIST in uniform_0.0_0.25 uniform_0.0_0.50 uniform_0.0_0.75 uniform_0.0_1.00 uniform_0.0_1.25 uniformeps_0.0_0.25_0.5_filter uniformeps_0.0_0.50_0.5_filter uniformeps_0.0_0.75_0.5_filter uniformeps_0.0_1.00_0.5_filter uniformeps_0.0_1.25_0.5_filter uniformeps_0.0_0.25_0.5_nofilt uniformeps_0.0_0.50_0.5_nofilt uniformeps_0.0_0.75_0.5_nofilt uniformeps_0.0_1.00_0.5_nofilt; do
    NAME=halfcheetah_BCQ_offline_${DIST}
    B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p
    for SEED in 10 20 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
            --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE} --algorithm BCQ
    done
done
