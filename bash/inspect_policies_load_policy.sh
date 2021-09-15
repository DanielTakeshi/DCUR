# ------------------------------------------------------------------------------ #
# Now let's use this to inspect the policy, so that we check how different noise levels
# will impact the cyclic nature of some environments? This is for loading a policy.
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=10

# You can pick these distributions. If For the uniform noise case, pick one:
#DIST=uniform_0.0_0.25
#DIST=uniform_0.0_0.50
#DIST=uniform_0.0_0.75
#DIST=uniform_0.0_1.00
#DIST=uniform_0.0_1.25
DIST=constant_0.0

# HC, act_noise=0.1, with noise injected into the snapshot.
# ENV=HalfCheetah-v3
# NAME=halfcheetah_td3_offline_${DIST}
# T_SOURCE=halfcheetah_td3_act0-1_s10
# B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p

# Hopper
ENV=Hopper-v3
NAME=hopper_td3_offline_${DIST}
T_SOURCE=hopper_td3_act0-1_s10
B_PATH=hopper_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p

# Walker2d
#ENV=Walker2d-v3
#NAME=walker2d_td3_offline_${DIST}
#T_SOURCE=walker2d_td3_act0-1_s10
#B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p


# Load and plot on the fly.
taskset -c ${CPU1}-${CPU2} python spinup/teaching/inspect_policy.py \
    --plot_load_pol  --env ${ENV}  --exp_name ${NAME}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --noise ${DIST}
