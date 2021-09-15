# ------------------------------------------------------------------------------ #
# Now let's use this to inspect the policy, so that we check how different noise
# levels will impact the cyclic nature of some environments? This loads a buffer.
#
# (Jan 20) All the below should produce reasonably informative plots.
# ------------------------------------------------------------------------------ #

## # HC, act_noise=0.1, with noise injected into the snapshot.
## ENV=HalfCheetah-v3
## T_SOURCE=halfcheetah_td3_act0-1_s10
##
## # Final Buffer
## B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
## python spinup/teaching/inspect_policy.py  --plot_buffer  \
##     --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE} --noise final_buffer
##
## # Rollouts.
## for DIST in constant_0.0 uniform_0.0_0.25 uniform_0.0_0.50 uniform_0.0_0.75 uniform_0.0_1.00 uniform_0.0_1.25 ; do
##     B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p
##     python spinup/teaching/inspect_policy.py  --plot_buffer  \
##         --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --noise ${DIST}
## done
## for DIST in uniformeps_0.0_0.25_0.5_filter uniformeps_0.0_0.50_0.5_filter uniformeps_0.0_0.75_0.5_filter uniformeps_0.0_1.00_0.5_filter uniformeps_0.0_1.25_0.5_filter ; do
##     B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p
##     python spinup/teaching/inspect_policy.py  --plot_buffer  \
##         --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --noise ${DIST}
## done
##
##
## # Hopper
## ENV=Hopper-v3
## T_SOURCE=hopper_td3_act0-1_s10
##
## # Final Buffer
## B_PATH=hopper_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
## python spinup/teaching/inspect_policy.py  --plot_buffer  \
##     --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE} --noise final_buffer
##
## # Rollouts.
## for DIST in constant_0.0 uniform_0.0_0.25 uniform_0.0_0.50 uniform_0.0_0.75 uniform_0.0_1.00 uniform_0.0_1.25 ; do
##     B_PATH=hopper_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p
##     python spinup/teaching/inspect_policy.py  --plot_buffer  \
##         --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --noise ${DIST}
## done
## for DIST in uniformeps_0.0_0.25_0.5_filter uniformeps_0.0_0.50_0.5_filter uniformeps_0.0_0.75_0.5_filter uniformeps_0.0_1.00_0.5_filter uniformeps_0.0_1.25_0.5_filter ; do
##     B_PATH=hopper_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p
##     python spinup/teaching/inspect_policy.py  --plot_buffer  \
##         --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --noise ${DIST}
## done
##
##
## # Walker2d
## ENV=Walker2d-v3
## T_SOURCE=walker2d_td3_act0-1_s10
##
## # Final Buffer
## B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
## python spinup/teaching/inspect_policy.py  --plot_buffer  \
##     --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE} --noise final_buffer
##
## # Rollouts.
## for DIST in constant_0.0 uniform_0.0_0.25 uniform_0.0_0.50 uniform_0.0_0.75 uniform_0.0_1.00 uniform_0.0_1.25 ; do
##     B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p
##     python spinup/teaching/inspect_policy.py  --plot_buffer  \
##         --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --noise ${DIST}
## done
## for DIST in uniformeps_0.0_0.25_0.5_filter uniformeps_0.0_0.50_0.5_filter uniformeps_0.0_0.75_0.5_filter uniformeps_0.0_1.00_0.5_filter uniformeps_0.0_1.25_0.5_filter ; do
##     B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p
##     python spinup/teaching/inspect_policy.py  --plot_buffer  \
##         --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --noise ${DIST}
## done


# ------------------------------------------------------------------------------ #
# Now let's use this to inspect the policy, so that we check how different noise
# levels will impact the cyclic nature of some environments? This loads a buffer.
#
# (Jan 27) Now using the updated teacher seeds, s40 and s50, since those have snapshots
# saved at epoch 0, which we can perhaps use for comparisons. I also have several
# cases of uniformeps here. Running on `constanteps` branch.
# ------------------------------------------------------------------------------ #

ENV=HalfCheetah-v3
T_SOURCE=halfcheetah_td3_act0-1_s40

# Final Buffer
B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
python spinup/teaching/inspect_policy.py  --plot_buffer  --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE} --noise final_buffer

# Rollouts.
for DIST in constanteps_0.00_0.0_filter uniformeps_0.0_1.50_0.5_filter uniformeps_0.0_1.50_0.75_filter ; do
    B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p
    python spinup/teaching/inspect_policy.py  --plot_buffer  --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --noise ${DIST}
done


ENV=Hopper-v3
T_SOURCE=hopper_td3_act0-1_s40

# Final Buffer
B_PATH=hopper_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
python spinup/teaching/inspect_policy.py  --plot_buffer  --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE} --noise final_buffer

# Rollouts.
for DIST in constanteps_0.00_0.0_filter uniformeps_0.0_1.50_0.5_filter uniformeps_0.0_1.50_0.75_filter ; do
    B_PATH=hopper_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p
    python spinup/teaching/inspect_policy.py  --plot_buffer  --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --noise ${DIST}
done


ENV=Walker2d-v3
T_SOURCE=walker2d_td3_act0-1_s50

# Final Buffer
B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
python spinup/teaching/inspect_policy.py  --plot_buffer  --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE} --noise final_buffer

# Rollouts.
for DIST in constanteps_0.00_0.0_filter uniformeps_0.0_1.50_0.5_filter uniformeps_0.0_1.50_0.75_filter ; do
    B_PATH=walker2d_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p
    python spinup/teaching/inspect_policy.py  --plot_buffer  --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --noise ${DIST}
done


# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
