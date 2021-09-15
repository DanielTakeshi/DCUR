# ------------------------------------------------------------------------------ #
# Now let's use this to inspect the policy, so that we check how different noise
# levels will impact the cyclic nature of some environments? This loads a buffer.
#
# (Jan 20) All the below should produce reasonably informative plots.
# ------------------------------------------------------------------------------ #


# HC, act_noise=0.1, with noise injected into the snapshot.
# ENV=HalfCheetah-v3
# for TEACHER in halfcheetah_td3_act0-1 ; do 
#     for SEED in 10 ; do
#         T_SOURCE=${TEACHER}_s${SEED}
#         B_PATH=${TEACHER}/${T_SOURCE}/buffer/
#         python spinup/teaching/inspect_actions.py --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}
#     done
# done

ENV=HalfCheetah-v3
for TEACHER in halfcheetah_td3_act0-1 halfcheetah_td3_act0-5 ; do 
    for SEED in 10 20 40 ; do
        T_SOURCE=${TEACHER}_s${SEED}
        B_PATH=${TEACHER}/${T_SOURCE}/buffer/
        python spinup/teaching/inspect_actions.py --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}
    done
done


# # Hopper
ENV=Hopper-v3
for TEACHER in hopper_td3_act0-1 hopper_td3_act0-5 ; do 
    for SEED in 10 20 40 ; do
        T_SOURCE=${TEACHER}_s${SEED}
        B_PATH=${TEACHER}/${T_SOURCE}/buffer/
        python spinup/teaching/inspect_actions.py --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}
    done
done

# # Walker2d
ENV=Walker2d-v3
for TEACHER in walker2d_td3_act0-1 walker2d_td3_act0-5 ; do 
    for SEED in 10 20 50 ; do # NOTE different seed
        T_SOURCE=${TEACHER}_s${SEED}
        B_PATH=${TEACHER}/${T_SOURCE}/buffer/
        python spinup/teaching/inspect_actions.py --env ${ENV}  -bp ${B_PATH}  --t_source ${T_SOURCE}
    done
done
