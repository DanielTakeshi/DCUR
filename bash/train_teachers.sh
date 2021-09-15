# ---------------------------------------------------------------------------------- #
# These are the TD3 teachers that I train. We should also use `--act_noise` with the
# `--final_buffer`, might as well handle that here.
# Can use `--seed 0 10 20` etc., with a single space as separators, for seeds.
# As of Feb 22, for teachers, using seed 40 for HalfCheetah/Hopper, seed 50 for Ant/Walker2d.
# ---------------------------------------------------------------------------------- #
CPU1=0
CPU2=14

# Change BOTH together. Valid envs: Ant-v3, HalfCheetah-v3, Walker2d-v3, Hopper-v3, etc.
# The name should have the algorithm used, e.g., `_td3`, to make it easier for us to remember.
ENV=HalfCheetah-v3
NAME=halfcheetah_td3
taskset -c ${CPU1}-${CPU2} python -m spinup.run td3 --epochs 250 --seed 40 --act_noise 0.1 0.5 --env ${ENV} --exp_name ${NAME} --final_buffer

ENV=Hopper-v3
NAME=hopper_td3
taskset -c ${CPU1}-${CPU2} python -m spinup.run td3 --epochs 250 --seed 40 --act_noise 0.1 0.5 --env ${ENV} --exp_name ${NAME} --final_buffer

ENV=Walker2d-v3
NAME=walker2d_td3
taskset -c ${CPU1}-${CPU2} python -m spinup.run td3 --epochs 250 --seed 50 --act_noise 0.1 0.5 --env ${ENV} --exp_name ${NAME} --final_buffer

# [Feb 21, 2021] For Ant I'm just doing seed 50, and act=0.1 (so I had to re-name the teacher files).
ENV=Ant-v3
NAME=ant_td3
taskset -c ${CPU1}-${CPU2} python -m spinup.run td3 --epochs 250 --seed 50 --act_noise 0.1     --env ${ENV} --exp_name ${NAME} --final_buffer

# ---------------------------------------------------------------------------------- #
# The below are SAC teachers that Mandi was training (as of Feb 20-22 ish).
# Variations Mandi tested: alpha={0.1, 0.2} and `--update_alpha`, new argument added
# to test tuning the alpha parameter to make it learnable as well.
# ---------------------------------------------------------------------------------- #
# Daniel note: use fixed alpha=0.2, do not use `--update_alpha`. That's the default
# for SAC on Spinningup. Perhaps do Ant at some point?
# ---------------------------------------------------------------------------------- #
CPU1=80
CPU2=155

ENV=HalfCheetah-v3
NAME=halfcheetah_sac
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 40 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.1 --update_alpha --final_buffer
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 40 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.2 --update_alpha --final_buffer
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 40 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.2                --final_buffer
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 40 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.1                --final_buffer

ENV=Hopper-v3
NAME=hopper_sac
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 40 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.1 --update_alpha --final_buffer
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 40 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.2 --update_alpha --final_buffer
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 40 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.2                --final_buffer
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 40 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.1                --final_buffer

ENV=Walker2d-v3
NAME=walker2d_sac
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 50 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.1 --update_alpha --final_buffer
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 50 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.2 --update_alpha --final_buffer
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 50 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.2                --final_buffer
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 50 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.1                --final_buffer

ENV=Ant-v3
NAME=ant_sac
taskset -c ${CPU1}-${CPU2} python -m spinup.run sac --epochs 250 --seed 50 --env ${ENV} --exp_name ${NAME} --act_noise 0 --alpha 0.2                --final_buffer
