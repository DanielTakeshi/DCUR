# -------------------------------------------------------------------------------------------------------------- #
# This is another noise predictor, this time for time, i.e., indices. This is for LOGGED DATA ONLY.
# Should save PyTorch files so that we can later load for OfflineRL runs.
# We should just be extracting indices.
# -------------------------------------------------------------------------------------------------------------- #
# Updated [Feb 10 2021] started on the `nonaddunif` branch. Also have random seeds now.
# Updated [Feb 25 2021] now on master branch, also adding Ant as an option. Increase to 80 epochs.
# Updated [Feb 26 2021] add training for SAC teachers.
# Updated [Mar 10 2021] add training with data augmentation
# -------------------------------------------------------------------------------------------------------------- #
EP=80
SEED=2

# CPUs to expose to PyTorch.
CPU1=0
CPU2=9

## # TD3, act=0.1.
## for NAME in halfcheetah_td3_act0-1 hopper_td3_act0-1 ; do
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  ${NAME}/${NAME}_s40/  --epochs ${EP}  --seed ${SEED}  --time_prediction
## done
## for NAME in ant_td3_act0-1 walker2d_td3_act0-1 ; do
##     taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  ${NAME}/${NAME}_s50/  --epochs ${EP}  --seed ${SEED}  --time_prediction
## done

## # SAC, act=0.1.
## taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  ant_sac_alpha0-2/ant_sac_alpha0-2_s50 \
##     --epochs ${EP}  --seed ${SEED}  --time_prediction
## taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  halfcheetah_sac_alpha0-2_fix_alpha/halfcheetah_sac_alpha0-2_fix_alpha_s40/ \
##     --epochs ${EP}  --seed ${SEED}  --time_prediction
## taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  hopper_sac_alpha0-2_fix_alpha/hopper_sac_alpha0-2_fix_alpha_s40/ \
##     --epochs ${EP}  --seed ${SEED}  --time_prediction
## taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  walker2d_sac_alpha0-2_fix_alpha/walker2d_sac_alpha0-2_fix_alpha_s50/ \
##     --epochs ${EP}  --seed ${SEED}  --time_prediction

# TD3, act=0.1, data augmentation, with multivariate Gaussians, mean=0 (vector), std=2 (independently).
taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  ant_td3_act0-1/ant_td3_act0-1_s50/ \
    --epochs ${EP}  --seed ${SEED}  --time_prediction  --data_aug
taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/ \
    --epochs ${EP}  --seed ${SEED}  --time_prediction  --data_aug
taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  hopper_td3_act0-1/hopper_td3_act0-1_s40/ \
    --epochs ${EP}  --seed ${SEED}  --time_prediction  --data_aug
taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  walker2d_td3_act0-1/walker2d_td3_act0-1_s50/ \
    --epochs ${EP}  --seed ${SEED}  --time_prediction  --data_aug
