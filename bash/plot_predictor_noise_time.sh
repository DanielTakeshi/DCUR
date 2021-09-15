# -------------------------------------------------------------------------------- #
# Plot the predictor. Happens after `bash/train_predictor_{time,sigma,vareps}.sh`
# -------------------------------------------------------------------------------- #

## # -------------------------------------------------------------------------------- #
## # Daniel: January 24, 2021. Updated with seeds 40 and 50.
## # -------------------------------------------------------------------------------- #
##
## SEED=40
## for DIST in uniform_0.0_0.25 uniform_0.0_0.50 uniform_0.0_1.00 uniform_0.0_1.50 ; do
##     for NAME in halfcheetah_td3 hopper_td3 ; do
##         python spinup/teaching/plot_noise_predictor.py  ${NAME}_act0-1/${NAME}_act0-1_s${SEED}/  --noise ${DIST}
##         python spinup/teaching/plot_noise_predictor.py  ${NAME}_act0-5/${NAME}_act0-5_s${SEED}/  --noise ${DIST}
##     done
## done
##
## SEED=50
## NAME=walker2d_td3
## for DIST in uniform_0.0_0.25 uniform_0.0_0.50 uniform_0.0_1.00 uniform_0.0_1.50 ; do
##     python spinup/teaching/plot_noise_predictor.py  ${NAME}_act0-1/${NAME}_act0-1_s${SEED}/  --noise ${DIST}
##     python spinup/teaching/plot_noise_predictor.py  ${NAME}_act0-5/${NAME}_act0-5_s${SEED}/  --noise ${DIST}
## done


## # -------------------------------------------------------------------------------- #
## # Daniel: February 11, 2021. Updated with ploting noise predictor for the 'time
## # predictor' and the 'xi/varepsilon' predictors. Now has NET_SEED as well.
## # -------------------------------------------------------------------------------- #
##
## NET_SEED=1
##
## T_SEED=40
## for DIST in time_prediction nonaddunif_0.0_0.5_filter nonaddunif_0.0_1.0_filter ; do
##     for NAME in halfcheetah_td3 hopper_td3 ; do
##         python spinup/teaching/plot_noise_predictor.py  ${NAME}_act0-1/${NAME}_act0-1_s${T_SEED}/  --noise ${DIST}  --seed ${NET_SEED}
##         python spinup/teaching/plot_noise_predictor.py  ${NAME}_act0-5/${NAME}_act0-5_s${T_SEED}/  --noise ${DIST}  --seed ${NET_SEED}
##     done
## done
##
## T_SEED=50
## NAME=walker2d_td3
## for DIST in time_prediction nonaddunif_0.0_0.5_filter nonaddunif_0.0_1.0_filter ; do
##     python spinup/teaching/plot_noise_predictor.py  ${NAME}_act0-1/${NAME}_act0-1_s${T_SEED}/  --noise ${DIST}  --seed ${NET_SEED}
##     python spinup/teaching/plot_noise_predictor.py  ${NAME}_act0-5/${NAME}_act0-5_s${T_SEED}/  --noise ${DIST}  --seed ${NET_SEED}
## done


## # -------------------------------------------------------------------------------- #
## # Feb 26, 2021. Seed 2, 80 epochs.
## # -------------------------------------------------------------------------------- #
##
## NET_SEED=2
## DIST=time_prediction
##
## # TD3 teachers
## python spinup/teaching/plot_noise_predictor.py  ant_td3_act0-1/ant_td3_act0-1_s50/                  --noise ${DIST}  --seed ${NET_SEED}
## python spinup/teaching/plot_noise_predictor.py  halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/  --noise ${DIST}  --seed ${NET_SEED}
## python spinup/teaching/plot_noise_predictor.py  hopper_td3_act0-1/hopper_td3_act0-1_s40/            --noise ${DIST}  --seed ${NET_SEED}
## python spinup/teaching/plot_noise_predictor.py  walker2d_td3_act0-1/walker2d_td3_act0-1_s50/        --noise ${DIST}  --seed ${NET_SEED}
##
## # SAC teachers
## python spinup/teaching/plot_noise_predictor.py  ant_sac_alpha0-2/ant_sac_alpha0-2_s50/                                      --noise ${DIST}  --seed ${NET_SEED}
## python spinup/teaching/plot_noise_predictor.py  halfcheetah_sac_alpha0-2_fix_alpha/halfcheetah_sac_alpha0-2_fix_alpha_s40   --noise ${DIST}  --seed ${NET_SEED}
## python spinup/teaching/plot_noise_predictor.py  hopper_sac_alpha0-2_fix_alpha/hopper_sac_alpha0-2_fix_alpha_s40             --noise ${DIST}  --seed ${NET_SEED}
## python spinup/teaching/plot_noise_predictor.py  walker2d_sac_alpha0-2_fix_alpha/walker2d_sac_alpha0-2_fix_alpha_s50         --noise ${DIST}  --seed ${NET_SEED}

# -------------------------------------------------------------------------------- #
# March 12, 2021. Seed 2, 80 epochs, data augmentataion. https://github.com/CannyLab/spinningup/pull/25
# -------------------------------------------------------------------------------- #

NET_SEED=2
DIST=time_prediction

# TD3 teachers
python spinup/teaching/plot_noise_predictor.py  ant_td3_act0-1/ant_td3_act0-1_s50/                  --noise ${DIST}  --seed ${NET_SEED}  --data_aug
python spinup/teaching/plot_noise_predictor.py  halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/  --noise ${DIST}  --seed ${NET_SEED}  --data_aug
python spinup/teaching/plot_noise_predictor.py  hopper_td3_act0-1/hopper_td3_act0-1_s40/            --noise ${DIST}  --seed ${NET_SEED}  --data_aug
python spinup/teaching/plot_noise_predictor.py  walker2d_td3_act0-1/walker2d_td3_act0-1_s50/        --noise ${DIST}  --seed ${NET_SEED}  --data_aug
