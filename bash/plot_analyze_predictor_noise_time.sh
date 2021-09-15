# -------------------------------------------------------------------------------- #
# Analyze the predictor. Happens after `bash/train_predictor_{time,sigma,vareps}.sh`
# So instead of just plotting the losses and histograms, let's try stuff like rolling
# out the policy so that we can inspect.
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Feb 26, 2021. Seed 2, 80 epochs. Doing some deeper analysis, loading policies.
# -------------------------------------------------------------------------------- #

## NET_SEED=2
## DIST=time_prediction
##
## # TD3
## python spinup/teaching/plot_noise_predictor.py  ant_td3_act0-1/ant_td3_act0-1_s50/                  --noise ${DIST}  --seed ${NET_SEED}  --load_policies
## python spinup/teaching/plot_noise_predictor.py  halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/  --noise ${DIST}  --seed ${NET_SEED}  --load_policies
## python spinup/teaching/plot_noise_predictor.py  hopper_td3_act0-1/hopper_td3_act0-1_s40/            --noise ${DIST}  --seed ${NET_SEED}  --load_policies
## python spinup/teaching/plot_noise_predictor.py  walker2d_td3_act0-1/walker2d_td3_act0-1_s50/        --noise ${DIST}  --seed ${NET_SEED}  --load_policies
##
## # SAC
## python spinup/teaching/plot_noise_predictor.py  ant_sac_alpha0-2/ant_sac_alpha0-2_s50/                                      --noise ${DIST}  --seed ${NET_SEED}  --load_policies
## python spinup/teaching/plot_noise_predictor.py  halfcheetah_sac_alpha0-2_fix_alpha/halfcheetah_sac_alpha0-2_fix_alpha_s40   --noise ${DIST}  --seed ${NET_SEED}  --load_policies
## python spinup/teaching/plot_noise_predictor.py  hopper_sac_alpha0-2_fix_alpha/hopper_sac_alpha0-2_fix_alpha_s40             --noise ${DIST}  --seed ${NET_SEED}  --load_policies
## python spinup/teaching/plot_noise_predictor.py  walker2d_sac_alpha0-2_fix_alpha/walker2d_sac_alpha0-2_fix_alpha_s50         --noise ${DIST}  --seed ${NET_SEED}  --load_policies


## # -------------------------------------------------------------------------------- #
## # March 01, 2021. Now doing some histograms of predictions.
## # -------------------------------------------------------------------------------- #
##
## NET_SEED=2
## DIST=time_prediction
##
## # TD3
## python spinup/teaching/plot_noise_predictor.py  ant_td3_act0-1/ant_td3_act0-1_s50/                  --noise ${DIST}  --seed ${NET_SEED}  --predict_buffer
## python spinup/teaching/plot_noise_predictor.py  halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/  --noise ${DIST}  --seed ${NET_SEED}  --predict_buffer
## python spinup/teaching/plot_noise_predictor.py  hopper_td3_act0-1/hopper_td3_act0-1_s40/            --noise ${DIST}  --seed ${NET_SEED}  --predict_buffer
## python spinup/teaching/plot_noise_predictor.py  walker2d_td3_act0-1/walker2d_td3_act0-1_s50/        --noise ${DIST}  --seed ${NET_SEED}  --predict_buffer
##
## # SAC
## python spinup/teaching/plot_noise_predictor.py  ant_sac_alpha0-2/ant_sac_alpha0-2_s50/                                      --noise ${DIST}  --seed ${NET_SEED}  --predict_buffer
## python spinup/teaching/plot_noise_predictor.py  halfcheetah_sac_alpha0-2_fix_alpha/halfcheetah_sac_alpha0-2_fix_alpha_s40   --noise ${DIST}  --seed ${NET_SEED}  --predict_buffer
## python spinup/teaching/plot_noise_predictor.py  hopper_sac_alpha0-2_fix_alpha/hopper_sac_alpha0-2_fix_alpha_s40             --noise ${DIST}  --seed ${NET_SEED}  --predict_buffer
## python spinup/teaching/plot_noise_predictor.py  walker2d_sac_alpha0-2_fix_alpha/walker2d_sac_alpha0-2_fix_alpha_s50         --noise ${DIST}  --seed ${NET_SEED}  --predict_buffer


# -------------------------------------------------------------------------------- #
# March 01, 2021. Also plotting the rewards from the buffer.
# -------------------------------------------------------------------------------- #

## NET_SEED=2
## DIST=time_prediction
##
## # TD3
## python spinup/teaching/plot_noise_predictor.py  ant_td3_act0-1/ant_td3_act0-1_s50/                  --noise ${DIST}  --seed ${NET_SEED}  --check_rew
## python spinup/teaching/plot_noise_predictor.py  halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/  --noise ${DIST}  --seed ${NET_SEED}  --check_rew
## python spinup/teaching/plot_noise_predictor.py  hopper_td3_act0-1/hopper_td3_act0-1_s40/            --noise ${DIST}  --seed ${NET_SEED}  --check_rew
## python spinup/teaching/plot_noise_predictor.py  walker2d_td3_act0-1/walker2d_td3_act0-1_s50/        --noise ${DIST}  --seed ${NET_SEED}  --check_rew
##
## # SAC (edit: ah, Mandi didn't have the Rew statistics saved for the SAC runs).
## python spinup/teaching/plot_noise_predictor.py  ant_sac_alpha0-2/ant_sac_alpha0-2_s50/                                      --noise ${DIST}  --seed ${NET_SEED}  --check_rew
## python spinup/teaching/plot_noise_predictor.py  halfcheetah_sac_alpha0-2_fix_alpha/halfcheetah_sac_alpha0-2_fix_alpha_s40   --noise ${DIST}  --seed ${NET_SEED}  --check_rew
## python spinup/teaching/plot_noise_predictor.py  hopper_sac_alpha0-2_fix_alpha/hopper_sac_alpha0-2_fix_alpha_s40             --noise ${DIST}  --seed ${NET_SEED}  --check_rew
## python spinup/teaching/plot_noise_predictor.py  walker2d_sac_alpha0-2_fix_alpha/walker2d_sac_alpha0-2_fix_alpha_s50         --noise ${DIST}  --seed ${NET_SEED}  --check_rew


# -------------------------------------------------------------------------------- #
# April 06, 2021. https://github.com/CannyLab/spinningup/pull/28
# Show predictions on teacher data as a function of time, and compare with student.
# Plus data augmented version.
# -------------------------------------------------------------------------------- #

NET_SEED=2
DIST=time_prediction
python spinup/teaching/plot_noise_predictor.py ant_td3_act0-1/ant_td3_act0-1_s50/                  --noise ${DIST} --seed ${NET_SEED} --predict_buffer_raw
python spinup/teaching/plot_noise_predictor.py halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/  --noise ${DIST} --seed ${NET_SEED} --predict_buffer_raw
python spinup/teaching/plot_noise_predictor.py hopper_td3_act0-1/hopper_td3_act0-1_s40/            --noise ${DIST} --seed ${NET_SEED} --predict_buffer_raw
python spinup/teaching/plot_noise_predictor.py walker2d_td3_act0-1/walker2d_td3_act0-1_s50/        --noise ${DIST} --seed ${NET_SEED} --predict_buffer_raw
python spinup/teaching/plot_noise_predictor.py ant_td3_act0-1/ant_td3_act0-1_s50/                  --noise ${DIST} --seed ${NET_SEED} --predict_buffer_raw --data_aug
python spinup/teaching/plot_noise_predictor.py halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/  --noise ${DIST} --seed ${NET_SEED} --predict_buffer_raw --data_aug
python spinup/teaching/plot_noise_predictor.py hopper_td3_act0-1/hopper_td3_act0-1_s40/            --noise ${DIST} --seed ${NET_SEED} --predict_buffer_raw --data_aug
python spinup/teaching/plot_noise_predictor.py walker2d_td3_act0-1/walker2d_td3_act0-1_s50/        --noise ${DIST} --seed ${NET_SEED} --predict_buffer_raw --data_aug
