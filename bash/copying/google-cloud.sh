## gsutil -m cp -r halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s10/buffer/*constant*               gs://shoe-dog-bucket/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s10/buffer/
## gsutil -m cp -r halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s20/buffer/*constant*               gs://shoe-dog-bucket/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s20/buffer/
## gsutil -m cp -r halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s10/rollout_buffer_txts/*constant*  gs://shoe-dog-bucket/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s10/rollout_buffer_txts/
## gsutil -m cp -r halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s20/rollout_buffer_txts/*constant*  gs://shoe-dog-bucket/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s20/rollout_buffer_txts/

## gsutil -m cp -r hopper_td3_act0-1/hopper_td3_act0-1_s10/buffer/*constant*                         gs://shoe-dog-bucket/hopper_td3_act0-1/hopper_td3_act0-1_s10/buffer/
## gsutil -m cp -r hopper_td3_act0-1/hopper_td3_act0-1_s20/buffer/*constant*                         gs://shoe-dog-bucket/hopper_td3_act0-1/hopper_td3_act0-1_s20/buffer/
## gsutil -m cp -r hopper_td3_act0-1/hopper_td3_act0-1_s10/rollout_buffer_txts/*constant*            gs://shoe-dog-bucket/hopper_td3_act0-1/hopper_td3_act0-1_s10/rollout_buffer_txts/
## gsutil -m cp -r hopper_td3_act0-1/hopper_td3_act0-1_s20/rollout_buffer_txts/*constant*            gs://shoe-dog-bucket/hopper_td3_act0-1/hopper_td3_act0-1_s20/rollout_buffer_txts/

## gsutil -m cp -r walker2d_td3_act0-1/walker2d_td3_act0-1_s10/buffer/*constant*                     gs://shoe-dog-bucket/walker2d_td3_act0-1/walker2d_td3_act0-1_s10/buffer/
## gsutil -m cp -r walker2d_td3_act0-1/walker2d_td3_act0-1_s20/buffer/*constant*                     gs://shoe-dog-bucket/walker2d_td3_act0-1/walker2d_td3_act0-1_s20/buffer/
## gsutil -m cp -r walker2d_td3_act0-1/walker2d_td3_act0-1_s10/rollout_buffer_txts/*constant*        gs://shoe-dog-bucket/walker2d_td3_act0-1/walker2d_td3_act0-1_s10/rollout_buffer_txts/
## gsutil -m cp -r walker2d_td3_act0-1/walker2d_td3_act0-1_s20/rollout_buffer_txts/*constant*        gs://shoe-dog-bucket/walker2d_td3_act0-1/walker2d_td3_act0-1_s20/rollout_buffer_txts/


## # ---------------------------------------------------------------------------------------------------- #
## # (Jan 23) copying over new constanteps buffers. In general, follow this pattern for uploading to GCP.
## # ---------------------------------------------------------------------------------------------------- #
## KEY=uniformeps_0.0_1.50_0.9
## HEAD=/data/spinup/data
##
## gsutil -m cp -r ${HEAD}/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/buffer/*${KEY}*  gs://shoe-dog-bucket/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/buffer/
## gsutil -m cp -r ${HEAD}/halfcheetah_td3_act0-5/halfcheetah_td3_act0-5_s40/buffer/*${KEY}*  gs://shoe-dog-bucket/halfcheetah_td3_act0-5/halfcheetah_td3_act0-5_s40/buffer/
##
## gsutil -m cp -r ${HEAD}/hopper_td3_act0-1/hopper_td3_act0-1_s40/buffer/*${KEY}*            gs://shoe-dog-bucket/hopper_td3_act0-1/hopper_td3_act0-1_s40/buffer/
## gsutil -m cp -r ${HEAD}/hopper_td3_act0-5/hopper_td3_act0-5_s40/buffer/*${KEY}*            gs://shoe-dog-bucket/hopper_td3_act0-5/hopper_td3_act0-5_s40/buffer/
##
## gsutil -m cp -r ${HEAD}/walker2d_td3_act0-1/walker2d_td3_act0-1_s50/buffer/*${KEY}*        gs://shoe-dog-bucket/walker2d_td3_act0-1/walker2d_td3_act0-1_s50/buffer/
## gsutil -m cp -r ${HEAD}/walker2d_td3_act0-5/walker2d_td3_act0-5_s50/buffer/*${KEY}*        gs://shoe-dog-bucket/walker2d_td3_act0-5/walker2d_td3_act0-5_s50/buffer/

# ---------------------------------------------------------------------------------------------------- #
# (March 14) Copying the updated time predictor, seed 02. See: https://github.com/CannyLab/spinningup/pull/25
# ---------------------------------------------------------------------------------------------------- #
KEY=seed-02_data-aug
HEAD=/data/spinup/data

gsutil -m cp -r ${HEAD}/ant_td3_act0-1/ant_td3_act0-1_s50/experiments/*${KEY}*                  gs://shoe-dog-bucket/ant_td3_act0-1/ant_td3_act0-1_s50/experiments/
gsutil -m cp -r ${HEAD}/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/experiments/*${KEY}*  gs://shoe-dog-bucket/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/experiments/
gsutil -m cp -r ${HEAD}/hopper_td3_act0-1/hopper_td3_act0-1_s40/experiments/*${KEY}*            gs://shoe-dog-bucket/hopper_td3_act0-1/hopper_td3_act0-1_s40/experiments/
gsutil -m cp -r ${HEAD}/walker2d_td3_act0-1/walker2d_td3_act0-1_s50/experiments/*${KEY}*        gs://shoe-dog-bucket/walker2d_td3_act0-1/walker2d_td3_act0-1_s50/experiments/
