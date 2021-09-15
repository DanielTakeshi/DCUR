# ------------------------------------------------------------------------------ #
# Copy noise predictor results from one machine to another.
# ------------------------------------------------------------------------------ #

HEAD1=/data/seita/spinup/data
HEAD2=/data/spinup/data
SEED=10

# Lowland to my machine.
#scp -r seita@lowland.cs.berkeley.edu:${HEAD1}/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s${SEED}/experiments/     ${HEAD2}/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s${SEED}/
#scp -r seita@lowland.cs.berkeley.edu:${HEAD1}/hopper_td3_act0-1/hopper_td3_act0-1_s${SEED}/experiments/               ${HEAD2}/hopper_td3_act0-1/hopper_td3_act0-1_s${SEED}/
#scp -r seita@lowland.cs.berkeley.edu:${HEAD1}/walker2d_td3_act0-1/walker2d_td3_act0-1_s${SEED}/experiments/           ${HEAD2}/walker2d_td3_act0-1/walker2d_td3_act0-1_s${SEED}/

# My machine to mason.
scp -r ${HEAD2}/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s${SEED}/experiments/  seita@128.32.111.77:${HEAD1}/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s${SEED}/
scp -r ${HEAD2}/halfcheetah_td3_act0-5/halfcheetah_td3_act0-5_s${SEED}/experiments/  seita@128.32.111.77:${HEAD1}/halfcheetah_td3_act0-5/halfcheetah_td3_act0-5_s${SEED}/
scp -r ${HEAD2}/hopper_td3_act0-1/hopper_td3_act0-1_s${SEED}/experiments/            seita@128.32.111.77:${HEAD1}/hopper_td3_act0-1/hopper_td3_act0-1_s${SEED}/
scp -r ${HEAD2}/hopper_td3_act0-5/hopper_td3_act0-5_s${SEED}/experiments/            seita@128.32.111.77:${HEAD1}/hopper_td3_act0-5/hopper_td3_act0-5_s${SEED}/
scp -r ${HEAD2}/walker2d_td3_act0-1/walker2d_td3_act0-1_s${SEED}/experiments/        seita@128.32.111.77:${HEAD1}/walker2d_td3_act0-1/walker2d_td3_act0-1_s${SEED}/
scp -r ${HEAD2}/walker2d_td3_act0-5/walker2d_td3_act0-5_s${SEED}/experiments/        seita@128.32.111.77:${HEAD1}/walker2d_td3_act0-5/walker2d_td3_act0-5_s${SEED}/
