# ------------------------------------------------------------------------------ #
# Offline RL, with the noise predictor, add '_np' to the end to NAME.
# ------------------------------------------------------------------------------ #
# This would be easier if we could have a scalable way of referring to the DATA
# and noise predictor files. Alas, for now we put in checks in the script.
# ------------------------------------------------------------------------------ #
CPU1=0
CPU2=10

# For the uniform noise case, pick one:
DIST=uniform_0.0_0.25
DIST=uniform_0.0_0.50
DIST=uniform_0.0_0.75
DIST=uniform_0.0_1.00
DIST=uniform_0.0_1.25

# Main new things: N_PATH for noise path, N_ALPHA for alpha strength of reward shaping.
N_ALPHA=10.0
ENV=HalfCheetah-v3
NAME=halfcheetah_td3_offline_${DIST}_np_${N_ALPHA}
T_SOURCE=halfcheetah_td3_act0-1_s10
B_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/buffer/rollout-maxsize-1000000-steps-1000000-noise-${DIST}-dtype-train.p
N_PATH=halfcheetah_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-${DIST}-t-1000000-v-0500000.tar

# These seeds are for the STUDENT, with a FIXED teacher.
for SEED in 10 20 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/offline_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  -np ${N_PATH}  -na ${N_ALPHA}
done
