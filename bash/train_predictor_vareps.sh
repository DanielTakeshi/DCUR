# -------------------------------------------------------------------------------------------------------------- #
# This is another noise predictor, this time for "vareps" (or "xi") data. Happens after `bash/load_teachers.sh`.
# Should save PyTorch files so that we can later load for OfflineRL runs.
# NOTE: the "vareps" data is stored as the "std" in the buffers, even through it is not a standard deviation value!
# -------------------------------------------------------------------------------------------------------------- #
# Updated [Feb 10 2021] on the `nonaddunif` branch. Also have random seeds now.
# -------------------------------------------------------------------------------------------------------------- #
EP=50
SEED=1

# Data sizes.
T_SIZE=1000000
V_SIZE=200000

# CPUs to expose to PyTorch.
CPU1=0
CPU2=12


# --- Act 0.1 --- #
for DIST in nonaddunif_0.0_0.5_filter  nonaddunif_0.0_1.0_filter ; do
    for NAME in halfcheetah_td3_act0-1 hopper_td3_act0-1 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  ${NAME}/${NAME}_s40/  \
                -ts ${T_SIZE}  -vs ${V_SIZE}  --epochs ${EP}  --noise ${DIST}  --seed ${SEED}
    done
    NAME=walker2d_td3_act0-1
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  ${NAME}/${NAME}_s50/  \
            -ts ${T_SIZE}  -vs ${V_SIZE}  --epochs ${EP}  --noise ${DIST}  --seed ${SEED}
done


# --- Act 0.5 --- #
for DIST in nonaddunif_0.0_0.5_filter  nonaddunif_0.0_1.0_filter ; do
    for NAME in halfcheetah_td3_act0-5 hopper_td3_act0-5 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  ${NAME}/${NAME}_s40/  \
                -ts ${T_SIZE}  -vs ${V_SIZE}  --epochs ${EP}  --noise ${DIST}  --seed ${SEED}
    done
    NAME=walker2d_td3_act0-5
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  ${NAME}/${NAME}_s50/  \
            -ts ${T_SIZE}  -vs ${V_SIZE}  --epochs ${EP}  --noise ${DIST}  --seed ${SEED}
done
