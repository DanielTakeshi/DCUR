# ----------------------------------------------------------------- #
# Train the noise predictor. Happens after `bash/load_teachers.sh`
# This should save PyTorch files so that we can later load for OfflineRL.
# ----------------------------------------------------------------- #
EP=50

# Data sizes.
T_SIZE=1000000
V_SIZE=200000

# CPUs to expose to PyTorch.
CPU1=0
CPU2=10

# Jan 23: new commands.
# First, action=0.1
for DIST in uniform_0.0_0.25  uniform_0.0_0.50  uniform_0.0_1.00  uniform_0.0_1.50  ; do
    for NAME in halfcheetah_td3_act0-1 hopper_td3_act0-1 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  ${NAME}/${NAME}_s40/  -ts ${T_SIZE}  -vs ${V_SIZE}  --epochs ${EP}  --noise ${DIST}
    done
done

for DIST in uniform_0.0_0.25  uniform_0.0_0.50  uniform_0.0_1.00  uniform_0.0_1.50  ; do
    NAME=walker2d_td3_act0-1
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  ${NAME}/${NAME}_s50/  -ts ${T_SIZE}  -vs ${V_SIZE}  --epochs ${EP}  --noise ${DIST}
done


# Second, action=0.1
for DIST in uniform_0.0_0.25  uniform_0.0_0.50  uniform_0.0_1.00  uniform_0.0_1.50  ; do
    for NAME in halfcheetah_td3_act0-5 hopper_td3_act0-5 ; do
        taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  ${NAME}/${NAME}_s40/  -ts ${T_SIZE}  -vs ${V_SIZE}  --epochs ${EP}  --noise ${DIST}
    done
done

for DIST in uniform_0.0_0.25  uniform_0.0_0.50  uniform_0.0_1.00  uniform_0.0_1.50  ; do
    NAME=walker2d_td3_act0-5
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/noise_predictor.py  ${NAME}/${NAME}_s50/  -ts ${T_SIZE}  -vs ${V_SIZE}  --epochs ${EP}  --noise ${DIST}
done
