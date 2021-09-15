# --------------------------------------------------------------------------------------------- #
# Now with online student samples. We can still try the time predictor reward shaping (seed 02).
# --------------------------------------------------------------------------------------------- #
CPU1=0
CPU2=12
ENV=Ant-v3
TYPE=logged
T_SOURCE=ant_td3_act0-1_s50
B_PATH=ant_td3_act0-1/${T_SOURCE}/buffer/final_buffer-maxsize-1000000-steps-1000000-noise-0.1.p
T_PATH=ant_td3_act0-1/${T_SOURCE}/experiments/sigma_predictor-time_prediction-tot-1000000-seed-02.tar
STUDENT_SIZE=100000

# --------------------------------------------------------------------------------------------- #
# April 2021. Final buffer.
# --------------------------------------------------------------------------------------------- #
PREV=1000000
NEXT=1000000

NAME=ant_td3_online_stud_total_${STUDENT_SIZE}_curriculum_${TYPE}_p_${PREV}_n_${NEXT}
for SEED in 30 31 32 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/online_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_prev ${PREV}  --c_next ${NEXT} \
        --student_size ${STUDENT_SIZE}
done

# --------------------------------------------------------------------------------------------- #
# April 2021. Concurrent.
# --------------------------------------------------------------------------------------------- #
C_SCALE=1.00

NAME=ant_td3_online_stud_total_${STUDENT_SIZE}_curriculum_${TYPE}_scale_${C_SCALE}t
for SEED in 30 31 32 ; do
    taskset -c ${CPU1}-${CPU2} python spinup/teaching/online_rl.py  \
        --env ${ENV}  --exp_name ${NAME}  --seed ${SEED}  -bp ${B_PATH}  --t_source ${T_SOURCE}  --curriculum ${TYPE}  --c_scale ${C_SCALE} \
        --student_size ${STUDENT_SIZE}
done
