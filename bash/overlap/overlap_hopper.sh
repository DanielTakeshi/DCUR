# ------------------------------------------------------------------------------ #
# Overlap Metric for snapshots.
# ------------------------------------------------------------------------------ #

# Coarse version (opt=1), tested Feb 07.
for NAME in hopper_td3_act0-1 hopper_td3_act0-5 ; do
    python spinup/teaching/overlap.py  ${NAME}/${NAME}_s40/  --opt 1  --use_gpu
    python spinup/teaching/overlap.py  ${NAME}/${NAME}_s40/  --opt 1  --use_gpu  --add_acts
done
