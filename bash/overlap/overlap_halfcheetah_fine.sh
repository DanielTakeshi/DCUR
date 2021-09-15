# ------------------------------------------------------------------------------ #
# Overlap Metric for snapshots.
# ------------------------------------------------------------------------------ #

# Fine version (opt=2), tested Feb 08.
for NAME in halfcheetah_td3_act0-1 halfcheetah_td3_act0-5 ; do
    python spinup/teaching/overlap.py  ${NAME}/${NAME}_s40/  --opt 2  --epochs 5  --use_gpu
    python spinup/teaching/overlap.py  ${NAME}/${NAME}_s40/  --opt 2  --epochs 5  --use_gpu  --add_acts
done
