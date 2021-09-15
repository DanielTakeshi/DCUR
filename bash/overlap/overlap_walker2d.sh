# ------------------------------------------------------------------------------ #
# Overlap Metric for snapshots.
# ------------------------------------------------------------------------------ #

# Coarse version (opt=1), tested Feb 07.
for NAME in walker2d_td3_act0-1 walker2d_td3_act0-5 ; do
    python spinup/teaching/overlap.py  ${NAME}/${NAME}_s50/  --opt 1  --use_gpu
    python spinup/teaching/overlap.py  ${NAME}/${NAME}_s50/  --opt 1  --use_gpu  --add_acts
done
