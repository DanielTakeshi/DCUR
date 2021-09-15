# ------------------------------------------------------------------------------- #
# Plot the Overlap. For now we can put coarse / fine together for ONE TEACHER.
# Means we usually computed some overlap for all buffers in teacher's directory.
# ------------------------------------------------------------------------------- #
# Feb 08, 2021. Plotting from overlap_1 directory.
# Feb 17, 2021. Plotting from overlap_2 directory. Saves ONE large plot.
# ------------------------------------------------------------------------------- #
SEED=40
for NAME in halfcheetah_td3 hopper_td3 ; do
    python spinup/teaching/plot_overlap.py  ${NAME}_act0-1/${NAME}_act0-1_s${SEED}/
    python spinup/teaching/plot_overlap.py  ${NAME}_act0-1/${NAME}_act0-1_s${SEED}/  --add_acts
    python spinup/teaching/plot_overlap.py  ${NAME}_act0-5/${NAME}_act0-5_s${SEED}/
    python spinup/teaching/plot_overlap.py  ${NAME}_act0-5/${NAME}_act0-5_s${SEED}/  --add_acts
done

SEED=50
NAME=walker2d_td3
python spinup/teaching/plot_overlap.py  ${NAME}_act0-1/${NAME}_act0-1_s${SEED}/
python spinup/teaching/plot_overlap.py  ${NAME}_act0-1/${NAME}_act0-1_s${SEED}/  --add_acts
python spinup/teaching/plot_overlap.py  ${NAME}_act0-5/${NAME}_act0-5_s${SEED}/
python spinup/teaching/plot_overlap.py  ${NAME}_act0-5/${NAME}_act0-5_s${SEED}/  --add_acts
