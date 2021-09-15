# --------------------------------------------------------------------------- #
# Plot the Offline RL results WITH time predictor applied on the student while training.
# This will help us detect performance. Plot this side by side with 3 values:
# Teacher performance, student performance, then time predictor. For April 13
# and earlier, we can detect these runs from `*_tp_0/` since we ran the TP but
# without actually using it for reward shaping.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# [April 13, 2021]: https://github.com/CannyLab/spinningup/pull/28
# Note that we are using PREV=50M since I wanted to use >250 epochs, and this
# means we'll always have the first few samples to sample from.
# --------------------------------------------------------------------------- #

## python spinup/teaching/plot_offline_rl_with_TP.py  --name ant          --window  6  --add_only_curr  --list_currs \
##         curriculum_ep_500_logged_p_50000000_n_1000000_tp_0  curriculum_ep_500_logged_p_50000000_n_1000000_data-aug_tp_0
## python spinup/teaching/plot_offline_rl_with_TP.py  --name halfcheetah  --window  6  --add_only_curr  --list_currs \
##         curriculum_ep_500_logged_p_50000000_n_1000000_tp_0  curriculum_ep_500_logged_p_50000000_n_1000000_data-aug_tp_0
## python spinup/teaching/plot_offline_rl_with_TP.py  --name hopper       --window 12  --add_only_curr  --list_currs \
##         curriculum_ep_500_logged_p_50000000_n_1000000_tp_0  curriculum_ep_500_logged_p_50000000_n_1000000_data-aug_tp_0
## python spinup/teaching/plot_offline_rl_with_TP.py  --name walker2d     --window 12  --add_only_curr  --list_currs \
##         curriculum_ep_500_logged_p_50000000_n_1000000_tp_0  curriculum_ep_500_logged_p_50000000_n_1000000_data-aug_tp_0
##
## python spinup/teaching/plot_offline_rl_with_TP.py  --name ant          --window  6  --add_only_curr  --list_currs \
##         curriculum_ep_500_logged_p_50000000_n_0_tp_0  curriculum_ep_500_logged_p_50000000_n_0_data-aug_tp_0
## python spinup/teaching/plot_offline_rl_with_TP.py  --name halfcheetah  --window  6  --add_only_curr  --list_currs \
##         curriculum_ep_500_logged_p_50000000_n_0_tp_0  curriculum_ep_500_logged_p_50000000_n_0_data-aug_tp_0
## python spinup/teaching/plot_offline_rl_with_TP.py  --name hopper       --window 12  --add_only_curr  --list_currs \
##         curriculum_ep_500_logged_p_50000000_n_0_tp_0  curriculum_ep_500_logged_p_50000000_n_0_data-aug_tp_0
## python spinup/teaching/plot_offline_rl_with_TP.py  --name walker2d     --window 12  --add_only_curr  --list_currs \
##         curriculum_ep_500_logged_p_50000000_n_0_tp_0  curriculum_ep_500_logged_p_50000000_n_0_data-aug_tp_0

# --------------------------------------------------------------------------- #
# [April 13, 2021]: https://github.com/CannyLab/spinningup/pull/28
# Same issue report, except we use different TPs on the SAME data. Add `--plot_multiple_TPs`.
# --------------------------------------------------------------------------- #

python spinup/teaching/plot_offline_rl_with_TP.py  --name ant          --window  6  --add_only_curr  --plot_multiple_TPs  --list_currs \
        curriculum_ep_500_logged_p_50000000_n_1000000_tp_0  curriculum_ep_500_logged_p_50000000_n_1000000_data-aug_tp_0
python spinup/teaching/plot_offline_rl_with_TP.py  --name halfcheetah  --window  6  --add_only_curr  --plot_multiple_TPs  --list_currs \
        curriculum_ep_500_logged_p_50000000_n_1000000_tp_0  curriculum_ep_500_logged_p_50000000_n_1000000_data-aug_tp_0
python spinup/teaching/plot_offline_rl_with_TP.py  --name hopper       --window 12  --add_only_curr  --plot_multiple_TPs  --list_currs \
        curriculum_ep_500_logged_p_50000000_n_1000000_tp_0  curriculum_ep_500_logged_p_50000000_n_1000000_data-aug_tp_0
python spinup/teaching/plot_offline_rl_with_TP.py  --name walker2d     --window 12  --add_only_curr  --plot_multiple_TPs  --list_currs \
        curriculum_ep_500_logged_p_50000000_n_1000000_tp_0  curriculum_ep_500_logged_p_50000000_n_1000000_data-aug_tp_0

python spinup/teaching/plot_offline_rl_with_TP.py  --name ant          --window  6  --add_only_curr  --plot_multiple_TPs  --list_currs \
        curriculum_ep_500_logged_p_50000000_n_0_tp_0  curriculum_ep_500_logged_p_50000000_n_0_data-aug_tp_0
python spinup/teaching/plot_offline_rl_with_TP.py  --name halfcheetah  --window  6  --add_only_curr  --plot_multiple_TPs  --list_currs \
        curriculum_ep_500_logged_p_50000000_n_0_tp_0  curriculum_ep_500_logged_p_50000000_n_0_data-aug_tp_0
python spinup/teaching/plot_offline_rl_with_TP.py  --name hopper       --window 12  --add_only_curr  --plot_multiple_TPs  --list_currs \
        curriculum_ep_500_logged_p_50000000_n_0_tp_0  curriculum_ep_500_logged_p_50000000_n_0_data-aug_tp_0
python spinup/teaching/plot_offline_rl_with_TP.py  --name walker2d     --window 12  --add_only_curr  --plot_multiple_TPs  --list_currs \
        curriculum_ep_500_logged_p_50000000_n_0_tp_0  curriculum_ep_500_logged_p_50000000_n_0_data-aug_tp_0
