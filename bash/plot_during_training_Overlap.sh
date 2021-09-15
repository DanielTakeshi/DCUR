# --------------------------------------------------------------------------------------- #
# Plot overlap during training. We can do this in several ways, such as using the built-in
# Spinningup plotter and then going from there.
# --------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------ #
# [April 13, 2021] Investigate overlap during training. https://github.com/CannyLab/spinningup/pull/26
# Similar to `plot_during_training_TP.sh` except this is overlap instead of TP, obviously!
# Also we don't have the two different time predictors here -- we don't use them.
# ------------------------------------------------------------------------------------------------------ #

python spinup/teaching/plot_offline_rl_with_Overlap.py  --name ant          --window  6  --add_only_curr  --list_currs curriculum_ep_500_logged_p_50000000_n_1000000_overlap
python spinup/teaching/plot_offline_rl_with_Overlap.py  --name halfcheetah  --window  6  --add_only_curr  --list_currs curriculum_ep_500_logged_p_50000000_n_1000000_overlap
python spinup/teaching/plot_offline_rl_with_Overlap.py  --name hopper       --window 12  --add_only_curr  --list_currs curriculum_ep_500_logged_p_50000000_n_1000000_overlap
python spinup/teaching/plot_offline_rl_with_Overlap.py  --name walker2d     --window 12  --add_only_curr  --list_currs curriculum_ep_500_logged_p_50000000_n_1000000_overlap

python spinup/teaching/plot_offline_rl_with_Overlap.py  --name ant          --window  6  --add_only_curr  --list_currs curriculum_ep_500_logged_scale_1.00t_overlap
python spinup/teaching/plot_offline_rl_with_Overlap.py  --name halfcheetah  --window  6  --add_only_curr  --list_currs curriculum_ep_500_logged_scale_1.00t_overlap
python spinup/teaching/plot_offline_rl_with_Overlap.py  --name hopper       --window 12  --add_only_curr  --list_currs curriculum_ep_500_logged_scale_1.00t_overlap
python spinup/teaching/plot_offline_rl_with_Overlap.py  --name walker2d     --window 12  --add_only_curr  --list_currs curriculum_ep_500_logged_scale_1.00t_overlap
