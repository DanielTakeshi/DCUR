# ----------------------------------------------------------------------------------------------------- #
# Plot the sparse reward case, when the student ONLY gets reward based on the reward shaping term. This
# deals with both offline and online RL settings (as well as final buffer vs concurrent within those).
# For full sets of plots: https://github.com/CannyLab/spinningup/pull/30
# ----------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------------- #
# [May 03-05, 2021]: Offline. Note: careful about using `data-aug` or not!
# ----------------------------------------------------------------------------------------------------- #

# Final Buffer
# python spinup/teaching/plot_offline_rl_with_TP.py  --name ant          --window  6  --add_only_curr \
#     --list_currs
python spinup/teaching/plot_offline_rl_with_TP.py  --name halfcheetah  --window  6  --add_only_curr \
    --list_currs  halfcheetah_td3_offline_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_sparse_data-aug_tp_1
python spinup/teaching/plot_offline_rl_with_TP.py  --name hopper       --window 12  --add_only_curr \
    --list_currs  hopper_td3_offline_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_sparse_data-aug_tp_1
python spinup/teaching/plot_offline_rl_with_TP.py  --name walker2d     --window 12  --add_only_curr \
    --list_currs  walker2d_td3_offline_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_sparse_data-aug_tp_1

# Concurrent
# python spinup/teaching/plot_offline_rl_with_TP.py  --name ant          --window  6  --add_only_curr \
#     --list_currs
python spinup/teaching/plot_offline_rl_with_TP.py  --name halfcheetah  --window  6  --add_only_curr \
    --list_currs  halfcheetah_td3_offline_curriculum_ep_500_logged_scale_1.00t_overlap_sparse_data-aug_tp_1
python spinup/teaching/plot_offline_rl_with_TP.py  --name hopper       --window 12  --add_only_curr \
    --list_currs  hopper_td3_offline_curriculum_ep_500_logged_scale_1.00t_overlap_sparse_data-aug_tp_1
python spinup/teaching/plot_offline_rl_with_TP.py  --name walker2d     --window 12  --add_only_curr \
    --list_currs  walker2d_td3_offline_curriculum_ep_500_logged_scale_1.00t_overlap_sparse_data-aug_tp_1

# ----------------------------------------------------------------------------------------------------- #
# [May 03-05, 2021]: Online. Note: careful about using `data-aug` or not!
# ----------------------------------------------------------------------------------------------------- #

# Final Buffer
python spinup/teaching/plot_online_rl.py  --name ant          --window  6  --add_only_curr  --with_tp \
    --list_currs  ant_td3_online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_sparse_tp_1
python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window  6  --add_only_curr  --with_tp \
    --list_currs  halfcheetah_td3_online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_sparse_data-aug_tp_1
python spinup/teaching/plot_online_rl.py  --name hopper       --window 12  --add_only_curr  --with_tp \
    --list_currs  hopper_td3_online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_sparse_tp_1
python spinup/teaching/plot_online_rl.py  --name walker2d     --window 12  --add_only_curr  --with_tp \
    --list_currs  walker2d_td3_online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_sparse_tp_1

# Concurrent
python spinup/teaching/plot_online_rl.py  --name ant          --window  6  --add_only_curr  --with_tp \
    --list_currs  ant_td3_online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_sparse_tp_1
python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window  6  --add_only_curr  --with_tp \
    --list_currs  halfcheetah_td3_online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_sparse_data-aug_tp_1
# python spinup/teaching/plot_online_rl.py  --name hopper       --window 12  --add_only_curr  --with_tp \
#     --list_currs
# python spinup/teaching/plot_online_rl.py  --name walker2d     --window 12  --add_only_curr  --with_tp \
#     --list_currs
