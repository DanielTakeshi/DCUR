# ----------------------------------------------------------------------------------------------------- #
# Plot the Online RL results. We're using `plot_online_rl.py` and making files that have `online` in their names.
# ----------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------------- #
# [April 20-21, 2021]: 200K and 100K online student data spaced throughout training.
# Plots: https://github.com/CannyLab/spinningup/pull/30
# ----------------------------------------------------------------------------------------------------- #

## python spinup/teaching/plot_online_rl.py  --name ant          --window  6  --add_only_curr  \
##     --list_currs online_stud_total_200000_curriculum_logged_p_1000000_n_1000000  online_stud_total_200000_curriculum_logged_scale_1.00t
## python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window  6  --add_only_curr  \
##     --list_currs online_stud_total_200000_curriculum_logged_p_1000000_n_1000000  online_stud_total_200000_curriculum_logged_scale_1.00t
## python spinup/teaching/plot_online_rl.py  --name hopper       --window 12  --add_only_curr  \
##     --list_currs online_stud_total_200000_curriculum_logged_p_1000000_n_1000000  online_stud_total_200000_curriculum_logged_scale_1.00t
## python spinup/teaching/plot_online_rl.py  --name walker2d     --window 12  --add_only_curr  \
##     --list_currs online_stud_total_200000_curriculum_logged_p_1000000_n_1000000  online_stud_total_200000_curriculum_logged_scale_1.00t
##
## python spinup/teaching/plot_online_rl.py  --name ant          --window  6  --add_only_curr  \
##     --list_currs online_stud_total_100000_curriculum_logged_p_1000000_n_1000000  online_stud_total_100000_curriculum_logged_scale_1.00t
## python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window  6  --add_only_curr  \
##     --list_currs online_stud_total_100000_curriculum_logged_p_1000000_n_1000000  online_stud_total_100000_curriculum_logged_scale_1.00t
## python spinup/teaching/plot_online_rl.py  --name hopper       --window 12  --add_only_curr  \
##     --list_currs online_stud_total_100000_curriculum_logged_p_1000000_n_1000000  online_stud_total_100000_curriculum_logged_scale_1.00t
## python spinup/teaching/plot_online_rl.py  --name walker2d     --window 12  --add_only_curr  \
##     --list_currs online_stud_total_100000_curriculum_logged_p_1000000_n_1000000  online_stud_total_100000_curriculum_logged_scale_1.00t

# ----------------------------------------------------------------------------------------------------- #
# [April 27, 2021]: 500 epochs, overlap and time predictor statistics, etc.
# Plots: https://github.com/CannyLab/spinningup/pull/30
# These just have the teacher + student reward (also can hard-code the grid width to make wider):
# Actually we probably don't need these plots since the other ones (overlap, TP) will contain this...
# ----------------------------------------------------------------------------------------------------- #

## # Final Buffer
## python spinup/teaching/plot_online_rl.py  --name ant          --window  6  --add_only_curr  \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window  6  --add_only_curr  \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name hopper       --window 12  --add_only_curr  \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name walker2d     --window 12  --add_only_curr  \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10
##
## # Concurrent
## python spinup/teaching/plot_online_rl.py  --name ant          --window  6  --add_only_curr  \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window  6  --add_only_curr  \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name hopper       --window 12  --add_only_curr  \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name walker2d     --window 12  --add_only_curr  \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10

# ----------------------------------------------------------------------------------------------------- #
# [April 27 to May 03, 2021]: same as before except we can include overlap, use --with_overlap argument.
# ----------------------------------------------------------------------------------------------------- #

## # Final Buffer
## python spinup/teaching/plot_online_rl.py  --name ant          --window  6  --add_only_curr  --with_overlap \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window  6  --add_only_curr  --with_overlap \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name hopper       --window 12  --add_only_curr  --with_overlap \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name walker2d     --window 12  --add_only_curr  --with_overlap \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10
##
## # Concurrent
## python spinup/teaching/plot_online_rl.py  --name ant          --window  6  --add_only_curr  --with_overlap \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window  6  --add_only_curr  --with_overlap \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name hopper       --window 12  --add_only_curr  --with_overlap \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name walker2d     --window 12  --add_only_curr  --with_overlap \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10

# ----------------------------------------------------------------------------------------------------- #
# [April 27 to May 03, 2021]: now the time predictor, use --with_tp, can go into the code to change
# whether we want the mean of the predictions ('TP_Mean') or the fraction of negatives ('TP_Neg').
# ----------------------------------------------------------------------------------------------------- #

## # Final Buffer
## python spinup/teaching/plot_online_rl.py  --name ant          --window  6  --add_only_curr  --with_tp \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window  6  --add_only_curr  --with_tp \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name hopper       --window 12  --add_only_curr  --with_tp \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name walker2d     --window 12  --add_only_curr  --with_tp \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10
##
## # Concurrent
## python spinup/teaching/plot_online_rl.py  --name ant          --window  6  --add_only_curr  --with_tp \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window  6  --add_only_curr  --with_tp \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name hopper       --window 12  --add_only_curr  --with_tp \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10
## python spinup/teaching/plot_online_rl.py  --name walker2d     --window 12  --add_only_curr  --with_tp \
##     --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
##                  online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10

# ----------------------------------------------------------------------------------------------------- #
# [May 05, 2021]: Now with the ACTUAL data-aug reward shaping, and ALSO with r_baseline=0. Above, we
# were using r_baseline=1 which is the default. I think the TP version is more readable. We can compare
# this with the prior runs that used alpha x (f(s') - f(s)) but that wasn't `data-aug` unfortunately.
# Results: https://github.com/CannyLab/spinningup/pull/30
# Use `f_override` to shortern the plot names.
# ----------------------------------------------------------------------------------------------------- #

# Final Buffer
python spinup/teaching/plot_online_rl.py  --name ant          --window  6  --add_only_curr  --with_tp \
    --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
                 online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10 \
                 online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_data-aug_tp_10 \
    --f_override ant_tp_comparisons_finalbuf
python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window  6  --add_only_curr  --with_tp \
    --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
                 online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10 \
                 online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_data-aug_tp_10 \
    --f_override halfcheetah_tp_comparisons_finalbuf
python spinup/teaching/plot_online_rl.py  --name hopper       --window 12  --add_only_curr  --with_tp \
    --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
                 online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10 \
                 online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_data-aug_tp_10 \
    --f_override hopper_tp_comparisons_finalbuf
python spinup/teaching/plot_online_rl.py  --name walker2d     --window 12  --add_only_curr  --with_tp \
    --list_currs online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_0 \
                 online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_tp_10 \
                 online_stud_total_100000_curriculum_ep_500_logged_p_50000000_n_1000000_overlap_data-aug_tp_10 \
    --f_override walker2d_tp_comparisons_finalbuf

# Concurrent
#python spinup/teaching/plot_online_rl.py  --name ant          --window  6  --add_only_curr  --with_tp \
#    --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
#                 online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10
#                 online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_data-aug_tp_10 \
#    --f_override ant_tp_comparisons_concurrent
python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window  6  --add_only_curr  --with_tp \
    --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
                 online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10 \
                 online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_data-aug_tp_10 \
    --f_override halfcheetah_tp_comparisons_concurrent
python spinup/teaching/plot_online_rl.py  --name hopper       --window 12  --add_only_curr  --with_tp \
    --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
                 online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10 \
                 online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_data-aug_tp_10 \
    --f_override hopper_tp_comparisons_concurrent
python spinup/teaching/plot_online_rl.py  --name walker2d     --window 12  --add_only_curr  --with_tp \
    --list_currs online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_0 \
                 online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_tp_10 \
                 online_stud_total_100000_curriculum_ep_500_logged_scale_1.00t_overlap_data-aug_tp_10 \
    --f_override walker2d_tp_comparisons_concurrent
