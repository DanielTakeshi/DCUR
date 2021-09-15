# ----------------------------------------------------------------------------------------------------- #
# Plot the Online RL results. We're using `plot_online_rl.py` and making files that have `online` in their names.
# ----------------------------------------------------------------------------------------------------- #

## python spinup/teaching/plot_online_rl.py  --name ant          --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_25000_curriculum_ep_250
## python spinup/teaching/plot_online_rl.py  --name ant          --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_50000_curriculum_ep_250
## python spinup/teaching/plot_online_rl.py  --name ant          --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_100000_curriculum_ep_250
##
## python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_25000_curriculum_ep_250
## python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_50000_curriculum_ep_250
## python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_100000_curriculum_ep_250
##
## python spinup/teaching/plot_online_rl.py  --name hopper       --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_25000_curriculum_ep_250
## python spinup/teaching/plot_online_rl.py  --name hopper       --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_50000_curriculum_ep_250
## python spinup/teaching/plot_online_rl.py  --name hopper       --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_100000_curriculum_ep_250
##
## python spinup/teaching/plot_online_rl.py  --name walker2d     --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_25000_curriculum_ep_250
## python spinup/teaching/plot_online_rl.py  --name walker2d     --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_50000_curriculum_ep_250
## python spinup/teaching/plot_online_rl.py  --name walker2d     --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_100000_curriculum_ep_250
##
## # Other, special runs.
## python spinup/teaching/plot_online_rl.py  --name ant          --window 20  --with_overlap  --add_only_curr \
##         --list_currs online_stud_total_10000_curriculum_ep_250

# ----------------------------------------------------------------------------------------------------- #
# Another way to express online RL, maybe show performance of two different curricula, side by side,
# and then show how the curriculum really helps? Maybe use `paper_fig`?
# ----------------------------------------------------------------------------------------------------- #

#python spinup/teaching/plot_online_rl.py  --name ant          --window 20  --paper_fig  --add_only_curr \
#        --list_currs online_stud_total
#
# python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window 20  --paper_fig  --add_only_curr \
#         --list_currs online_stud_total
#
# python spinup/teaching/plot_online_rl.py  --name hopper       --window 20  --paper_fig  --add_only_curr \
#         --list_currs online_stud_total
#
# python spinup/teaching/plot_online_rl.py  --name walker2d     --window 20  --paper_fig  --add_only_curr \
#         --list_currs online_stud_total

# These have the 0% in the plot as a baseline.
python spinup/teaching/plot_online_rl.py  --name ant          --window 20  --paper_fig  --add_only_curr \
        --list_currs  online_stud_total offline_curriculum_ep_250_logged_p_1000000_n_1000000 offline_curriculum_ep_250_logged_scale_1.00t \
        --exclude_currs  10000_curriculum_ep_250_logged_p_1000000_n_1000000_overlap 10000_curriculum_ep_250_logged_scale_1.00t_overlap \
        --f_override online_ant_comparisons

python spinup/teaching/plot_online_rl.py  --name halfcheetah  --window 20  --paper_fig  --add_only_curr \
        --list_currs  online_stud_total offline_curriculum_ep_250_logged_p_1000000_n_1000000 offline_curriculum_ep_250_logged_scale_1.00t \
        --exclude_currs  10000_curriculum_ep_250_logged_p_1000000_n_1000000_overlap 10000_curriculum_ep_250_logged_scale_1.00t_overlap \
        --f_override online_halfcheetah_comparisons

python spinup/teaching/plot_online_rl.py  --name hopper       --window 20  --paper_fig  --add_only_curr \
        --list_currs  online_stud_total offline_curriculum_ep_250_logged_p_1000000_n_1000000 offline_curriculum_ep_250_logged_scale_1.00t \
        --exclude_currs  10000_curriculum_ep_250_logged_p_1000000_n_1000000_overlap 10000_curriculum_ep_250_logged_scale_1.00t_overlap \
        --f_override online_hopper_comparisons

python spinup/teaching/plot_online_rl.py  --name walker2d     --window 20  --paper_fig  --add_only_curr \
        --list_currs  online_stud_total offline_curriculum_ep_250_logged_p_1000000_n_1000000 offline_curriculum_ep_250_logged_scale_1.00t \
        --exclude_currs  10000_curriculum_ep_250_logged_p_1000000_n_1000000_overlap 10000_curriculum_ep_250_logged_scale_1.00t_overlap \
        --f_override online_walker2d_comparisons
