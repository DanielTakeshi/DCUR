# --------------------------------------------------------------------------- #
# Plot the Offline RL results. We should just refer to our spinup/ directory
# and detect all the directories with 'offline' in them. Ideally this will
# cycle through all seeds and results for us.
# --------------------------------------------------------------------------- #

## # (Jan 18) More fine-grained comparison with and without noise predictor. For this, add both
## # flags, plus the distributions we explicitly want to plot as multi-length argument (a list).
## # https://github.com/CannyLab/spinningup/pull/7
## for NAME in halfcheetah hopper walker2d ; do
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs uniform_0.0_0.25
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs uniform_0.0_0.50
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs uniform_0.0_0.75
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs uniform_0.0_1.00
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs uniform_0.0_1.25
## done


## # (Jan 19) now has the 'uniformeps' distribution, and simplifies the teacher subplots.
## # https://github.com/CannyLab/spinningup/pull/8
## for NAME in halfcheetah hopper walker2d ; do
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  uniformeps_0.0_0.25_0.5_filter
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  uniformeps_0.0_0.50_0.5_filter
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  uniformeps_0.0_0.75_0.5_filter
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  uniformeps_0.0_1.00_0.5_filter
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  uniformeps_0.0_1.25_0.5_filter
## done


## # (Jan 26 - Feb 02) now has the 'constanteps' distribution, which might be easier to interpret.
## # https://github.com/CannyLab/spinningup/pull/9   But maybe I shouldn't have done this.
## # ALL OF THESE SHOULD USE TEACHER SEED 40 (except 50 for walker2d).
## for NAME in halfcheetah hopper walker2d ; do
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  constanteps_0.00_0.0
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  constanteps_0.25_0.5_filter
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  constanteps_0.50_0.5_filter
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  constanteps_1.00_0.5_filter
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  constanteps_1.50_0.5_filter
## done


## # (Jan 26) Continuing runs from today where I now make a separate `add_logged` category
## # to handle finalb and concurrent.  https://github.com/CannyLab/spinningup/pull/9
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_logged
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_logged
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_logged


## # (Feb 02) actually this is simialr to the prior 'uniformeps' updates, but this time
## # for seeds 40/50 teachers. New pull request: https://github.com/CannyLab/spinningup/pull/9
## # And here we also have new distributions focusing on higher noise cases.
## for NAME in halfcheetah hopper walker2d ; do
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  uniformeps_0.0_1.50_0.5_filter
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  uniformeps_0.0_1.50_0.75_filter
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  uniformeps_0.0_1.50_0.9_filter
## done


## # (Feb 05) with the new randact_* set of distribhutions, for non-additive Gaussian noise.
## # On `uniform-act-data` branch, results: https://github.com/CannyLab/spinningup/pull/11
## for NAME in halfcheetah hopper walker2d ; do
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  randact_0.25_filter
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  randact_0.50_filter
##
##     python spinup/teaching/plot_offline_rl.py  --name ${NAME}  --add_naive  --add_np \
##         --list_distrs  randact_0.75_filter
## done


## # [Feb 12-15, 2021]: Updated finalb and concurrent plots. https://github.com/CannyLab/spinningup/pull/11
## # This also has learned time predictors, with scale parameter specified by '_np_{key}'.
## # Also includes the 'gtrewshape' results even though we probably shouldn't do that.
## # Also on Feb 15 I updated plots to report stat1 and stat2.
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_logged
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_finalb
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_concur
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_logged       --window 10
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_finalb  --window 10
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_concur  --window 10
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_logged       --window 10
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_finalb  --window 10
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_concur  --window 10


## # [Feb 15, 2021]: Updated uniformeps plots: https://github.com/CannyLab/spinningup/pull/9
## # This has the higher noise distribution cases. Teacher seeds 40/50, with stat1/stat2.
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_naive  --add_np  --list_distrs  uniformeps_0.0_1.50_0.5_filter
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_naive  --add_np  --list_distrs  uniformeps_0.0_1.50_0.75_filter
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_naive  --add_np  --list_distrs  uniformeps_0.0_1.50_0.9_filter
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_naive  --add_np  --list_distrs  uniformeps_0.0_1.50_0.5_filter   --window 10
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_naive  --add_np  --list_distrs  uniformeps_0.0_1.50_0.75_filter  --window 10
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_naive  --add_np  --list_distrs  uniformeps_0.0_1.50_0.9_filter   --window 10
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_naive  --add_np  --list_distrs  uniformeps_0.0_1.50_0.5_filter   --window 10
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_naive  --add_np  --list_distrs  uniformeps_0.0_1.50_0.75_filter  --window 10
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_naive  --add_np  --list_distrs  uniformeps_0.0_1.50_0.9_filter   --window 10


## # [Feb 15, 2021]: Updated nonaddunif plots: https://github.com/CannyLab/spinningup/pull/14
## # This has the higher noise distribution cases. Teacher seeds 40/50, with stat1/stat2.
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_naive  --add_np  --list_distrs  nonaddunif_0.0_0.5_filter
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_naive  --add_np  --list_distrs  nonaddunif_0.0_1.0_filter
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_naive  --add_np  --list_distrs  nonaddunif_0.0_0.5_filter  --window 10
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_naive  --add_np  --list_distrs  nonaddunif_0.0_1.0_filter  --window 10
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_naive  --add_np  --list_distrs  nonaddunif_0.0_0.5_filter  --window 10
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_naive  --add_np  --list_distrs  nonaddunif_0.0_1.0_filter  --window 10


## # ----------------------------------------------------------------------------------------------------- #
## # [Feb 23, 2021]: Curriculum Plots for Logged Data: https://github.com/CannyLab/spinningup/pull/17
## # Note: the `concurrent` and `final_buffer` cases are special cases of curriculum.
## # For now, it's concurrent + final_buffer + whatever `list_currs` we pick, for each plot.
## # ----------------------------------------------------------------------------------------------------- #

## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_p_1000000_n_50000
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_p_1000000_n_50000
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_p_1000000_n_50000  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_p_1000000_n_50000  --window 12

## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_p_1000000_n_200000
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_p_1000000_n_200000
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_p_1000000_n_200000  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_p_1000000_n_200000  --window 12

## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_p_1000000_n_400000
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_p_1000000_n_400000
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_p_1000000_n_400000  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_p_1000000_n_400000  --window 12

## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_p_1000000_n_700000
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_p_1000000_n_700000
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_p_1000000_n_700000  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_p_1000000_n_700000  --window 12

## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_p_450000_n_50000
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_p_450000_n_50000
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_p_450000_n_50000  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_p_450000_n_50000  --window 12

## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_p_800000_n_50000
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_p_800000_n_50000
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_p_800000_n_50000  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_p_800000_n_50000  --window 12


## # ----------------------------------------------------------------------------------------------------- #
## # [Feb 25, 2021]: Curriculum Plots for Logged Data w/scaling: https://github.com/CannyLab/spinningup/pull/18
## # Note: the `concurrent` and `final_buffer` cases are special cases of curriculum.
## # For now, it's concurrent + final_buffer + whatever `list_currs` we pick, for each plot.
## # Note: starting the next day I actually changed the naming to use curriculum_logged for this.
## # ----------------------------------------------------------------------------------------------------- #
##
## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_scale_0.50t
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_scale_0.50t
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_scale_0.50t  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_scale_0.50t  --window 12

## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_scale_0.75t
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_scale_0.75t
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_scale_0.75t  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_scale_0.75t  --window 12

## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_logged_scale_0.90t
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_logged_scale_0.90t
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_logged_scale_0.90t  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_logged_scale_0.90t  --window 12

## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_logged_scale_1.00t
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_logged_scale_1.00t
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_logged_scale_1.00t  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_logged_scale_1.00t  --window 12

## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_logged_scale_1.10t
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_logged_scale_1.10t
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_logged_scale_1.10t  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_logged_scale_1.10t  --window 12


## # ----------------------------------------------------------------------------------------------------- #
## # [Feb 26, 2021]: For curriculum_noise_rollout https://github.com/CannyLab/spinningup/pull/19
## # ----------------------------------------------------------------------------------------------------- #
##
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_naive  \
##         --list_distrs  uniform_0.0_1.50   uniform_0.0_1.50_curriculum_noise_rollout_p_1000000_n_0
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_naive  \
##         --list_distrs  uniformeps_0.0_1.50_0.9_filter  uniformeps_0.0_1.50_0.9_filter_curriculum_noise_rollout_p_1000000_n_0
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_naive  \
##         --list_distrs  nonaddunif_0.0_0.5_filter  nonaddunif_0.0_0.5_filter_curriculum_noise_rollout_p_1000000_n_0
##
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_naive  --window 12  \
##         --list_distrs  uniform_0.0_1.50   uniform_0.0_1.50_curriculum_noise_rollout_p_1000000_n_0
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_naive  --window 12  \
##         --list_distrs  uniformeps_0.0_1.50_0.9_filter  uniformeps_0.0_1.50_0.9_filter_curriculum_noise_rollout_p_1000000_n_0
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_naive  --window 12  \
##         --list_distrs  nonaddunif_0.0_0.5_filter  nonaddunif_0.0_0.5_filter_curriculum_noise_rollout_p_1000000_n_0
##
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_naive  --window 12  \
##         --list_distrs  uniform_0.0_1.50   uniform_0.0_1.50_curriculum_noise_rollout_p_1000000_n_0
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_naive  --window 12  \
##         --list_distrs  uniformeps_0.0_1.50_0.9_filter  uniformeps_0.0_1.50_0.9_filter_curriculum_noise_rollout_p_1000000_n_0
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_naive  --window 12  \
##         --list_distrs  nonaddunif_0.0_0.5_filter  nonaddunif_0.0_0.5_filter_curriculum_noise_rollout_p_1000000_n_0


## # ----------------------------------------------------------------------------------------------------- #
## # [March 09-17, 2021]: Hopefully finalized full buffer vs curriculum (scale 1.00t) results with the
## # time prediction reward shaping? Plots: https://github.com/CannyLab/spinningup/pull/24
## # ----------------------------------------------------------------------------------------------------- #
##
## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_logged_p_1000000_n_1000000
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_logged_p_1000000_n_1000000
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_logged_p_1000000_n_1000000  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_logged_p_1000000_n_1000000  --window 12
## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_logged_scale_1.00t
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_logged_scale_1.00t
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_logged_scale_1.00t  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_logged_scale_1.00t  --window 12

## # These are for plotting a subset of the runs above, for visual clarity, e.g., to share in my slides.
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_logged_p_1000000_n_1000000 \
##         --exclude_currs  curriculum_logged_p_1000000_n_1000000_tp_30  curriculum_logged_p_1000000_n_1000000_tp_50  curriculum_logged_p_1000000_n_1000000_tp_100
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_logged_p_1000000_n_1000000  --window 12 \
##         --exclude_currs  curriculum_logged_p_1000000_n_1000000_tp_1  curriculum_logged_p_1000000_n_1000000_tp_50  curriculum_logged_p_1000000_n_1000000_tp_100
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_logged_p_1000000_n_1000000  --window 12 \
##         --exclude_currs  curriculum_logged_p_1000000_n_1000000_tp_1  curriculum_logged_p_1000000_n_1000000_tp_10  curriculum_logged_p_1000000_n_1000000_tp_100
## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_logged_scale_1.00t  \
##         --exclude_currs  curriculum_logged_scale_1.00t_tp_30  curriculum_logged_scale_1.00t_tp_50  curriculum_logged_scale_1.00t_tp_100
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_logged_scale_1.00t \
##         --exclude_currs  curriculum_logged_scale_1.00t_tp_30  curriculum_logged_scale_1.00t_tp_50  curriculum_logged_scale_1.00t_tp_100
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_logged_scale_1.00t  --window 12 \
##         --exclude_currs  curriculum_logged_scale_1.00t_tp_1  curriculum_logged_scale_1.00t_tp_50  curriculum_logged_scale_1.00t_tp_100
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_logged_scale_1.00t  --window 12 \
##         --exclude_currs  curriculum_logged_scale_1.00t_tp_50  curriculum_logged_scale_1.00t_tp_100

## # Also includes some cases of running with 2x more training epochs.
## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_ep_500_logged_p_1000000_n_1000000
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_500_logged_p_1000000_n_1000000
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_500_logged_p_1000000_n_1000000  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_500_logged_p_1000000_n_1000000  --window 12
## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_ep_500_logged_scale_1.00t
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_500_logged_scale_1.00t
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_500_logged_scale_1.00t  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_500_logged_scale_1.00t  --window 12

## # And 4x (1000 epochs) :D I did not do this for Ant since it was obviously not working well.
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah --add_only_curr  --list_currs curriculum_ep_1000_logged_scale_1.00t
## python spinup/teaching/plot_offline_rl.py  --name hopper      --add_only_curr  --list_currs curriculum_ep_1000_logged_scale_1.00t  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d    --add_only_curr  --list_currs curriculum_ep_1000_logged_scale_1.00t  --window 12

## # Data augmented time predictor cases.
## python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_logged_p_1000000_n_1000000 curriculum_logged_p_1000000_n_1000000_data-aug \
##     --exclude_currs  curriculum_logged_p_1000000_n_1000000_tp_1 curriculum_logged_p_1000000_n_1000000_tp_10 curriculum_logged_p_1000000_n_1000000_tp_30  curriculum_logged_p_1000000_n_1000000_tp_50  curriculum_logged_p_1000000_n_1000000_tp_100
## python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_logged_p_1000000_n_1000000 curriculum_logged_p_1000000_n_1000000_data-aug \
##     --exclude_currs  curriculum_logged_p_1000000_n_1000000_tp_1 curriculum_logged_p_1000000_n_1000000_tp_10 curriculum_logged_p_1000000_n_1000000_tp_30  curriculum_logged_p_1000000_n_1000000_tp_50  curriculum_logged_p_1000000_n_1000000_tp_100
## python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_logged_p_1000000_n_1000000 curriculum_logged_p_1000000_n_1000000_data-aug \
##     --exclude_currs  curriculum_logged_p_1000000_n_1000000_tp_1 curriculum_logged_p_1000000_n_1000000_tp_10 curriculum_logged_p_1000000_n_1000000_tp_30  curriculum_logged_p_1000000_n_1000000_tp_50  curriculum_logged_p_1000000_n_1000000_tp_100  --window 12
## python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_logged_p_1000000_n_1000000 curriculum_logged_p_1000000_n_1000000_data-aug \
##     --exclude_currs  curriculum_logged_p_1000000_n_1000000_tp_1 curriculum_logged_p_1000000_n_1000000_tp_10 curriculum_logged_p_1000000_n_1000000_tp_30  curriculum_logged_p_1000000_n_1000000_tp_50  curriculum_logged_p_1000000_n_1000000_tp_100  --window 12


# ----------------------------------------------------------------------------------------------------- #
# [April 19, 2021]: Let's try again with large-scale runs. https://github.com/CannyLab/spinningup/pull/24
# ----------------------------------------------------------------------------------------------------- #

python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_ep_2500_logged_p_50000000_n_1000000
python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_2500_logged_p_50000000_n_1000000
python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_2500_logged_p_50000000_n_1000000  --window 12
python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_2500_logged_p_50000000_n_1000000  --window 12
