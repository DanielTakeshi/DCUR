# ----------------------------------------------------------------------------------------------------- #
# Now plot overlap. Plot structure may be similar to the Q-values one.
# Note that we didn't do this analysis for the long-horizon training.
# ----------------------------------------------------------------------------------------------------- #

# As before, we can stick with logged, additive curricula here, except just the prev=1M cases?
# Edit: actually wait we still have to change a toggle in the code, `c_add_only` and `c_scale_only` -- we should remove that.
# Otherwise we can really only run half of these at once...
#python spinup/teaching/plot_paper.py  --name ant          --add_only_curr  --list_currs curriculum_ep_250_logged_p_1    --window 20  --overlap
python spinup/teaching/plot_paper.py  --name ant          --add_only_curr  --list_currs curriculum_ep_250_logged_scale  --window 20  --overlap
#python spinup/teaching/plot_paper.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_250_logged_p_1    --window 20  --overlap
python spinup/teaching/plot_paper.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_250_logged_scale  --window 20  --overlap
#python spinup/teaching/plot_paper.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_250_logged_p_1    --window 20  --overlap
python spinup/teaching/plot_paper.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_250_logged_scale  --window 20  --overlap
#python spinup/teaching/plot_paper.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_250_logged_p_1    --window 20  --overlap
python spinup/teaching/plot_paper.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_250_logged_scale  --window 20  --overlap

## # We can use this for debugging:
## python spinup/teaching/plot_offline_rl_with_Overlap.py  --name ant          --add_only_curr  --list_currs curriculum_ep_250_logged_p_1    --window 20
## python spinup/teaching/plot_offline_rl_with_Overlap.py  --name ant          --add_only_curr  --list_currs curriculum_ep_250_logged_scale  --window 20
## python spinup/teaching/plot_offline_rl_with_Overlap.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_250_logged_p_1    --window 20
## python spinup/teaching/plot_offline_rl_with_Overlap.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_250_logged_scale  --window 20
## python spinup/teaching/plot_offline_rl_with_Overlap.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_250_logged_p_1    --window 20
## python spinup/teaching/plot_offline_rl_with_Overlap.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_250_logged_scale  --window 20
## python spinup/teaching/plot_offline_rl_with_Overlap.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_250_logged_p_1    --window 20
## python spinup/teaching/plot_offline_rl_with_Overlap.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_250_logged_scale  --window 20
