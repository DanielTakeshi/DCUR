# ----------------------------------------------------------------------------------------------------- #
# Now plot Q-values AND overlap together.
# ----------------------------------------------------------------------------------------------------- #

python spinup/teaching/plot_paper.py  --name ant          --add_only_curr  --list_currs curriculum_ep_250_logged_p_1  --window 20  --qvalues  --overlap
python spinup/teaching/plot_paper.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_250_logged_p_1  --window 20  --qvalues  --overlap
python spinup/teaching/plot_paper.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_250_logged_p_1  --window 20  --qvalues  --overlap
python spinup/teaching/plot_paper.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_250_logged_p_1  --window 20  --qvalues  --overlap
