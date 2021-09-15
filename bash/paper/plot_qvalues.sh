# ----------------------------------------------------------------------------------------------------- #
# Now plot Q-values, which we currently do with the --qvalues argument. I'm hoping this will make
# it easy to understand why we see the results we're seeing with data curricula.
# ----------------------------------------------------------------------------------------------------- #
# These default to additive curricula, but could do scaling curricula w/c=1.25 results.
# ----------------------------------------------------------------------------------------------------- #

## # For now let's not do the 800K here, we want to keep 1 pattern apparent. Maybe make window larger?
## python spinup/teaching/plot_paper.py  --name ant          --add_only_curr  --list_currs curriculum_ep_250_logged_p_1  --window 20  --qvalues
## python spinup/teaching/plot_paper.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_250_logged_p_1  --window 20  --qvalues
## python spinup/teaching/plot_paper.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_250_logged_p_1  --window 20  --qvalues
## python spinup/teaching/plot_paper.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_250_logged_p_1  --window 20  --qvalues

# One row, makes the figure smaller.
python spinup/teaching/plot_paper.py  --name ant          --add_only_curr  --list_currs curriculum_ep_250_logged_p_1  --window 20  --qvalues_row
python spinup/teaching/plot_paper.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_250_logged_p_1  --window 20  --qvalues_row
python spinup/teaching/plot_paper.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_250_logged_p_1  --window 20  --qvalues_row
python spinup/teaching/plot_paper.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_250_logged_p_1  --window 20  --qvalues_row

## # Haven't tried 2500 epochs.
## #python spinup/teaching/plot_paper.py  --name ant          --add_only_curr  --list_currs curriculum_ep_2500_logged  --window  6  --qvalues  --longer_2500
## #python spinup/teaching/plot_paper.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_2500_logged  --window  6  --qvalues  --longer_2500
## #python spinup/teaching/plot_paper.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_2500_logged  --window 12  --qvalues  --longer_2500
## #python spinup/teaching/plot_paper.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_2500_logged  --window 12  --qvalues  --longer_2500
