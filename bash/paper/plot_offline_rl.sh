# ----------------------------------------------------------------------------------------------------- #
# The OFFICIAL results we want to show in the paper. Use spinup.teaching.plot_paper
# Can also check out: https://github.com/CannyLab/spinningup/pull/32
# These will NOT contain the teacher rewards directly on there.
# See `bash/paper_plot_qvalues.sh` for plotting Q-values.
# ----------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------------- #
# [June 03-18 2021]: Results with different data curriculum choies. Actually since we're doing the
# table here, we can put a bunch of items together,so this puts ALL curricula together.
# The --window argument can be ignored here. This does both a plot and a table.
# ----------------------------------------------------------------------------------------------------- #

# Daniel: use this to generate ALL logged data (scaling AND additive), this will print out a table that I can use.
# And it should also pretty-format the standard error regions.
python spinup/teaching/plot_paper.py  --name ant          --add_only_curr  --list_currs curriculum_ep_250_logged  --window 20
python spinup/teaching/plot_paper.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_250_logged  --window 20
python spinup/teaching/plot_paper.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_250_logged  --window 20
python spinup/teaching/plot_paper.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_250_logged  --window 20

## # Daniel: But for plots IN THE PAPER, we may want to keep prev=1M to keep lots manageable, so use the exclude function.
## python spinup/teaching/plot_paper.py  --name ant          --add_only_curr  --list_currs curriculum_ep_250_logged  --window 20  \
##     --exclude_currs curriculum_ep_250_logged_p_800000_n_0_overlap
## python spinup/teaching/plot_paper.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_250_logged  --window 20  \
##     --exclude_currs curriculum_ep_250_logged_p_800000_n_0_overlap
## python spinup/teaching/plot_paper.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_250_logged  --window 20  \
##     --exclude_currs curriculum_ep_250_logged_p_800000_n_0_overlap
## python spinup/teaching/plot_paper.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_250_logged  --window 20  \
##     --exclude_currs curriculum_ep_250_logged_p_800000_n_0_overlap

## # And use this for the 2500 epoch plots.
## python spinup/teaching/plot_paper.py  --name ant          --add_only_curr  --list_currs curriculum_ep_2500_logged  --window 20  --longer_2500
## python spinup/teaching/plot_paper.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_2500_logged  --window 20  --longer_2500
## python spinup/teaching/plot_paper.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_2500_logged  --window 20  --longer_2500
## python spinup/teaching/plot_paper.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_2500_logged  --window 20  --longer_2500
