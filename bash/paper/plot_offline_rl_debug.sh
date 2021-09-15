# ----------------------------------------------------------------------------------------------------- #
# Plot the Offline RL results. We should just refer to our spinup/ directory and detect all the directories
# with 'offline' in them. Ideally this will cycle through all seeds and results for us.
# https://github.com/CannyLab/spinningup/pull/32
# ----------------------------------------------------------------------------------------------------- #
# NOTE: this is for 'debugging' and my usual way of plotting stuff. For an actual paper we should use
# something a little more formal. For that: `bash/paper/plot_offline_rl.sh`.
# ----------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------------- #
# [June 03-18 2021]: Results with different data curriculum choies.
# ----------------------------------------------------------------------------------------------------- #

python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_ep_250_logged_p_     --window  6
python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_ep_250_logged_scale  --window  6
python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_250_logged_p_     --window  6
python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_250_logged_scale  --window  6
python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_250_logged_p_     --window 12
python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_250_logged_scale  --window 12
python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_250_logged_p      --window 12
python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_250_logged_scale  --window 12

#python spinup/teaching/plot_offline_rl.py  --name ant          --add_only_curr  --list_currs curriculum_ep_2500_logged  --window  6
#python spinup/teaching/plot_offline_rl.py  --name halfcheetah  --add_only_curr  --list_currs curriculum_ep_2500_logged  --window  6
#python spinup/teaching/plot_offline_rl.py  --name hopper       --add_only_curr  --list_currs curriculum_ep_2500_logged  --window 12
#python spinup/teaching/plot_offline_rl.py  --name walker2d     --add_only_curr  --list_currs curriculum_ep_2500_logged  --window 12
