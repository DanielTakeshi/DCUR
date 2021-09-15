import os
import os.path as osp
import warnings

# Default neural network backend for each algo
# (Must be either 'tf1' or 'pytorch')
DEFAULT_BACKEND = {
    'vpg': 'pytorch',
    'trpo': 'tf1',
    'ppo': 'pytorch',
    'ddpg': 'pytorch',
    'td3': 'pytorch',
    'sac': 'pytorch',
    'BCQ': 'pytorch',
}

# Where experiment outputs are saved by default.
if 'SPINUP_DATA_DIR' in os.environ:
    SPINUP = os.environ['SPINUP_DATA_DIR']
    if not os.path.exists(SPINUP):
        os.makedirs(SPINUP)
else:
    warnings.warn("SPINUP_DATA_DIR environment variable not set.\n \
        Please set it in your bashrc. For example, like this: \
        export SPINUP_DATA_DIR=/data/seita/spinup")
    SPINUP = osp.abspath(osp.dirname(osp.dirname(__file__)))
DEFAULT_DATA_DIR = osp.join(SPINUP,'data')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching
# experiments.
WAIT_BEFORE_LAUNCH = 5
