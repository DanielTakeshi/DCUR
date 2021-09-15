import time
import joblib
import os
import os.path as osp
import torch
import numpy as np
import pickle
from spinup import EpochLogger
from spinup.algos.pytorch.td3.td3 import ReplayBuffer
np.set_printoptions(linewidth=200)


def load_policy_and_env(fpath, itr='last'):
    """Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations. We're using PyTorch only here.

    Each file in this folder has naming convention 'modelXX.pt', where 'XX'
    is either an integer or empty string. Empty string case corresponds to
    len(x)==8, hence that case is excluded.

    Daniel: actually I made it 'model-XX.pt', oops. Ah well, let's adjust.
    """
    if itr == 'last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value
        pytsave_path = osp.join(fpath, 'pyt_save')
        saves = [int(x.split('.')[0][6:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    get_action = load_pytorch_policy(fpath, itr)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_pytorch_policy(fpath, itr):
    """Load a pytorch policy saved with Spinning Up Logger.

    Daniel: we don't use dropout or batch norm, so I don't think it's necessary to call
    model.eval(). https://pytorch.org/tutorials/beginner/saving_loading_models.html

    Returns: function for producing an action given a single state.
    """
    fname = osp.join(fpath, 'pyt_save', 'model-'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)
    model = torch.load(fname)

    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def sample_std(noise_distr):
    """Based on the distribution, return a sample. Distribution types:

    constant_x:                 sigma = x
    constanteps_x_eps_filter:   sigma = x w/prob eps, but 0 (noise-free) otherwise
    discrete_[...]:             sigma ~ Discrete([...])
    uniform_x_y:                sigma ~ Uniform(x,y)
    uniformeps_x_y_eps_filter:  sigma ~ Uniform(x,y) w/prob eps, but 0 (noise-free) otherwise
    randact_eps_filter:         [sigma doesn't matter, we don't use additive Gaussian noise]
    nonaddunif_x_y_filter:      xi ~ Uniform(x,y), this is NOT Gaussian sigma but the xi param
                                Let's not do it here since we want to return Gaussian sigma.
                                So it should returna value of 0 like the randact_ distribution.
                                Do this in the `sample_varepsilon` method (varepsilon == xi).

    The ones with epsilon have 'F' in {filter, nofilt} to determine if we should even
    keep random actions (anything with a sigma applied). I'm only using 'filter' for
    now, but we support 'nofilter' if desired.

    HOWEVER, we actually combine constant and constanteps, along with uniform and
    uniformeps, because the application of epsilon happens per action in an episode.
    Unlike sigma, for example, which is fixed for an episode. The sigma is fixed so
    that we get a consistent noise level per sigma choice in an episode.
    """
    sigma = 0.0

    if 'constant_' in noise_distr or 'constanteps_' in noise_distr:
        sigma = float(noise_distr.split('_')[1])
    elif 'discrete_' in noise_distr:
        noise = noise_distr.split('_')
        noise = noise[1:]  # dump 'discrete_'
        noise = [float(n) for n in noise]   # turn to floats
        sigma = np.random.choice(noise)  # pick one of these
    elif 'uniform_' in noise_distr or 'uniformeps_' in noise_distr:
        L = float(noise_distr.split('_')[1])
        H = float(noise_distr.split('_')[2])
        sigma = np.random.uniform(low=L, high=H)
    elif 'randact_' in noise_distr or 'nonaddunif_' in noise_distr:
        pass
    else:
        raise NotImplementedError(noise_distr)

    return sigma


def sample_varepsilon(noise_distr):
    """Like sample_std, except here we sample xi (or varepsilon, I use both internally).

    TL;DR this is the parameter that tells us if we are applying noise at all.  If 1,
    we always apply noise. The buffer should only contain tuples w/NOISE-FREE actions.
    The distinctive distribution here is 'nonaddunif' which requires us to sample this.
    """
    vareps = 1.0

    if 'constanteps_' in args.noise:
        vareps   = float(args.noise.split('_')[2])
        filter_s = str(args.noise.split('_')[3])
    elif 'uniformeps_' in args.noise:
        vareps   = float(args.noise.split('_')[3])
        filter_s = str(args.noise.split('_')[4])
    elif 'randact_' in args.noise:
        vareps   = float(args.noise.split('_')[1])
        filter_s = str(args.noise.split('_')[2])
    elif 'nonaddunif_' in args.noise:
        L        = float(args.noise.split('_')[1])
        H        = float(args.noise.split('_')[2])
        vareps   = np.random.uniform(low=L, high=H)
        filter_s = str(args.noise.split('_')[3])
    else:
        filter_s = 'nofilt'
    assert filter_s in ['filter'], 'We should only be doing filter now.'

    return vareps, filter_s


def get_buffer_base_name(noise, size, data_type, ending):
    """Get the replay buffer name automatically from arguments.

    We make two versions (for now), one with .p for the pickle file of data, and
    another with .txt for logged data, to tell us behavioral policy performance.
    """
    assert ('constant_' in noise or 'discrete_' in noise or 'uniform_' in noise or
            'constanteps_' in noise or 'uniformeps_' in noise or 'randact_' in noise or
            'nonaddunif_' in noise), noise
    if 'uniform_' in noise:
        assert len(noise.split('_')) == 3, noise.split('_')
    elif 'uniformeps_' in noise:
        assert len(noise.split('_')) == 5, noise.split('_')
    elif 'randact_' in noise:
        assert len(noise.split('_')) == 3, noise.split('_')
    elif 'nonaddunif_' in noise:
        assert len(noise.split('_')) == 4, noise.split('_')
    assert size > 0, size
    assert data_type in ['train', 'valid', 'neither'], data_type
    assert ending in ['.txt', '.p'], ending
    base = f'rollout-maxsize-{size}-steps-{size}-noise-{noise}-dtype-{data_type}{ending}'
    return base


def run_policy(env, args, get_action, render=True, data_type='train', max_ep_len=1000):
    """Actually runs the policy, but more importantly, gathers data.

    Max episode length should be 1000 for standard MuJoCo environments.
    https://github.com/openai/spinningup/issues/37
    https://github.com/openai/gym/blob/master/gym/envs/__init__.py
    """
    assert env is not None

    # The `size` will be the `max_size` variable of the replay buffer.
    if data_type == 'train':
        size = int(args.train_size)
    elif data_type == 'valid':
        size = int(args.valid_size)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=size)

    # Sigma value for action = pi(o) + N(0, sigma)
    sigma = sample_std(noise_distr=args.noise)

    # Per-action epsilon and filtering. With vareps prob, apply noise, o/w do not.
    vareps, filter_samples = sample_varepsilon(noise_distr=args.noise)

    print(f'\nMade new buffer w/obs: {obs_dim}, act: {act_dim}, act_limit: {act_limit}.')
    print(f'Data type: {data_type}. Will run as many time steps until {size} items...\n')
    print(f'Noise distr: {args.noise}, starting sigma: {sigma:0.3f}, filter? --> {filter_samples}')

    base_pth = osp.join(args.fpath, 'buffer')
    base_txt = get_buffer_base_name(args.noise, size, data_type=data_type, ending='.txt')
    logger = EpochLogger(output_dir=base_pth, output_fname=base_txt)
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

    # Because we may filter samples, need `replay_buffer.size`.
    while (replay_buffer.size < size):
        if render:
            env.render()
            time.sleep(1e-3)

        # Get noise-free action from the policy.
        a = get_action(o)
        random_act = False

        # Perturb action based on our desired noise parameter. We clip here. The policy
        # net ALREADY has a tanh at the end, but we need another one for the added noise.
        # Also we need a special case if doing non-additive noise.
        if np.random.rand() < vareps:
            if 'randact_' in args.noise or 'nonaddunif_' in args.noise:
                a = env.action_space.sample()
            else:
                a += sigma * np.random.randn(act_dim)
            random_act = True

        # Only really needed for additive noise cases but should not hurt to have here.
        a = np.clip(a, -act_limit, act_limit)

        # Do the usual environment stepping.
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store information for usage later. If no filtering, add everything. If filtering,
        # only add the NON-random actions. And let's abuse 'std' term and save the 'vareps'.
        if filter_samples == 'nofilt':
            print('Warning: deprecated, will remove.')
            replay_buffer.store(o, a, r, o2, d, std=sigma)
        if filter_samples == 'filter' and (not random_act):
            if 'randact_' in args.noise or 'nonaddunif_' in args.noise:
                replay_buffer.store(o, a, r, o2, d, std=vareps)  # TODO(daniel) std=vareps is hacky
            else:
                replay_buffer.store(o, a, r, o2, d, std=sigma)

        # Update most recent observation [mainly to make rbuffer.store() just use one call].
        o = o2

        # Need max_ep_len if we are also using it to assign d=False if we hit the max len.
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d \t Sigma %.3f \t Vareps %.3f' %
                    (n, ep_ret, ep_len, sigma, vareps))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

            # Whenever episode finishes, reset the noise.
            sigma = sample_std(noise_distr=args.noise)
            vareps, _ = sample_varepsilon(noise_distr=args.noise)

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

    # Similar to algos.pytorch.td3.td3, except make sure we set with_std=True!
    base_p = get_buffer_base_name(args.noise, size, data_type=data_type, ending='.p')
    save_path = osp.join(args.fpath, 'buffer', base_p)
    print(f'\nSaving buffer of rolled out data to:\n\t{save_path}')
    params_save = dict(noise=args.noise, save_path=save_path)
    replay_buffer.dump_to_disk(path=save_path, with_std=True, params=params_save)


if __name__ == '__main__':
    # Main thing to adjust here is the noise, which specifies a distribution.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath',                type=str)
    parser.add_argument('--itr', '-i',          type=int, default=-1)
    parser.add_argument('--noise',              type=str, default='const_0.0')
    parser.add_argument('--train_size', '-ts',  type=int, default=1000000)
    parser.add_argument('--valid_size', '-vs',  type=int, default=200000)
    parser.add_argument('--norender', '-nr',    action='store_true')
    args = parser.parse_args()

    # Adjust fpath by prepending data dir.
    from spinup.user_config import DEFAULT_DATA_DIR
    assert 'data/' not in args.fpath, f'Double check {args.fpath}'
    args.fpath = osp.join(DEFAULT_DATA_DIR, args.fpath)
    print(f'Loading:  {args.fpath}')

    # Load env and the policy, the latter provides `get_action`.
    env, get_action = load_policy_and_env(args.fpath, args.itr if args.itr >=0 else 'last')

    # First, do train, then valid. They draw different sigmas and use the same env,
    # sequentially, hence there shouldn't be issues with overlapping train/valid data.
    run_policy(env, args, get_action, not(args.norender), data_type='train')
    run_policy(env, args, get_action, not(args.norender), data_type='valid')
