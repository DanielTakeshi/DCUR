"""Script for Offline RL experiments."""
from copy import deepcopy
import itertools
import numpy as np
import sys
import os
import os.path as osp
import time
import gym
import torch
from torch.optim import Adam
from spinup.utils.logx import EpochLogger
from spinup.user_config import DEFAULT_DATA_DIR
from spinup.teaching.train_batchRL_BCQ import BCQ
from spinup.teaching.offline_sac import sacStudent
from spinup.teaching.offline_utils import (
    should_we_end_early, load_noise_predictor, get_revised_reward, get_gt_shaping_logged,
    TestTimePredictor
)
from spinup.teaching.overlap import OverlapOfflineRL
import spinup.algos.pytorch.td3.core as td3_core  # Daniel: make td3 vs sac distinction clear
import spinup.algos.pytorch.sac.core as sac_core  # Mandi: note different 'core' than td3
from spinup.algos.pytorch.td3.td3 import ReplayBuffer
np.set_printoptions(precision=4, suppress=True, linewidth=180)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=180)


def sanity_check_args(args):
    """Sanity check the args.

    Let's do some sanity checks, the arguments are a bit tricky. But let's assume
    naming convention for tail, which is the pickle file containing the relay buffer,
    is: dtype, followed by attribute and its value. First thing is to parse the buffer
    path and get properties from the file name.

    For checking the curriculum parameters, in general please see `check_curriculum()`.
    """
    _, tail = os.path.split(args.buffer_path)
    tail = (tail.replace('.p','')).split('-')
    buf_dtype = tail[0]
    buf_data = {}
    for tidx in range(1, len(tail), 2):
        buf_key = tail[tidx]
        buf_val = tail[tidx+1]
        buf_data[buf_key] = buf_val
    print(f'\nLoading buffer from disk for Offline RL. Type: {buf_dtype}')
    for key in sorted(buf_data.keys()):
        print(f'\tbuffer[{key}]: {buf_data[key]}')
    print()

    # Assume the experiment name should signal that it's offline, etc.
    envname, _ = (args.env).split('-')
    envname = envname.lower()
    assert envname in args.exp_name, f'Put {envname} in {args.exp_name}'
    assert 'offline' in args.exp_name, f'Put \'offline\' in {args.exp_name}'

    # Do checks on the experiment name, mainly to make sure we save in informative directories.
    if ('_finalb' in args.exp_name or '_concurrent' in args.exp_name):
        print('Warning! Using finalb or concurrent is deprecated.')
        sys.exit()
    assert ('_uniform_' in args.exp_name or '_constanteps_' in args.exp_name or
            '_uniformeps_' in args.exp_name or '_randact_' in args.exp_name or
            '_nonaddunif_' in args.exp_name or '_curriculum_' in args.exp_name), args.exp_name

    if buf_dtype == 'final_buffer':
        # Using the logged replay buffer from the teacher's training history.
        assert 'curriculum' in args.exp_name
        assert 'rollout' not in args.buffer_path
        assert args.curriculum is not None
        assert args.np_path is None
    else:
        # For any uniform noise buffers. For now supporting unif_ or uniform_ in name.
        buf_noise = (buf_data['noise']).replace('uniform_', 'unif_')
        assert (buf_noise in args.exp_name) or (buf_data['noise'] in args.exp_name)
        assert 'final_buffer' not in args.buffer_path
        assert 'valid' not in args.buffer_path
        # Also check if we copied over the corresponding .txt file for data generation.
        buffer_stats = (args.buffer_path).replace('.p', '.txt')
        assert osp.exists(buffer_stats), f'Did we scp this:  {buffer_stats}'
    assert args.t_source in args.buffer_path, f'{args.t_source} vs {args.buffer_path}'

    # Let's enforce that only one of these is true.
    assert not ((args.np_path is not None) and args.gt_shaping_logged)
    assert not ((args.np_path is not None) and (args.tp_path is not None))

    # If using noise predictor, check that it exists and matches noise. E.g., if
    # data was generated w/uniform_0.01_0.40, our NP should also be trained on that.
    # (Jan 17) Need to handle 'uniformeps_x_y_eps_[...]' since that should probably
    # use the same x_y that was used for the 'uniform_x_y' distribution.
    if args.np_path is not None:
        assert ('sigma_predictor' in args.np_path) and ('.tar' in args.np_path)
        args.np_path = osp.join(DEFAULT_DATA_DIR, args.np_path)
        assert osp.exists(args.np_path), args.np_path
        assert f'_np_{args.n_alpha}' in args.exp_name, args.exp_name
        if buf_dtype == 'final_buffer':
            pass
        elif 'uniformeps' in buf_data['noise']:
            # Check buffer noise matches noise used to train noise predictor.
            # Let's get rid of any epsilons and filter to handle new distribution.
            bufnoise = (buf_data['noise']).replace('uniformeps_', 'uniform_')
            bufnoise = '_'.join((bufnoise.split('_'))[:3]) # Only keep first 3
            assert bufnoise in args.np_path, args.np_path
        elif 'constanteps' in buf_data['noise']:
            # Given constanteps_x_eps, we want uniform_0.0_x
            bufstr1 = (buf_data['noise'].split('_'))[0]
            bufstr2 = (buf_data['noise'].split('_'))[1]
            assert bufstr1 == 'constanteps', bufstr1
            desired_str = f'uniform_0.0_{bufstr2}'
            assert desired_str in args.np_path, args.np_path
        elif 'randact' in buf_data['noise']:
            # We're not going to enforce anything for now.
            print('For {} we are not enforcing a specific noise predictor'.format(
                    buf_data['noise']))
        elif 'nonaddunif' in buf_data['noise']:
            # TODO(daniel) actually we can train a noise predictor pretty easily.
            print('For {} we are not enforcing a specific noise predictor'.format(
                    buf_data['noise']))
        else:
            raise ValueError('Buffer is not suported: {}'.format(buf_data['noise']))

    # Time predictor! This must be paired with logged data.
    if args.tp_path is not None:
        assert ('sigma_predictor-time_prediction' in args.tp_path) and ('.tar' in args.tp_path)
        args.tp_path = osp.join(DEFAULT_DATA_DIR, args.tp_path)
        assert osp.exists(args.tp_path), args.tp_path
        assert f'_tp_{int(args.n_alpha)}' in args.exp_name, args.exp_name
        assert f'_np_{int(args.n_alpha)}' not in args.exp_name, args.exp_name
        if 'data-aug' in args.tp_path:
            assert 'data-aug_tp_' in args.exp_name
        if 'data-aug' in args.exp_name:
            assert 'data-aug' in args.tp_path

    # Probably shouldn't be using this, but just in case.
    if args.gt_shaping_logged:
        assert buf_dtype == 'final_buffer', buf_data
        assert f'_gtrewshape_{args.n_alpha}' in args.exp_name, args.exp_name

    # Some preliminary checks. See `check_curriculum` for more.
    if args.curriculum is not None:
        if args.curriculum == 'logged':
            assert buf_dtype == 'final_buffer', buf_data
        elif args.curriculum == 'noise_rollout':
            assert buf_dtype != 'final_buffer', buf_data

    # Miscellaneous.
    if args.epochs != 250:
        assert f'_ep_{args.epochs}' in args.exp_name
    if args.overlap:
        assert '_overlap' in args.exp_name
    if 'overlap' in args.exp_name:
        assert args.overlap
    if args.sparse:
        assert '_sparse' in args.exp_name
    if 'sparse' in args.exp_name:
        assert args.sparse
    assert args.r_baseline in [0, 1]


def check_curriculum(args):
    """Checks and defines the curriculum.

    Supports two cases: logged and noise_rollout, and for each, these curricula:

    Case 1: sample from indices [t - c_prev, t + c_prev].
    Case 2: sample from indices [0, t * c_scale].

    In all cases, indices are subject to [0, 1M] limits.
    """
    assert args.curriculum in ['logged', 'noise_rollout']

    # Only one of these conditions should be true.
    cond1 = (args.c_prev is not None) and (args.c_next is not None)
    cond2 = (args.c_scale is not None)
    assert cond1 or cond2
    assert not (cond1 and cond2)

    # Now for these conditions, check if values make sense.
    assert args.buffer_size == int(1e6), args.buffer_size
    if cond1:
        # Daniel: ah! If we run >250 epochs, we need c_prev >1M unless we want to
        # ignore older samples deliberately.
        #assert 0 <= args.c_prev <= args.buffer_size, args.c_prev
        assert 0 <= args.c_next <= args.buffer_size, args.c_next
        assert args.c_scale is None
    if cond2:
        assert 0 <= args.c_scale <= 2.0, args.c_scale
        assert args.c_prev is None and args.c_next is None


def td3Student(env_fn, actor_critic=td3_core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10, buffer_path=None, buffer_size=int(1e6),
        np_path=None, env_arg=None, n_alpha=None, r_baseline=1, gt_shaping_logged=False,
        tp_path=None, curriculum=None, c_prev=None, c_next=None, c_scale=None, overlap=False,
        sparse=False):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3).

    For now, copying the method from `pytorch/td3/td3.py` and making changes:

    buffer_path: path to the buffer which we load from, for Offline RL.
    buffer_size: size of data we use for Offline RL. Should be <= size of stored data.
    np_path: path to the PyTorch file of the noise predictor, which we can use for an
        intrinsic / shaped reward. Detect via a `None` check. This applies to both the
        sigma and the xi/vareps predictor. Higher values mean worse states, roughly.
    env_arg: added so we can check environment names.
    n_alpha: scalar factor to the reward shaping, for both noise and time.
    r_baseline: {0,1}, affects choice of the extra reward. We should use 1 probably.
    concurrent: True if doing the 'concurrent' setting from (Fujimoto et al, ICML 2019).
        Requires some adjusting to the replay buffer code so if we're at time step `t`
        during Offline RL gradient updates, we only sample up to index `t` in the buffer.
    gt_shaping_logged: True if doing reward shaping for the logged data setting (either
        finalb or concurrent) AND we want to do reward shaping on that based on fraction
        of time in the online training the data sample is from, AND we want to use ground
        truth info instead of a learned network.
    tp_path: adding time predictor since it has a different effect than reward predictor.
    curriculum, c_prev, c_next, c_scale: see `check_curriculum()`.
    overlap: Adding if we want to do overlap analysis, False by default (good).
    sparse: Adding if we want to ignore env reward and just use our own reward shaping.
    """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer, which contains teacher buffer. Critical change from vanilla TD3.
    params_desired = dict(buffer_path=buffer_path, np_path=np_path, env_arg=env_arg)
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    replay_buffer.load_from_disk(path=buffer_path,
                                 buffer_size=buffer_size,
                                 with_std=(curriculum == 'noise_rollout'),
                                 params_desired=params_desired)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(td3_core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # New for Offline RL: if necessary, load a noise/time predictor or overlap predictor.
    predictor = None
    TestTP = None
    if np_path is not None:
        predictor = load_noise_predictor(obs_dim, model_path=np_path)
    elif tp_path is not None:
        predictor = load_noise_predictor(obs_dim, model_path=tp_path)
        TestTP = TestTimePredictor(net=predictor, log_dir=logger.output_dir)
    overlap_model = None
    if overlap:
        overlap_model = OverlapOfflineRL(env, curriculum, t_buffer=replay_buffer)

    # Set up function for computing TD3 Q-losses.
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # NOTE(daniel): introduce reward shaping here and overrides `r`.
        if np_path is not None:
            r, r_stats = get_revised_reward(
                    predictor, n_alpha, r_baseline, o, a, r, o2, d, sparse=sparse, time=False)
        elif tp_path is not None:
            r, r_stats = get_revised_reward(
                    predictor, n_alpha, r_baseline, o, a, r, o2, d, sparse=sparse, time=True)
        elif gt_shaping_logged:
            r, r_stats = get_gt_shaping_logged(n_alpha, data, replay_buffer, r)

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy(),
                         Rew=r.detach().numpy())

        # In this case, np_path and the gt ones belong in the same return case.
        if (np_path is not None) or (tp_path is not None) or gt_shaping_logged:
            return loss_q, loss_info, r_stats
        else:
            return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q1_pi = ac.q1(o, ac.pi(o))
        return -q1_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)
    logger.save_state({'env': env}, itr=0)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        if (np_path is not None) or (tp_path is not None) or gt_shaping_logged:
            # Daniel: mirroring storage of Q-values. Storing arrays in `logger.store()`
            # means the logger concatenates them later when calling `log_tabular()`.
            loss_q, loss_info, r_stats = compute_loss_q(data)
            logger.store(**r_stats)
        else:
            loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things. Daniel: saves 'Q1Vals', 'Q2Vals', numpy arays of size (batch_size).
        logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:
            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        assert not np.isnan(a).any(), a
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def test_agent_tp():
        # Let's run the time predictor on this data.
        TestTP.clear_student_data()
        n_samples = 0
        while True:
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                a = get_action(o, 0)
                TestTP.store_student_data(o)
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
                n_samples += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            if n_samples >= TestTP.n_student_samples:
                break

    def test_agent_overlap():
        # Let's keep running until we generate sufficient samples. Report the same
        # test episode stats, except it might have different numbers of episodes.
        # (June 03) Actually for this we should NOT use the logger.store here! We
        # run this after test_agent() which should have the logger statistics.
        overlap_model.clear_student_data()
        n_samples = 0
        while True:
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # For overlap analysis, need the action computed earlier.
                a = get_action(o, 0)
                overlap_model.store_student_data(o,a)  # save (o,a).
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
                n_samples += 1
            #logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            if n_samples >= overlap_model.n_student_samples:
                break

    # Prepare for offline RL (no env interaction).
    total_steps = steps_per_epoch * epochs
    start_time = time.time()

    # For rollouts parameterized by noise, we reorder the buffer.
    if curriculum == 'noise_rollout':
        replay_buffer.sort_by_decreasing_noise()

    # For any curricula thus far, start with `update_after` items in buffer, since during
    # training, we waited until this many steps before doing gradient updates.
    start_t = 0
    if curriculum is not None:
        start_t = update_after
        replay_buffer.set_curriculum_values(min_value=0, max_value=update_after)

    # Main loop: Update using actions taken from replay buffer data
    for t in range(start_t, total_steps):

        # Update handling. NOTE(daniel): for the VERY FIRST minibatch (t=0), this will
        # sample indices [0, 1000) even in the 'final_buffer' case, but all t >= 1 times
        # result in sampling from [0, 1000000) as desired. If we had it in a loop (as in
        # online training, 50 times) this should happen after setting the curriculum.
        batch = replay_buffer.sample_batch(batch_size)
        update(data=batch, timer=t)

        # If curriculum, change the replay buffer indexing. See `check_curriculum()`.
        if (curriculum is not None) and (t % update_every == 0):
            if (c_prev is not None) and (c_next is not None):
                new_min = max(t - c_prev, 0)
                new_max = min(t + c_next, buffer_size)
            elif c_scale is not None:
                new_min = 0
                new_max = min(int(c_scale * t), buffer_size)
            replay_buffer.set_curriculum_values(min_value=new_min, max_value=new_max)

        # End of epoch handling, can save the student policy.
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, itr=epoch)

            # Test the performance of the deterministic version of the agent, using `test_env`,
            # NOT `env` which is for online student samples. These methods use `o`, `ep_len`,
            # etc., but they will not override those values corresponding to the current `env`.
            if overlap:
                # NOTE(daniel) We NEED to be consistent with test_agent(), since it will
                # produce a flat 10 episodes, which we need for consistent reporting.
                test_agent()
                test_agent_overlap()
                o_res_1, o_res_2 = overlap_model.compute_overlap(teacher_data=replay_buffer)
            if tp_path is not None:
                test_agent_tp()
                tp_res = TestTP.predict_student_data()
            if (not overlap) and (tp_path is None):
                test_agent()  # If none of the above two case apply.
            print('buffer:', replay_buffer.curr_min, replay_buffer.curr_max)

            # Log info about epoch. Test termination BEFORE dumping.
            terminate = should_we_end_early(logger)
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalGradSteps', t)  # gradient steps
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Rew', with_min_and_max=True)
            if np_path is not None:
                logger.log_tabular('RExtra', with_min_and_max=True)
                logger.log_tabular('Sigma1', with_min_and_max=True)  # or xi/vareps
                logger.log_tabular('Sigma2', with_min_and_max=True)  # or xi/vareps
            if tp_path is not None:
                logger.log_tabular('RExtra', with_min_and_max=True)
                logger.log_tabular('Time1', with_min_and_max=True)
                logger.log_tabular('Time2', with_min_and_max=True)
                logger.log_tabular('TP_Mean', tp_res['TP_Mean'])  # on student data
                logger.log_tabular('TP_Medi', tp_res['TP_Medi'])  # on student data
                logger.log_tabular('TP_Std',  tp_res['TP_Std'])   # on student data
                logger.log_tabular('TP_Neg',  tp_res['TP_Neg'])   # on student data
            if gt_shaping_logged:
                logger.log_tabular('RExtra', with_min_and_max=True)
            if overlap:
                logger.log_tabular('O_NoActOverlapV',  o_res_1['OverlapV'])
                logger.log_tabular('O_NoActAccT',      o_res_1['AccT'])
                logger.log_tabular('O_NoActAccV',      o_res_1['AccV'])
                logger.log_tabular('O_NoActLossT',     o_res_1['LossT'])
                logger.log_tabular('O_NoActLossV',     o_res_1['LossV'])
                logger.log_tabular('O_NoActBestEpoch', o_res_1['Epoch'])
                #logger.log_tabular('O_ActOverlapV',    o_res_2['OverlapV'])
                #logger.log_tabular('O_ActAccT',        o_res_2['AccT'])
                #logger.log_tabular('O_ActAccV',        o_res_2['AccV'])
                #logger.log_tabular('O_ActLossT',       o_res_2['LossT'])
                #logger.log_tabular('O_ActLossV',       o_res_2['LossV'])
                #logger.log_tabular('O_ActBestEpoch',   o_res_2['Epoch'])
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            if terminate:
                print('\nEnding early due to passing termination test.\n')
                sys.exit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--algorithm', type=str, default='td3')
    parser.add_argument('--sac_alpha', type=float, default=0.2,
        help='Mandi: in case we want a different alpha for student')

    # New stuff for Offline RL
    parser.add_argument('--buffer_size', '-bs', type=int, default=int(1e6),
        help='Size of the Offline RL data.')
    parser.add_argument('--buffer_path', '-bp', type=str, default=None,
        help='Full path to the saved replay buffer we load for Offline RL.')
    parser.add_argument('--np_path', '-np', type=str, default=None,
        help='If enabled, we should load a noise predictor from this path.')
    parser.add_argument('--tp_path', '-tp', type=str, default=None,
        help='If enabled, we should load a TIME predictor from this path.')
    parser.add_argument('--n_alpha', '-na', type=float, default=None,
        help='Strength of noise OR time predictor reward shaping.')
    parser.add_argument('--t_source', type=str, default=None,
        help='The exp_name of the teacher, with the seed please.')
    parser.add_argument('--r_baseline', type=int, default=1,
        help='If 1, then we use the baseline version of extra reward.')
    parser.add_argument('--curriculum', type=str, default=None,
        help='Specify curriculum. See `check_curriculum()` for valid strings')
    parser.add_argument('--c_prev', type=int, default=None,
        help='Allowed samples behind the current time/noise cursor `t`')
    parser.add_argument('--c_next', type=int, default=None,
        help='Allowed samples ahead of the current time cursor `t`')
    parser.add_argument('--c_scale', type=float, default=None,
        help='Allowed samples ahead of the current time/noise cursor `t`')
    parser.add_argument('--gt_shaping_logged', action='store_true', default=False)
    # NOTE(daniel) Only TD3 students for now.
    parser.add_argument('--overlap', action='store_true', default=False,
        help='Enable overlap analysis if desired.')
    parser.add_argument('--sparse', action='store_true', default=False,
        help='Ignore env-reward if needed, only reward should be extra stuff we add.')
    args = parser.parse_args()

    # The usual, adding data dir to start.
    if not os.path.exists(args.buffer_path):
        print(f'{args.buffer_path} does not exist, pre-pending {DEFAULT_DATA_DIR}')
        args.buffer_path = osp.join(DEFAULT_DATA_DIR, args.buffer_path)
        assert os.path.exists(args.buffer_path), args.buffer_path

    # Sanity check! Should be thorough and catch inconsistencies in arguments.
    sanity_check_args(args)
    if args.curriculum is not None:
        check_curriculum(args)

    # Back to 'normal', but for the logger I want to get the teacher type in
    # the dir, so we save to `data/exp_name_offline/teacher_s[..]`.
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed,
        offline_rl_source=args.t_source)

    if args.algorithm == 'BCQ':
        # BCQ baseline (Fujimoto et al., ICML 2019). Implemented in separate script.
        BCQ(lambda : gym.make(args.env), discount=args.gamma,
            seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs,
            buffer_path=args.buffer_path, buffer_size=args.buffer_size)
    elif args.algorithm == 'td3':
        # TD3-based students. Uses a different 'core' than SAC.
        td3Student(lambda : gym.make(args.env), actor_critic=td3_core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
            seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs,
            buffer_path=args.buffer_path, buffer_size=args.buffer_size, np_path=args.np_path,
            env_arg=args.env, n_alpha=args.n_alpha, r_baseline=args.r_baseline,
            gt_shaping_logged=args.gt_shaping_logged, tp_path=args.tp_path, curriculum=args.curriculum,
            c_prev=args.c_prev, c_next=args.c_next, c_scale=args.c_scale, overlap=args.overlap,
            sparse=args.sparse)
    elif args.algorithm == 'sac':
        # SAC-based students. Uses a different 'core' than TD3, has extra `alpha=sac_alpha` arg.
        sacStudent(lambda : gym.make(args.env), actor_critic=sac_core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
            seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs,
            buffer_path=args.buffer_path, buffer_size=args.buffer_size, np_path=args.np_path,
            env_arg=args.env, alpha=args.sac_alpha, n_alpha=args.n_alpha, r_baseline=args.r_baseline,
            gt_shaping_logged=args.gt_shaping_logged, tp_path=args.tp_path, curriculum=args.curriculum,
            c_prev=args.c_prev, c_next=args.c_next, c_scale=args.c_scale)
    else:
        raise ValueError(f'{args.algorithm} not supported')
