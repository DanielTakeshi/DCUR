from copy import deepcopy
import itertools
import numpy as np
import sys
import time
import torch
from torch.optim import Adam
import spinup.algos.pytorch.sac.core as core            # Daniel: use SAC's core, not TD3.
from spinup.algos.pytorch.td3.td3 import ReplayBuffer   # Daniel: use TD3's buffer, not SAC.
from spinup.utils.logx import EpochLogger
from spinup.teaching.offline_utils import (
    should_we_end_early, load_noise_predictor, get_revised_reward, get_gt_shaping_logged
)
np.set_printoptions(precision=4, suppress=True, linewidth=180)
torch.set_printoptions(precision=4, linewidth=180)


def sacStudent(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=50, buffer_path=None, buffer_size=int(1e6), # offline-rl changes below
        act_noise=0, np_path=None, env_arg=None, n_alpha=None, r_baseline=1, gt_shaping_logged=False,
        tp_path=None, curriculum=None, c_prev=None, c_next=None, c_scale=None):
    """Soft-Actor Critic (SAC).

    Following the td3Student, copy the method from `pytorch/sac/sac.py` and making changes.
    See td3Student for up to date documentation. NOTE: for now let's assume we're not
    updating alpha, which can be tuned in SAC. See the SAC teacher code for details.

    Also, we're not doing the overlap or sparse environment here, unlike with TD3 students.
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

    # Experience buffer, which contains teacher buffer. Critical change from vanilla TD3/SAC.
    params_desired = dict(buffer_path=buffer_path, np_path=np_path, env_arg=env_arg)
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    replay_buffer.load_from_disk(path=buffer_path,
                                 buffer_size=buffer_size,
                                 with_std=(curriculum == 'noise_rollout'),
                                 params_desired=params_desired)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # New for Offline RL: load a noise/time predictor. (NOTE: please ignore for SAC students)
    predictor = None
    if np_path is not None:
        predictor = load_noise_predictor(obs_dim, model_path=np_path)
    elif tp_path is not None:
        predictor = load_noise_predictor(obs_dim, model_path=tp_path)

    # Set up function for computing SAC Q-losses.
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # Daniel NOTE: introduce reward shaping here and overrides `r`.
        if np_path is not None:
            r, r_stats = get_revised_reward(predictor, n_alpha, r_baseline, o, a, r, o2, d)
        elif tp_path is not None:
            r, r_stats = get_revised_reward(predictor, n_alpha, r_baseline, o, a, r, o2, d, time=True)
        elif gt_shaping_logged:
            r, r_stats = get_gt_shaping_logged(n_alpha, data, replay_buffer, r)

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Mandi NOTE: SAC doesn't have a second target actor like TD3,
            # hence also doesn't do policy smoothing
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

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

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)
    logger.save_state({'env': env}, itr=0)

    def update(data):
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

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info  = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False, noise_scale=0):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32),
                      deterministic) # NOTE: different act than TD3
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, deterministic=True, noise_scale=0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

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

        # Update handling
        batch = replay_buffer.sample_batch(batch_size)
        update(data=batch)

        # If curriculum, change the replay buffer indexing. See `check_curriculum()`.
        if (curriculum is not None) and (t % update_every == 0):
            if (c_prev is not None) and (c_next is not None):
                new_min = max(t - c_prev, 0)
                new_max = min(t + c_next, buffer_size)
            elif c_scale is not None:
                new_min = 0
                new_max = min(int(c_scale * t), buffer_size)
            replay_buffer.set_curriculum_values(min_value=new_min, max_value=new_max)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, itr=epoch)

            # Test the performance of the deterministic version of the agent.
            test_agent()
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
            if gt_shaping_logged:
                logger.log_tabular('RExtra', with_min_and_max=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            if terminate:
                print('\nEnding early due to passing termination test.\n')
                sys.exit()
