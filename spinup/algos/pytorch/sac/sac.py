from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import os
import os.path as osp
import gym
import time
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger
from spinup.algos.pytorch.td3.td3 import ReplayBuffer


def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=25, final_buffer=False, act_noise=0.0, update_alpha=False):
    """Soft Actor-Critic (SAC)

    See https://spinningup.openai.com/en/latest/algorithms/sac.html
    Notable changes from OpenAI SpinningUp:

    (1) Increase default `save_freq` and enable saving multiple snapshots.
    (2) Add `final_buffer` to save the buffer of data during training. Fujimoto
        used N(0, 0.5) which I think means act_noise=0.5 (which is the standard
        deviation) but we can change act_noise.
    (3) Recording reward statistics so we can see what values to expect, which may
        also help us with reward shaping.
    (4) adding act_noise, which we don't use in SAC in normal cases, so set default to 0.
        It may be useful in case we want to roll out the poliices. In TD3, it is 0.1.
    (5) Inclusion of a target entropy, thanks to @Mandi. That's `update_alpha`.
    """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    target_entropy = - act_dim

    if update_alpha:
        log_alpha = torch.tensor(np.log(alpha))
        log_alpha.requires_grad = True

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

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            if update_alpha:
                use_alpha = torch.exp(log_alpha).detach()
                backup = r + gamma * (1 - d) * (q_pi_targ - use_alpha * logp_a2)
            else:
                backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy(),
                         Rew=r.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        if update_alpha:
            use_alpha = torch.exp(log_alpha).detach()
            loss_pi = (use_alpha * logp_pi - q_pi).mean()
        else:
            loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        # Mandi NOTE: TODO: add alpha loss to automatically tune alpha too
        if update_alpha:
            assert log_alpha.requires_grad
            use_alpha = torch.exp(log_alpha)
            loss_alpha = (use_alpha * (-logp_pi - target_entropy).detach()).mean()
            alpha_info = dict(Alpha=use_alpha.detach().numpy())

            return loss_pi, pi_info, loss_alpha, alpha_info

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)
    # Mandi NOTE: also try updating alpha
    if update_alpha:
        alpha_optimizer = Adam([log_alpha], lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)
    logger.save_state({'env': env}, itr=0)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        if update_alpha:
            pi_optimizer.zero_grad()
            alpha_optimizer.zero_grad()
            loss_pi, pi_info, loss_alpha, alpha_info = compute_loss_pi(data)
            loss_pi.backward()
            loss_alpha.backward()
            pi_optimizer.step()
            alpha_optimizer.step()
        else:
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)
        if update_alpha:
            logger.store(LossAlpha=loss_alpha.item(), **alpha_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False, noise_scale=0.0):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32),
                      deterministic)
        # Mandi NOTE: original td3 doesn't add noise (Daniel: original sac)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(o, act_noise)
            # probably don't need act_noise for teacher training but may need for rollout data generation
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook: update the most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, itr=epoch)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('Rew', with_min_and_max=True)  # Daniel: new
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            if update_alpha:
                logger.log_tabular('Alpha', with_min_and_max=True)
                logger.log_tabular('LossAlpha', with_min_and_max=True)
            logger.dump_tabular()

    if final_buffer:
        base = f'final_buffer-maxsize-{replay_size}-steps-{total_steps}-noise-{act_noise}-alpha-{alpha}.p'
        if update_alpha:
            base = base.replace('.p', f'-updateAlpha-{update_alpha}.p')
        save_path = osp.join(logger.get_output_dir(), 'buffer', base)
        print(f'\nSaving final buffer of data to:\n\t{save_path}')
        params_save = dict(logger=logger_kwargs, act_noise=act_noise)
        replay_buffer.dump_to_disk(path=save_path, params=params_save)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--final_buffer', action='store_true',
        help='Runs the final buffer setting from (Fujimoto et al., ICML 2019)')
    parser.add_argument('--update_alpha', action='store_true',
        help='In some implementations alpha gets updates to control actor entropy')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    # Don't call it this way, use Spinup's run script.
    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs)
