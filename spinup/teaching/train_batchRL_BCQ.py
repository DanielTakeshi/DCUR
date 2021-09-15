import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spinup.utils.logx import EpochLogger
from spinup.algos.pytorch.td3.td3 import ReplayBuffer
import time
import os.path as osp
import joblib
import gym


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        self.phi = phi


    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)


    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim


    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(state, z)

        return u, mean, std


    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).clamp(-0.5,0.5)
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

def load_pytorch_policy(fpath, itr, deterministic=False):
    """Load a pytorch policy saved with Spinning Up Logger."""
    # TODO: Implement fix that doesn't require full path to be passed in
    # Need to standardize the way models are saved.
    fname = fpath
    print('\n\nLoading from %s.\n\n'%fname)
    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action

def load_policy_and_env(fpath, itr='last'):
    
    """Load a policy from save, whether it's TF or PyTorch, along with RL env.
    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations. We're using PyTorch only here.
    Abhinav: For now, this links directly to the folder where the agents data is stored.
    """

    # load the get_action function
    get_action = load_pytorch_policy(fpath + '/pyt_save/model.pt', itr)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action

def generate_data(env, teacher_path, buffer_size, teacher_noise, output_dir):
    env, get_action = load_policy_and_env(teacher_path, buffer_size if buffer_size >=0 else 'last')
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    rbuffer = ReplayBuffer(obs_dim, act_dim, size=buffer_size)
    # for now, we will do only uniform normal sigma
    #TODO: Add different types of noise depending on need
    sigma = teacher_noise
    print(f'\nMade new buffer w/obs: {obs_dim}, act: {act_dim}, act_limit: {act_limit}.')
    print(f'Will run as many time steps until {buffer_size} items...\n')
    print(f'Noise distr normal with: {teacher_noise}, starting sigma: {sigma:0.3f}')

    # The usual initialization.
    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

    for _ in range(buffer_size):
        # Get noise-free action from the policy.
        a = get_action(o)

        # Perturb action based on our desired noise parameter. We clip here. The policy
        # net ALREADY has a tanh at the end, but we need another one for the added noise.
        a += sigma * np.random.randn(act_dim)
        a = np.clip(a, -act_limit, act_limit)

        # Do the usual environment stepping.
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Store information for usage later.
        rbuffer.store(o, a, r, o2, d)

        # Update most recent observation [not done in test_policy, but is in td3.py,
        # mainly to make rbuffer.store() just use one call].
        o = o2

        if d:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d \t Sigma %.3f' % (n, ep_ret, ep_len, sigma))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()
    buffer_file = output_dir + '/teacher_buffer' + '_' + str(sigma) + '.pt'
    rbuffer.dump_to_disk(buffer_file)
    return buffer_file
    
def BCQ(env_fn, buffer_path, seed = 0, steps_per_epoch = 4000, epochs = 100, pi_lr=1e-3, q_lr=1e-3, batch_size=100, update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, replay_size=int(1e6), buffer_size=int(1e6),
        logger_kwargs=dict(), save_freq=25, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05, final_buffer=False):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)
    env, test_env = env_fn(), env_fn()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1
    latent_dim = action_dim * 2
    
    replay_buffer = ReplayBuffer(obs_dim=state_dim, act_dim=action_dim, size=replay_size)
    replay_buffer.load_from_disk(buffer_path, buffer_size)
    
    actor = Actor(state_dim, action_dim, max_action, phi)
    actor_target = copy.deepcopy(actor)
    actor_optimizer = torch.optim.Adam(actor.parameters(), pi_lr)

    critic = Critic(state_dim, action_dim)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = torch.optim.Adam(critic.parameters(), q_lr)

    vae = VAE(state_dim, action_dim, latent_dim, max_action)
    vae_optimizer = torch.optim.Adam(vae.parameters()) 

    max_action = max_action
    action_dim = action_dim
    discount = discount
    tau = tau
    lmbda = lmbda
    
    def select_action(state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1)
            action = actor(state, vae.decode(state))
            q1 = critic.q1(state, action)
            ind = q1.argmax(0)
        return action[ind].cpu().data.numpy().flatten()
    
    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(select_action(o))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    timer = 0
    
    for it in range(total_steps):
        # Sample replay buffer / batch
        data = replay_buffer.sample_batch(batch_size)
        state, action, reward, next_state, not_done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        # Variational Auto-Encoder Training
        recon, mean, std = vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()
        
        # Critic Training
        with torch.no_grad():

            # Duplicate next state 10 times
            next_state = torch.repeat_interleave(next_state, 10, 0)
            vae.decode(next_state)
            target_Q1, target_Q2 = critic_target(next_state, actor_target(next_state, vae.decode(next_state)))
            # Soft Clipped Double Q-learning 
            target_Q = lmbda * torch.min(target_Q1, target_Q2) + (1. - lmbda) * torch.max(target_Q1, target_Q2)
            # Take max over each action sampled from the VAE
            target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
            target_Q = reward.reshape(-1, 1) + not_done.reshape(-1, 1) * discount * target_Q

        current_Q1, current_Q2 = critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Pertubation Model / Action Training
        sampled_actions = vae.decode(state)
        perturbed_actions = actor(state, sampled_actions)

        # Update through DPG
        actor_loss = -critic.q1(state, perturbed_actions).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        # Update Target Networks 
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        if (it+1) % steps_per_epoch == 0:
            epoch = (it+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
#             logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
#             logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', it)
#             logger.log_tabular('Q1Vals', with_min_and_max=True)
#             logger.log_tabular('Q2Vals', with_min_and_max=True)
#             logger.log_tabular('LossPi', average_only=True)
#             logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
        timer += 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_path', type=str)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--n_steps', type=int, default=1000000)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--teacher_noise', type=float, default = 0.1)
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--buffer_path', type=str, default = None)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    args = parser.parse_args()
    
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    output_dir = logger_kwargs['output_dir']
    print(output_dir)
    #load teacher and rollout
    if not args.buffer_path:
        teacher_buffer_path = generate_data(args.env, args.teacher_path, args.buffer_size, args.teacher_noise, output_dir)
    else:
        teacher_buffer_path = args.buffer_path
    student = BCQ(lambda : gym.make(args.env), buffer_path=teacher_buffer_path, discount=args.gamma,
        seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, steps_per_epoch=args.steps_per_epoch)