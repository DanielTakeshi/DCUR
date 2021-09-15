# DCUR

This is the code used for the paper **DCUR: Data Curriculum for Teaching via Samples with Reinforcement Learning**, by Daniel Seita, Abhinav Gopal, Zhao Mandi, and John Canny. The project website: https://sites.google.com/view/teach-curr/home


# README

This code has been tested on Ubuntu 18.04. To install, first follow the
[SpinningUp installation instructions][1], including the MuJoCo installation.
For example you can do this:

```
wget "https://www.roboti.us/download/mujoco200_linux.zip"
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200
```

We're using MuJoCo 2.0 here. Put `mujoco200` inside a new `.mujoco` directory.
Make sure the MuJoCo license is on the machine, and that `.bashrc` points to the path,
for example:

```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/seita/.mujoco/mujoco200/bin
```

The first one above is only if we want to view rollout videos. Here's my typical setup:

```
.mujoco/
  mjkey.txt
  mujoco200/
    [code here...]
```

Once we have dependencies set up, install MuJoCo:

```
conda install patchelf
pip install -U 'mujoco-py<2.1,>=2.0'
pip install imageio-ffmpeg
```

If the pip install for `mujoco-py` is causing issues, try:

```
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

The first command installs a dependency that isn't listed on the instructions.
The last command is used in case we want videos from MuJoCo.
The command `pip install gym[mujoco,robotics]` doesn't work for me since it
assumes MuJoCo 1.5, but I'm able to run the MuJoCo environments anyway.

**Finally**, since we save a lot of directories, please add this to your
`.bashrc` to point to a machine-dependent path, ideally in disk space.

```
export SPINUP_DATA_DIR=/data/seita/spinup
```

Data from training RL will be saved in these directories.

## Training RL Algorithms

(These usually form the basis of "teachers".) See `bash/train_teachers.sh`. The
detailed version is as follows. Training vanilla TD3 teachers can be done like
this:

```
python -m spinup.run td3 --epochs 250 --env Ant-v3         --exp_name ant_td3
python -m spinup.run td3 --epochs 250 --env HalfCheetah-v3 --exp_name halfcheetah_td3
python -m spinup.run td3 --epochs 250 --env Hopper-v3      --exp_name hopper_td3
python -m spinup.run td3 --epochs 250 --env Walker2d-v3    --exp_name walker2d_td3
```

but again, see the bash scripts for precise commands.

See [the paper][3] for what to expect for reward, and the [official author code
here][4] (updated to Python 3.7) for additional suggested hyperparameters. The
standard is to train for 1 million time steps. The TD3 paper evaluates every 5K
time steps, where each evaluation is the mean of 10 episodes with *no*
exploration noise. In our case, we evaluate *every 4K steps, not 5K*.

We set `epochs=250` so there are `250 x 4000 = 1M` total environment steps.
Fujimoto uses a 256-256 hidden layer design for the actor and critic, which
matches our defaults. Also, "epoch" means that after each one, we report agent
performance on a test set. Within a single epoch, we do multiple rounds of data
collection and gradient steps (one gradient step per environment step).

The OpenAI SpinningUp docs say to use -v2, but *we should upgrade to -v3 for
all envs*. You can find details [of the versions here][6], and the relevant
source files also explain the states (observations).  This [GitHub issue
report][9] also helps -- Erwin Coumans explains the state representation.

See `bash/plot_teachers.sh` for plotting vanilla RL. They are saved in
`data/<exp_name>/<exp_name>_s<seed>/figures`.

## Results in the Paper

See `bash/paper` for the bash scripts that we used to run experiments for the paper.

## General Tips

For any new stuff for our project, please see the `spinup.teaching` module.

Note: because SpinningUp appears to consume significant amounts of CPU
resources, preface the code with:

```
taskset -c x-y
```

where `x` and `y` represent zero-indexed CPU indices to reveal to the code.

If we benchmark with BCQ, which we probably should, then [use code from the
author][5].


[1]:https://spinningup.openai.com/en/latest/user/installation.html
[2]:https://github.com/openai/mujoco-py
[3]:https://arxiv.org/abs/1802.09477
[4]:https://github.com/sfujim/TD3
[5]:https://github.com/sfujim/BCQ
[6]:https://github.com/openai/gym/blob/2d247dc93a8c98360ebeb6a3807a9b3d945424ee/gym/envs/__init__.py
[7]:https://github.com/openai/gym/issues/1541
[8]:https://github.com/openai/gym/issues/1636
[9]:https://github.com/openai/gym/issues/585
