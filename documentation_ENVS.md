# Environment Docuentation

State/action dimensions:

- **HalfCheetah-v3**: state = 17, action = 6.
- **Hopper-v3**: state = 11, action = 3.
- **Walker2d-v3**: state = 17, action = 6. (Same as HalfCheetah-v2)
- **Ant-v3**: state = 111, action = 8.

All of these have actions bounded within [-1,1] for each component.

HalfCheetah is a fixed episode length environment, with all episodes at 1000
steps (1e+03). The others start with shorter-length episodes and only last
longer once the agent avoids the termination condition (which usually happens
quickly for TD3).

Ant-v3 uses contact forces in its state representation, in addition to position
and velocity.

For example, with TD3 we have this for the actor and Q-networks with Ant (state
space 111 as reviewed here):

```
MLPActor(
  (pi): Sequential(
    (0): Linear(in_features=111, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=256, bias=True)
    (3): ReLU()
    (4): Linear(in_features=256, out_features=8, bias=True)
    (5): Tanh()
  )
)

MLPQFunction(
  (q): Sequential(
    (0): Linear(in_features=119, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=256, bias=True)
    (3): ReLU()
    (4): Linear(in_features=256, out_features=1, bias=True)
    (5): Identity()
  )
)
```

There are two Q-networks. They both have identical parameter counts as above.

The policy will have a tanh layer at the end. Then, we later multiply this by
the action limit value to scale values to the appropriate range.
