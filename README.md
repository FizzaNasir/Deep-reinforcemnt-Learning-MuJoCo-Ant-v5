## Presentation Link:
[Link Text](https://docs.google.com/presentation/d/1mhL4XpBBhYXzObotYD-8V7mg0e_0Zsku/edit?usp=drive_link&ouid=105810472473991072710&rtpof=true&sd=true)

## Ant Robot Overview
A 3D quadruped robot designed for locomotion tasks 
Consists of a central torso with free rotational movement 
Has four legs, symmetrically attached to the torso 
Each leg is made up of two connected body segments

## Mechanical Structure
Total 9 body parts (1 torso + 8 leg segments) 
Connected via 8 hinge joints 
Hinges allow rotational movement between segments 
Movement is controlled by applying torque at each joint

| Action space  | Observation Space |
|---------------|-------------------|
| Box(-1.0, 1.0, (8,), float32)      | Box(-inf, inf, (105,), float64)

## Ants with 3-legs
The Ant.xml of MuJoCo env was modified to
Remove one of its limbs. 
The required configurations we had to made was:
Modification in action and observation space:

#### Original Obs space: (105,)
After removing one leg it became: (83,)

#### Solution:
#### ObserverWrapper:
Intercept every observation coming from environment and modify it before giving it to the agent.

self.observation_space = Box(shape=(target_dim,))

<!-- <video width="700" controls> 
  <source src="https://github.com/FizzaNasir/Deep-reinforcemnt-Learning-MuJoCo-Ant-v5/blob/master/3_legged_trained/rl-video-episode-0.mp4" type="video/mp4">
</video> -->

[▶ Watch Video](https://raw.githubusercontent.com/FizzaNasir/Deep-reinforcemnt-Learning-MuJoCo-Ant-v5/master/3_legged_trained/rl-video-episode-0.mp4)



Smaller observation => pad it with zero to make it equal to 105
Larger observation => clip it off

Modification in Action space:Original action space: Box(-1, 1, (8,), float32)

After removing one leg, it has been reduced. So it needed action adaption also:def adapt_action(action, env):
    return action[:env.action_space.shape[0]]

 action, _ = model.predict(obs, deterministic=True)
 action = adapt_action(action, env)
 obs, reward, terminated, truncated, info = env.step(action)


## Results of training 3 legged ANT using transfer learning
 We evaluated the ant with 3 legs did work well
With transfer learning as expected.
 we tried training the 3 legged Ant with 3M steps 
and tuned parameters, but its didn’t work.

Inference:
The 3 legged ant might need need reward shaping
Its possible that its action space need modifications 
that allow it to balance with 3 legs.
The problem is not solvable at all

|               | Mean Reward | STD |
|---------------|-------------------|---------------------|
| 4 legged Ant   | 2.260624327196621 | 1.1535515161764045 |
| 3 legged Ant   | 0.226867835292213 | 0.6277322313113424 |


## Model Used : PPO
Policy used: MlpPolicy,
learning_rate=3e-4,
n_steps=2048,
batch_size=64,
gamma=0.99
Total_timesteps = 1000000

### Features:
PPO
Learns optimal actions by interacting with the environment 
Uses a clipped objective function to limit large policy updates 
Improves training stability and reliability 
Balances exploration (trying new actions) and exploitation (using learned actions) 

## Performance on Number of steps
### 4 legged Ant on 10k steps:

[▶ Watch Video]("https://github.com/FizzaNasir/Deep-reinforcemnt-Learning-MuJoCo-Ant-v5/blob/master/ppo_videos/trained-ant-100000.mp4")

### 4 legged Ant on 50k steps:

[▶ Watch Video]("https://github.com/FizzaNasir/Deep-reinforcemnt-Learning-MuJoCo-Ant-v5/blob/master/ppo_videos/trained-ant-50000_steps.mp4")

### 4 legged Ant on 1M steps:

[▶ Watch Video]("https://github.com/FizzaNasir/Deep-reinforcemnt-Learning-MuJoCo-Ant-v5/blob/master/ppo_videos/trained-ant-1M.mp4")

## Training results
We evaluate the 3 legged ant using the model that was used to train 4 legged ant to see how would it perform and below were the results:

### 4 legged Ant:
[▶ Watch Video]("https://github.com/FizzaNasir/Deep-reinforcemnt-Learning-MuJoCo-Ant-v5/blob/master/ppo_videos/trained-ant-1M.mp4")

### 3 legged Ant:
[▶ Watch Video]("https://github.com/FizzaNasir/Deep-reinforcemnt-Learning-MuJoCo-Ant-v5/blob/master/3_legged_trained/rl-video-episode-0.mp4")


## A2C vs PPO
We evaluated the model with A2C and PPO by giving same paramters to them. It turned out that PPO works better than A2C under same number of steps and model parameters

### PPO:
[▶ Watch Video]("https://github.com/FizzaNasir/Deep-reinforcemnt-Learning-MuJoCo-Ant-v5/blob/master/ppo_videos/trained-ant-1M.mp4")

### A2C:
[▶ Watch Video]("https://github.com/FizzaNasir/Deep-reinforcemnt-Learning-MuJoCo-Ant-v5/blob/master/ppo_videos/A2C-1M-steps.mp4")


## Reward plots (a2c vs ppo)
<img width="1348" height="815" alt="image" src="https://github.com/user-attachments/assets/27aa7878-c92d-415b-91fa-6ba28138d03c" />

## Convergence comparison A2C vs PPO
<img width="1189" height="860" alt="image" src="https://github.com/user-attachments/assets/a62893be-ff32-4a1a-928c-17c7d3dabff8" />


## Conclusion:
A2C is known for its fast but unstable learning. PPO is known for its stable learning due to its clipping updates feature. Thats why the rewards for A2C spikes up at start but became zero at the end.
