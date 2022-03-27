# RaveForce
RaveForce is an Python environment that allows you to define your musical task in [Glicol](https://glicol.org) synth, and train an agent to do the task with APIs similar to the [OpenAI Gym](https://gym.openai.com).

## Why RaveForce

It seems that `music generation` researches have been dominated by MIDI generation or audio generation with either supervised method via a corpus or unsupervised method with some sample library.

But let's consider a simple example: you want to train an agent to play the synth sequencer for you. The goal is to copy a famous bass line. Therefore, in each `step`, the `agent` needs to make a decision on which note to play and what kind of timbre to make. The agent can have an `observation` of what has been synthesised, and the `reward` calculated by comparing the similarity at the moment.

Yet it can be very diffucult and time-comsuming if we want to have a real-world environment (such as a music robot) setup. Another option is to use some built-in Python function to compose our `music tasks`, but a better way is to find a common place between our simulation and the real-world music practices.

Live coding is exactly such a practice where the artistic perform improvised algorithmic music by writing program code in real-time.

Glicol is a new live coding language that can be accessed in the browsers:

https://glicol.org

The syntax of Glicol is very similar to synth or sequencers, which perfectly fits our needs. Plus, Glicol is written in Rust and can be used in Python easily (which is already wrapped in this pip package).

Therefore, the final architecture is:

```
Agent
-> Generate Glicol code
-> Glicol does the non-real-time synthesis in Python
-> Get the reward, observation space, etc.
```

This process should involve some deep neural network as the synthesised audio is much difficult to process than the symbolic sequences.

For more background, please refer to this paper:
> Lan, Qichao, Jim Tørresen, and Alexander Refsum Jensenius. "RaveForce: A Deep Reinforcement Learning Environment for Music Generation." (2019).
```
@article{lan2019raveforce,
  title={RaveForce: A Deep Reinforcement Learning Environment for Music Generation},
  author={Lan, Qichao and Torresen, Jim and Jensenius, Alexander Refsum},
  year={2019}
}
```
> Note that the implementation of this paper has been moved to the `sc` branch

## How to use RaveForce

### Install
This is quite straightforward:
`pip install raveforce`

### Be familiar with Glicol syntax.

Visit Glicol website to get familiar with its syntax and concept:

https://glicol.org

### Python
Since we are going to define our own musical task, we should make some changes to the `make` method.

Let's consider the simplest example: just let the agent to play for 1 step, tweaking `attack`, `decay` and `freq` of a sine wave synth to simulate a kick drum.

```python
import raveforce
import librosa

target, sr = librosa.load("YOUR_KICK_DRUM_SAMPLE", sr=None)
dur = len(target) / sr

env = gym.make(
    """
     ~env: imp 0.1 >> envperc {} {}
    
    kick_drum: sin {} >> mul ~env
    """,
    total_step=1,
    step_len=dur,
    target = target,
    action_space=[
      ["lin", 0.0001, dur-0.0001], 
      ["rel", 0, lambda x: dur-0.0001-x], # related to para 0
      ["exp", 10, 10000]
    ]
)
```

Then, use as a normal `Gym` env:
```python
observation = env.reset()
action = env.action_space.sample()
print(action)

observation, reward, done, info = env.step(action)
plt.plot(observation) # make your own import matplotlib
print(reward, done, info)
```

I also made [an interactive exmaple on the Google Colab](https://colab.research.google.com/drive/1mngiLHKrtCs4V2yfSfeILByCTtmdkPoJ?usp=sharing), you can play around with it.