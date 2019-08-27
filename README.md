# RaveForce
RaveForce is an environment that allows you to define your musical task in SuperCollider and train an agent to do the task in Python. It aims to be an (OpenAI Gym](https://gym.openai.com/) alike environment for music.

## SuperCollider

To use it, copy the ```Extensions``` folder to the SuperCollider ```Open user support directory```. You can find it in the ```File``` menu of the SuperCollider IDE.

Recompile the SuperCollider library. You can find the command and shortcut in the ```Language``` menu of SuperCollider IDE.

An example code to run is as below:

```
(
SynthDef(\sin, { |freq, att=0.01, rel=0.49|
	var sig, freq_range;
	freq_range = freq.linexp(0, 1, 20, 22000);
	sig = SinOsc.ar(freq_range!2);
	sig = sig * Env.perc(att, rel).kr(2);
	Out.ar(0, sig);
}).load;

p = Pbind(
	\instrument, \sin,
	\freq, 2000,
	\dur, 0.5,
);

RaveForce.start(p, key: \freq, bpm:30, total_step:1);

// start the task with the pattern we define
// key states which parameters to control, in this example, the frequency is the only one for the agent to choose
)
```

The total step of one means every time the pattern reaches the total step, the pattern will be rendered.

Typically, the workflow is:

- Python sends parameters
- SC reads the paramters and render the pattern in non-real-time with the length of current step
- Python reads the audio output and train the neural networks
- Repreat

## Python

After the SuperCollider code is running, command can be sent from Python.

Jupyter notebook is recommended for the training.

```
import raveforce

env = raveforce.make("continuous", 1) // one continuous parameter is between 0 to 1
observation = env.reset()
action = env.action_space.sample()
observation, reward, done, info = env.step(action)
env.render()
```
