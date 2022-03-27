# RaveForce
RaveForce is an Python environment that allows you to define your musical task in [Glicol](https://glicol.org) synth, and train an agent to do the task with APIs similar to the [OpenAI Gym](https://gym.openai.com).

## Why Gym for Music

OpenAI Gym is famous as a toolkit for reinforcement learning, and can also be used for exploring other algorithms such as evolutionary algorithms.

In these algorithms, an `agent` is trained through interacting with and getting `reward` from the `environment`.

Similar pattern can be found in music production and performance, especially on beat-based music.

Musician makes decision at each beat on which notes and what kind of timbre to play.

This might not be the case for physical instrument (e.g. piano, guitar) players from psychology or musicology perspective.

However, if we consider how electronic music is made, it starts to make sense.

For electronic instruments like drum machine or sequencer, the player needs to make those decision as abovementioned.

Yet to simplify and model this process, we can study live coding music.

> Live coding is a new form of music performance where the artists write computer program code to make music in real-time.


## How to use RaveForce

First, you should be familiar with Glicol syntax.

Since we are going to define our own musical task, we should make some changes to the `make` method.