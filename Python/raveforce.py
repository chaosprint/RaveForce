import OSC
import librosa
from librosa.feature import mfcc
from librosa.core import stft
from librosa.core import cqt
from librosa.effects import time_stretch
import numpy as np

SC_PORT = ('127.0.0.1', 57120)
PY_PORT = ('127.0.0.1', 57310)

class Synth(object):

##################################

    # OSC help function
    def snd_msg(self, addr, msgs):
        self.osc_msg = OSC.OSCMessage()
        self.osc_msg.setAddress(addr)
        for msg in msgs:
            self.osc_msg.append(msg)

        self.client = OSC.OSCClient()
        self.client.connect(SC_PORT)
        self.client.send(self.osc_msg)

    def rec_msg(self, addr, func):
        self.server = OSC.OSCServer(PY_PORT)
        self.server.addMsgHandler(addr, func)
        self.server.serve_forever()

##################################

    def calculate_reward(self, observation, target): # whole file with padded zeros
        target_length = len(target)

        self.relevant_target = target[:int(target_length * (self.current_step + 1) / self.total_step )]

        if (target_length - len(self.relevant_target)) > 0:
            padding = np.zeros(target_length - len(self.relevant_target))
            padded_relevant_target = np.append(self.relevant_target, padding)
        else:
            padded_relevant_target = self.relevant_target[:target_length]

        valid_target = time_stretch(padded_relevant_target, target_length/len(observation))

        if len(observation) < len(valid_target):
            valid_target = valid_target[:len(observation)]
        else:
            observation = observation[:len(valid_target)]

        method = "mfcc"
        if method is "mfcc":
            mse = np.square( mfcc(observation, n_mfcc=20) - mfcc(valid_target, n_mfcc=20) )
            mse = mse.mean(axis=0).mean(axis=0)

        elif method is "raw":
            # mse = np.square()
            mse = np.square(observation - valid_target)
            mse = mse.mean()
        else:
            if method is "stft":
                real = stft(observation).real - stft(valid_target).real
                imag = stft(observation).imag - stft(valid_target,).imag
                mse = np.concatenate((real, imag), axis=1)
            else:
                real = cqt(observation).real - cqt(valid_target).real
                imag = cqt(observation).imag - cqt(valid_target).imag
            mse = np.square(mse)
            mse = mse.mean(axis=0).mean(axis=0)
        reward = -mse
        return reward

##################################

    def __init__(self, env_info):
        # answer to the synet.make return a synth
        # need to check if SuperCollider is ready
        # the check can be finished later
        self.snd_msg('/make', env_info)
        self.rec_msg('/make', self.make_handler)

        if env_info[0] == "drum_loop":
            self.target, _ = librosa.load("drum_loop.wav", sr=None)
        else:
            self.target, _ = librosa.load("bd.wav", sr=None)


    def make_handler(self, addr, tags, data, client_addr):
        path = data[0].decode("utf-8")
        self.total_step = data[-1]
        self.observation_space, _ = librosa.load(path, sr=None)


        space_kind = data[1].decode("utf-8")
        if space_kind == 'io':
            self.action_space = IO()
        elif space_kind == 'combination':
            self.action_space = Combination( int( data[2]) )
        elif space_kind == 'continuous':
            self.action_space = Continuous( int( data[2]) )
        elif space_kind == 'note':
            self.action_space = Note()
        elif space_kind == 'time':
            self.action_space = Time()
        elif space_kind == 'freq':
            self.action_space = Freq()
        elif space_kind == 'amp':
            self.action_space = Amp()
        else:
            pass
        self.server.close()

################################

    def reset(self): # return the default parameter for the synth
        self.snd_msg('/reset', [])
        self.rec_msg('/reset', self.reset_handler)
        self.current_step = 0
        self.total_length = len(self.observation_space)
        return self.observation_space

    def reset_handler(self, addr, tags, data, client_addr):
        path = data[0].decode("utf-8")
        self.observation_space, _ = librosa.load(path, sr=None)
        self.server.close()

##################################

    def step(self, action):
        msg_list = [self.current_step]

        try:
            for i in action:
                msg_list.append(i.item()) # continuous
        except:
            msg_list.append(action)
        self.snd_msg('/step', msg_list)
        self.rec_msg('/step', self.step_handler)

        #observation
        self.current_length = len(self.observation_space)

        if (self.total_length - self.current_length) > 0:
            self.padding = np.zeros(self.total_length - self.current_length)
            padded_observation = np.append(self.observation_space, self.padding)
        else:
            padded_observation = self.observation_space[:self.total_length]


        reward = self.calculate_reward(padded_observation, self.target)

        #done
        if self.current_step is self.total_step - 1:
            done = True
        else:
            done = False

        self.current_step += 1

        #info
        info = {
            "current_step": self.current_step,
            "observation_space": self.observation_space,
            "relevant_target": self.relevant_target,
        }

        return padded_observation, reward, done, info

    def step_handler(self, addr, tags, data, client_addr):
        path = data[0].decode("utf-8")
        self.observation_space, _ = librosa.load(path, sr=None)
        self.server.close()

##################################
# play the sound, currently without GUI
    def render(self):
        self.snd_msg('/render', [])

##################################

class ActionSpace(object):
    def sample(self):
        return np.random.choice(self.space, 1).item()

class IO(ActionSpace):
    def __init__(self):
        self.space = np.array([0,1])
        self.n = 2

class Combination(ActionSpace):
    def __init__(self, n):
        self.space = np.linspace(0, n-1, n).astype(int)
        self.n = n

class Continuous(ActionSpace):
    def __init__(self, n):
        self.space = np.zeros(n)
        self.n = n
    def sample(self):
        return np.random.random_sample((self.n, ))

class Note(ActionSpace):
    def __init__(self):
        self.space = np.array(range(128))

class Freq(ActionSpace):
    def __init__(self):
        self.space = np.logspace(np.log10(20), np.log10(22000), 50)

class Amp(ActionSpace):
    def __init__(self):
        self.space = np.linspace(0.0, 1.0, num=21)

class Time(ActionSpace):
    def __init__(self):
        self.space = np.logspace(np.log10(0.002), 1, num=50)

def make(*env_info):
    return Synth(env_info)
