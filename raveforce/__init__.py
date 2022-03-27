from wasmer import engine, Store, Module, Instance, Memory, ImportObject, Function, FunctionType, Type
from wasmer_compiler_cranelift import Compiler
from urllib.request import urlopen
import numpy as np
import sys
import datetime
import struct, random

def make(code="", target=[], total_step=16, step_len=0.125, criteria="raw",
    action_space=[]):
    return Env(code=code, target=target, total_step=total_step, step_len=step_len,
    criteria=criteria, action_space=action_space)
#end

def now():
    return datetime.datetime.now()

class Env(object):
    def __init__(self, code="", target=[], total_step=16, step_len=0.125, criteria="raw",
    action_space=[]):
        self.elapse_block = 0
        self.code = code
        self.target = target
        self.total_step = total_step
        self.step_len = step_len
        self.criteria = criteria
        self.action_space = ActionSpace(action_space)
        self.loaded = False
        # some calculation
        self.para_num = code.count('{}')

    def reset(self, plot=False):
        # use wasmer-python
        if self.loaded:
            self.instance.exports.reset()
        else:
            import_object = ImportObject()
            store = Store(engine.JIT(Compiler))
            import_object.register(
                "env",
                {
                    "now": Function(store, now, FunctionType([], [Type.F64]))
                }
            )
            # self.module = Module(self.store, open('glicol_wasm.wasm', 'rb').read())
            # self.module = Module(self.store, urlopen('https://cdn.jsdelivr.net/gh/chaosprint/glicol/js/src/glicol_wasm.wasm', 'rb').read())
            binary = urlopen('https://cdn.jsdelivr.net/gh/chaosprint/glicol@v0.9.22/js/src/glicol_wasm.wasm').read()
            module = Module(store, binary)
            self.instance = Instance(module, import_object)
            self.loaded = True

        # empty_observation = np.zeros(int(self.step_len * self.total_step))
        self.audio = [[], []]
        empty_observation = np.zeros(len(self.target))
        # print(self.action_space.actions)
        # if plot:
        #     plt.plot(empty_observation)
        return empty_observation

    def step(self, action):

        
        inPtr = self.instance.exports.alloc(128)
        resultPtr = self.instance.exports.alloc_uint8array(256)

        self.audioBufPtr = self.instance.exports.alloc_uint8array(256)

        # send the code to the wasm
        code = self.code.format(*action)
        # print(self.criteria)
        code = bytes(code, "utf-8")
        codeLen = len(code)
        codePtr = self.instance.exports.alloc_uint8array(codeLen)
        self.memory = self.instance.exports.memory.uint8_view(codePtr)
        self.memory[0:codeLen] = code

        self.instance.exports.update(codePtr, codeLen)

        audioBufPtr = self.instance.exports.alloc(256)

        # start the engine
        # self.instance.exports.run_without_samples(codePtr, codeLen)
        self.num_block = int(self.step_len * 44100 / 128)
        # self.elapse_block += self.num_block

       
        # self.audio = []

        for _ in range(self.num_block):

            self.instance.exports.process(inPtr, audioBufPtr, 256, resultPtr)
            bufuint8 = self.instance.exports.memory.uint8_view(offset=int(audioBufPtr))[:1024]
            nth = 0
            buf = []
            while nth < 1024:
                byte_arr = bytearray(bufuint8[nth: nth+4])
                num = struct.unpack('<f', byte_arr)
                buf.append(num[0])
                nth += 4
            result = self.instance.exports.memory.uint8_view(offset=resultPtr)
            result_str = "".join(map(chr, filter(lambda x : x != 0, result[:256])))
            # deprecated in 2022
            # self.instance.exports.process_u8(self.audioBufPtr)
            # self.memory = self.instance.exports.memory.uint8_view(self.audioBufPtr)
            # self.buf = [struct.unpack('<f', bytearray(
            #     self.memory[i: i+4]))[0] for i in range(0,256,4)]
            self.audio[0].extend(buf[:128])
            # self.audio[1].extend(buf[128:256])

        padded_observation = self.padding_to_total()
        reward = self.calc_reward(padded_observation)
        print("step {} in total {}".format(self.step, self.total_step))
        done = self.step == self.total_step
        info = 0 if result_str == "" else result_str
        return padded_observation, reward, done, info

    def render(self):
        pass

    def padding_to_total(self):
        # pad_width = len(self.target) - len(self.audio[0])
        # print(len(self.target), len(self.audio))
        # padded = np.pad(self.audio, pad_width, "constant", constant_values=0)
        # print(padded)
        padded = np.zeros(len(self.target))
        padded[:len(self.audio[0])] = self.audio[0]
        return padded

    def calc_reward(self, padded_observation):
        if self.criteria == "raw":
            # print(padded_observation)
            # print(len(padded_observation), len(self.target))
            # print((padded_observation - self.target))
            # mse = 0.1
            mse = (np.square(padded_observation - self.target)).mean(axis=0)
            return -mse
        else:
            return 0

class ActionSpace(object):
    def __init__(self, actions):
        self.actions = actions

    def sample(self):
        result = list(range(len(self.actions)))
        for i, action in enumerate(self.actions):
            if action[0] == "lin":
                result[i] = action[1] + random.random() * (action[2] - action[1])
            elif action[0] == "exp":
                point_a = np.log(action[1]+sys.float_info. min)
                point_b = np.log(action[2])
                result[i] = np.exp(point_a + random.random()*(point_b - point_a))
            elif action[0] == "rel":
                func = action[2]
                by = result[action[1]]
                result[i] = func(by)
            elif action[0] == "choose":
                result[i] = random.choice(action[1])
            else:
                pass
        return result