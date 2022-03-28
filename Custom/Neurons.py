from math import exp
import itertools


class Neuron:
    def __init__(self, theta=20, *args, **kwargs):
        self.theta = theta
        self.U = 0

        self.t_fired=[]
        self.internal_clock = 0

        self.post_syn = []
        self.pre_syn = []
        self.syn_input = 0

    def send_pulse(self):
        a=self.theta #* self.U//self.theta
        for syn in self.post_syn: syn.receive_pulse(a)

    def simulate_one_step(self, I_t=0):
        self.internal_clock += 1

        self.U += I_t + self.syn_input
        self.syn_input = 0

        if self.U >= self.theta:
            self.t_fired.append(self.internal_clock)
            self.send_pulse()
            self.U=self.U-self.theta#self.U%(self.theta)
            return self.U, I_t

        return self.U, I_t

