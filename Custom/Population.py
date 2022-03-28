import random
from numpy import arange


class Synapse:
    def __init__(self, pre_n, post_n, w=0):
        self.pre_n = pre_n
        self.post_n = post_n
        self.synapse_charge = []
        self.inject_pulse = False
        self.w = w
        self.last_inject_time = []

    def receive_pulse(self, input):
        self.synapse_charge.append(input*self.w)

    def simulate_synapse(self, stdp_eng=None):
        if self.synapse_charge==[]: return None

        if stdp_eng is not None and self.inject_pulse:
            stdp_eng.train_pre_to_post_syn(self)
        self.inject_pulse=False

        for input_u in self.synapse_charge:
            self.post_n.syn_input += input_u * self.post_n
            self.inject_pulse = True
            self.last_inject_time.append(self.pre_n.internal_clock)
        self.synapse_charge=[]


class CustomModel:
    def __init__(self, stdp_eng=None, *args, **kwargs):
        self.neurons = []
        self.stdp_eng=stdp_eng
        self.layers = []
        self.populate_neurons(*args, **kwargs)

        self.create_network()

    def populate_neurons(self, n_type, n_config=None, excit_count=None, inhib_count=None, *args, **kwargs):
        for i in range(excit_count):
            self.neurons.append(eval('n_type(is_exc=True, ' + n_config + ')'))
        for i in range(inhib_count):
            self.neurons.append(eval('n_type(is_exc=False, ' + n_config + ')'))

    def create_network(self, layer_size):
        ii=0
        for i in layer_size:
            self.layers.append(self.neurons[ii:i+ii])
            ii+=i

        for l1,l2 in zip(self.layers,self.layers[1:]):
            for pre_neuron in l1:
                for post_neuron in l2:
                    self.connect_neurons(pre_neuron, post_neuron, w)

    def simulate_network_one_step(self):
        for neuron in self.neurons:
            for post_s in neuron.post_syn: post_s.simulate_synapse(self.stdp_eng)

        u_history = []
        i_history = []
        for neuron in self.neurons:
            inter_U, curr = neuron.simulate_one_step(0)
            u_history.append(inter_U)
            i_history.append(curr)

        if self.stdp_eng is not None:
            for neuron in self.neurons:
                if neuron.last_fired: self.stdp_eng.train_post_to_pre_syn(neuron)

        if neuron.internal_clock % 20 == 0: print(neuron.internal_clock)

        return u_history, i_history

    def connect_neurons(self, pre_neuron, post_neuron, w):
        syn=Synapse(pre_neuron, post_neuron, w)
        pre_neuron.post_syn.append(syn)
        post_neuron.pre_syn.append(syn)

    def reset_synapse_charge(self):
        for neuron in self.neurons:
            for post_syn in neuron.post_syn: post_syn.synapse_charge=[]
