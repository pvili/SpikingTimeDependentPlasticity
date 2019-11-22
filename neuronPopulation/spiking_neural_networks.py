
import numpy as np 
import heapq
import random
import neuron_models as nm


neurons_per_population = 100#100
synapses_per_neuron = 100#400

delay_SP_max = 200#100
delay_SP_min = 1

weight_SP_max = 10
weight_SP_min = 0

fraction_negative = 0.2
ratio_inhibition_weight = 4

noise_initialization = 300 
time_interval_noise = [-20,20]

jitter = 0

max_simulation_time = 1000000

recurrentConnections = False
rec_synapses_per_neuron = 10
recurrentDelayRange = [1, 5]

class LIF_Network:
       
    def __init__(self, neurons_per_pop = neurons_per_population, 
                 synapses_per_neuron = synapses_per_neuron, 
                 populations_count = 1): 
        neuron_origin = nm.Neuron_LIF(0)
        self.count_neurons = neurons_per_pop
        self.neurons = []
        self.synapses = []
        self.spikes_heap = []
        self.neurons.append(neuron_origin)

        for neuron_idx in range(1,neurons_per_pop+1):
            neuron = nm.Neuron_LIF(neuron_idx)
            self.neurons.append(neuron)

            for syn in range(synapses_per_neuron):
                delay = random.uniform(delay_SP_min, delay_SP_max)
                weight = getRandomWeight()
                synapse = nm.Synapse( neuron_origin, neuron, delay, weight)
                self.synapses.append(synapse)   
                
        if recurrentConnections:
            for neuron in self.neurons:
                r = 0
                while r <rec_synapses_per_neuron:
                    neur_idx = np.random.randint(1,self.count_neurons)
                    if neur_idx != neuron.id:
                        neuron_origin = self.neurons[neur_idx]
                        delay = random.uniform(recurrentDelayRange[0], recurrentDelayRange[1])
                        weight = getRandomWeight()
                        synapse = nm.Synapse( neuron_origin, neuron, delay, weight)
                        self.synapses.append(synapse)   
                        r += 1
                        synapse = nm.Synapse( neuron_origin, neuron, delay, weight)
                
                
        for pop in range(1, populations_count):
            min_neuron_idx = neurons_per_pop*pop + 1
            max_neuron_idx = min_neuron_idx + neurons_per_population
            for neuron_idx in range(min_neuron_idx,max_neuron_idx):
                neuron = nm.Neuron_LIF(neuron_idx)
                self.neurons.append(neuron)
                
                presyn_idx_min = min_neuron_idx - neurons_per_pop
                presyn_idx_max = min_neuron_idx -1
                for syn in range(synapses_per_neuron):
                    presyn_idx = np.random.randint(presyn_idx_min, presyn_idx_max)
                    presyn_neuron = self.neurons[presyn_idx]
                    delay = random.uniform(delay_SP_min, delay_SP_max)
                    weight = getRandomWeight()
                    synapse = nm.Synapse( presyn_neuron, neuron, delay, weight)
                    self.synapses.append(synapse)   

    def active_stdp(self, stdp_active):
        for syn in self.synapses:
            syn.stdp_active = stdp_active
    
    def active_learning(self, learning_binary):
        self.active_stdp(learning_binary)
    
    def count_active_synapses(self):
        count = 0
        for syn in self.synapses:
            if syn.active_synapse:
                count +=1
        return count
    
    def initial_noise(self):
        for r in range(0,noise_initialization):
            self.execute_noise()
    
    def execute_noise(self, noise_prob = 1.):
        t_max = time_interval_noise[1]
        t_min = time_interval_noise[0]
        for syn in self.synapses:    
            if random.random() < noise_prob:
                delta_t = t_min + (t_max - t_min)*np.random.rand()
                syn.stdp_noise(delta_t)
    
    def add_spikes(self, spike_list):
        for s in spike_list:
            heapq.heappush(self.spikes_heap,s)
            
    def rewire(self, timeInterval):
        t_0 = min(timeInterval)
        t_max = max(timeInterval)
        for syn in self.synapses:
            syn.delay = random.uniform(t_0, t_max)
    
    def init_activity(self, input_neuron_list):
        
        for neuron in input_neuron_list:
            spike = nm.generate_input_spike(neuron)
            heapq.heappush(self.spikes_heap, spike)
    
    def init_stimulus(self):
        spike = nm.generate_input_spike(self.neurons[0])
        heapq.heappush(self.spikes_heap, spike)
    
    def reset_activity(self):
        del self.spikes_heap[:] 
        for n in self.neurons:
            n.reset_activity()
        for s in self.synapses:
            s.last_spike = -1
            
    def incomming_synapses_per_neuron(self):
        list_syn_per_neuron = [[] for n in range(self.count_neurons+1)]
        for syn in self.synapses:
            postsyn_neuron_id = syn.postsynaptic_neuron.id
            list_syn_per_neuron[postsyn_neuron_id].append(syn)
        return list_syn_per_neuron
    
    def count_rate_per_neuron(self, list_post_spikes):
        list_rate_per_neuron = [0 for n in range(self.count_neurons+1)]
        for spk in list_post_spikes:
            neuron = spk[0]
            list_rate_per_neuron[neuron] += 1
        return list_rate_per_neuron
        
    def normalize_weights(self, list_post_spikes):
        syn_list = self.incomming_synapses_per_neuron()
        rate_list = self.count_rate_per_neuron(list_post_spikes)
        
        for n in range(self.count_neurons+1):
            rate = rate_list[n]
            for syn in syn_list[n]:
                syn.weight_decay(rate)

    def run(self):
        
        spike_count = 0
        #spikes_list_output = []
        neurons_fired = []
        
        t = 0
        
        while self.spikes_heap and t < max_simulation_time:
            
            spike_count = spike_count +1
            spike = heapq.heappop(self.spikes_heap)
            t = spike.arrival_time
            #spikes_list_output.append(spike)
            execution_result = spike.execute()
            
            if execution_result['firing_neuron']: #meaning the receiving neuron fired
                
                firing_time = spike.arrival_time + 1
                firing_neuron = spike.synapse.postsynaptic_neuron
                neurons_fired.append((firing_neuron.id, firing_time))
                

            if execution_result['new_spikes']: #meaning new spikes were triggered
                firing_neuron = spike.synapse.postsynaptic_neuron.id
                firing_time = spike.arrival_time
                self.add_spikes(execution_result['new_spikes'])
                
        return neurons_fired
         
    
    def print_network(self):
        for n in self.neurons:
            n.print_neuron()
        
    def print_synapses(self):
        for syn in self.synapses:
            syn.print_synapse()
            
            
    
    
def get_permutation(min_int, max_int, number_elements):
    population = range(min_int, max_int)
    return random.sample(population, number_elements)    

def getRandomWeight():
    p= random.uniform(0.,1.)
    if p<fraction_negative:
        w = random.uniform(ratio_inhibition_weight*weight_SP_min,ratio_inhibition_weight*weight_SP_max)
        return -w
    else:
        w = random.uniform(weight_SP_min, weight_SP_max)
        return w
