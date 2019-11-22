import numpy as np 
import heapq
import random
import neuron_models as nm


neurons_per_population = 500#100
synapses_per_neuron = 400#400

delay_SP_max = 200#100
delay_SP_min = 1

weight_max = 10
weight_min = 0

fraction_negative = 0.2
ratio_inhibition_weight = 3

noise_initialization = 500
time_interval_noise = [-20,20]

time_max_sim = 3000

class LIF_Network:
       
    def __init__(self, neurons_per_pop = [neurons_per_population], 
                 connections = [[0, 1, 400, [0, 400], [10, 10]]]): 
        neuron_origin = nm.Neuron_LIF(0)
        self.count_neurons = sum(neurons_per_pop)
        self.neurons = []
        self.synapses = []
        self.spikes_heap = []
        self.neurons.append(neuron_origin)
        self.population_idx = []

        for neuron_idx in range(1,self.count_neurons +1):
            neuron = nm.Neuron_LIF(neuron_idx)
            self.neurons.append(neuron)
        
        curr_idx = 0
        self.population_idx.append([0,0])
        for pop_size in neurons_per_pop:
            new_idx = curr_idx + pop_size
            self.population_idx.append([curr_idx+1,new_idx])
            curr_idx = new_idx

        for con in connections:
            source = con[0]
            dest = con[1]
            synapseCount = con[2]
            time_interval = con[3]
            weight_interval = con[4]
            
            dest_idx_first = self.population_idx[dest][0]
            dest_idx_last = self.population_idx[dest][1]
            source_idx_first = self.population_idx[source][0]
            source_idx_last = self.population_idx[source][1]
            
            for dest_idx in range(dest_idx_first, dest_idx_last):
                for syn in range(synapseCount):
                    origin_idx = random.randint(source_idx_first, source_idx_last)
                    delay = random.uniform(time_interval[0], time_interval[1])
                    weight = getRandomWeight(weight_interval[0], weight_interval[1])
                    synapse = nm.Synapse( self.neurons[origin_idx], self.neurons[dest_idx], delay, weight)
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
            self.execute_noise(time_interval_noise)
    
    def execute_noise(self, time_interval):
        t_max = max(time_interval)
        t_min = min(time_interval)
        delta_t = t_min + (t_max - t_min)*np.random.rand()
        for syn in self.synapses:    
            delta_t = t_min + (t_max - t_min)*np.random.rand()
            syn.stdp_noise(delta_t)
    
    def add_spikes(self, spike_list):
        for s in spike_list:
            heapq.heappush(self.spikes_heap,s)
    
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

    def run(self):
        
        spike_count = 0
        #spikes_list_output = []
        neurons_fired = []
        firing_time = 0

        while self.spikes_heap and firing_time < time_max_sim:
            
            spike_count = spike_count +1
            spike = heapq.heappop(self.spikes_heap)
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
                
        return neurons_fired#, 'spikes':spikes_list_output }
         
    def print_network(self):
        for n in self.neurons:
            n.print_neuron()
        
    def print_synapses(self):
        for syn in self.synapses:
            syn.print_synapse()
            
            
    
    
def get_permutation(min_int, max_int, number_elements):
    population = range(min_int, max_int)
    return random.sample(population, number_elements)    

def getRandomWeight(weightMin, weightMax):
    p= random.uniform(0.,1.)
    if p<fraction_negative:
        w = random.uniform(ratio_inhibition_weight*weightMin,ratio_inhibition_weight*weightMax)
        return -w
    else:
        w = random.uniform(weightMin, weightMax)
        return w
