decay_constant = 0.1
refractory_period = 8. 
resting_potential = -70.0
firing_threshold = -50.0
#reset_potential = -65.0

non_existant_time = -1 #indicates that nothing has happened yet
irresistible_spike = 1000.0 #meaning that we make sure that the neuron fires

#synapse parameters
synaptic_weight_default = 5.0 

#stdp parameters from the scholarpedia 
stdp_weight_min = 0.001 
stdp_excitatory_weight_max = 10.0 
stdp_inhibitory_weight_max = 30.0 #15
stdp_excitatory_increase_nu = 0.01#0.01
stdp_excitatory_decrease_nu = 0.015#0.0075#0.015
stdp_inhibitory_increase_nu = 0.03#0.04
stdp_inhibitory_decrease_nu = 0.045#0.024#0.048
stdp_tau = 20.0
delete_synapse_threshold = 0.0 #prunning

jitter = 0

import numpy as np 

class Spike:
    def __init__(self, synapse, arrival_time = 0, voltage=10.0):
        self.arrival_time = arrival_time
        self.synapse = synapse
        self.voltage = voltage
    
    def __lt__(self,other):
        return self.arrival_time < other.arrival_time

    def execute(self):
        
        #self.print_spike()
        self.synapse.last_spike_arrived = self.arrival_time
        new_spikes = self.synapse.postsynaptic_neuron.receive_spike(self)
        
        if new_spikes: #meaning that the postsynaptic neruon fired
            firing_neuron = self.synapse.postsynaptic_neuron
            if new_spikes == 1: #meanning that the neuron had no outgoing synapses
                new_spikes = []
            return {'new_spikes':new_spikes, 'firing_neuron': firing_neuron}
        else:
            self.synapse.last_spike = self.arrival_time
            return {'new_spikes':0, 'firing_neuron': 0}
    
    def print_spike(self):
        string = ('Spike from ' + str(self.synapse.presynaptic_neuron.id)+
                  ' to ' + str(self.synapse.postsynaptic_neuron.id) +
                  ' arriving at t='+ str(self.arrival_time) )
        print(string)
        
def generate_input_spike(input_neuron):
    virtual_neuron = Neuron_LIF(-1)
    virtual_synapse = Synapse(virtual_neuron, input_neuron, 0, irresistible_spike)
    return Spike(virtual_synapse, 0, irresistible_spike)




class Synapse:
    def __init__(self, presynaptic_neuron, postsynaptic_neuron, 
                 delay=5, weight=synaptic_weight_default, stdp_active = True, last_spike =-1):
        self.delay = delay 
        self.weight = weight
        self.excitatory = weight>0
        self.presynaptic_neuron = presynaptic_neuron
        self.postsynaptic_neuron = postsynaptic_neuron
        #self.last_spike_arrived = last_spike
        self.active_synapse = True
        self.stdp_active = stdp_active
        
        if self.presynaptic_neuron.id != -1:
            self.postsynaptic_neuron.incomming_synapses.append(self)
        self.presynaptic_neuron.outgoing_synapses.append(self)
            
    def stdp(self):
        if self.weight != irresistible_spike and self.stdp_active:
            presynaptic_time = self.presynaptic_neuron.last_firing + self.delay
            postsynaptic_time = self.postsynaptic_neuron.last_firing
            if presynaptic_time != non_existant_time and postsynaptic_time != non_existant_time:

                stdp_reinforces = presynaptic_time < postsynaptic_time
                A = self.A_stdp(stdp_reinforces)
                stdp_decay = np.exp(-abs(postsynaptic_time - presynaptic_time)/stdp_tau )
                self.change_weight( A*stdp_decay)
                
        if abs(self.weight) < delete_synapse_threshold:
            self.invalidate_synapse()
     
    def stdp_noise(self, delta_t):
        if self.stdp_active:
            stdp_reinforces = delta_t > 0
            A = self.A_stdp(stdp_reinforces)
            stdp_decay = np.exp(-abs(delta_t/stdp_tau) )
            self.change_weight( A*stdp_decay)
    
    def change_weight(self, weight_change):
        if self.excitatory:
            self.weight = min(stdp_excitatory_weight_max, max(self.weight + weight_change, stdp_weight_min))
        else:
            self.weight = max(-stdp_inhibitory_weight_max, min(self.weight + weight_change, -stdp_weight_min))

     
    def A_stdp(self, stdp_reinforce):
        weight_modulus = abs(self.weight)
        if stdp_reinforce:
            if self.excitatory:    
                softCoeff = (stdp_excitatory_weight_max - weight_modulus)
                A = softCoeff*stdp_excitatory_increase_nu
                return A
            else:
                softCoeff = (stdp_inhibitory_weight_max - weight_modulus)
                A = softCoeff*stdp_excitatory_increase_nu
                return -A
        else:
            if self.excitatory:    
                return -weight_modulus*stdp_excitatory_decrease_nu
            else:
                return weight_modulus*stdp_inhibitory_decrease_nu
    
    def invalidate_synapse(self):
        self.presynaptic_neuron.outgoing_synapses.remove(self)
        self.postsynaptic_neuron.incomming_synapses.remove(self)
        self.active_synapse = False
        
    def print_synapse(self):
        print('From ' + str(self.presynaptic_neuron.id)
              + ' to ' + str(self.postsynaptic_neuron.id)
             + '  weight ' + str(self.weight)
             + '  delay '+ str(self.delay))


class Neuron_LIF:

    def __init__(self, neuron_id = 1,spike_time=-1, membrane_potential=resting_potential, 
                 outgoing_synapses_list=None, incomming_synapses_list=None, 
                 threshold_adaptation_active = True ):
        self.id = neuron_id
        self.last_spike = spike_time
        self.last_firing = non_existant_time
        self.membrane_potential = membrane_potential
        self.firing_threshold = firing_threshold
        
        if outgoing_synapses_list is None:
            self.outgoing_synapses = []
        if incomming_synapses_list is None:
            self.incomming_synapses = []
            
    def __del__(self):
        del self.outgoing_synapses[:]
        del self.incomming_synapses[:]
        

    def reset_activity(self):
        self.last_spike = -1
        self.last_firing = -1
        self.membrane_potential = resting_potential

    def receive_spike(self, spike):
        #calculation_str= ' Original potential: '+str(self.membrane_potential)+ ' Incomming: ' + str(spike.voltage)

        if spike.voltage == irresistible_spike:
            self.last_spike = spike.arrival_time
            return self.fire(spike.arrival_time +1 + jitter)
        
        if self.is_refractory_period(spike.arrival_time):
            #print 'Hit refractory period'
            return 0
        else:
            self.last_spike = spike.arrival_time
            self.stdp_on_receiving(spike.synapse, spike.arrival_time)
            pre_spike_potential = self.compute_membrane_potential(spike.arrival_time)
            self.membrane_potential = pre_spike_potential + spike.voltage
            
            #calculation_str += ' Final: ' + str(self.membrane_potential)
            #print calculation_str
            
        if self.membrane_potential >= self.firing_threshold:
            return self.fire(spike.arrival_time +1)
        else:
            return 0

    def fire(self, time):
        self.membrane_potential = resting_potential
        self.last_spike = time + refractory_period
        self.last_firing = time
        self.stdp_on_firing()
        return self.create_postynaptic_spikes(time)
        
    def is_refractory_period(self, time):
        return (time - self.last_firing) < refractory_period and self.last_firing != -1
    
    def stdp_on_firing(self):
        for syn in self.incomming_synapses:
            syn.stdp()
            
    def stdp_on_receiving(self, synapse, time):
        if self.last_firing != -1:
            synapse.stdp()
    
    def create_postynaptic_spikes(self, time):
        if self.outgoing_synapses:
            list_spikes = []
            for syn in self.outgoing_synapses:
                arrival_time = time + syn.delay
                spike = Spike(syn, arrival_time, syn.weight)
                list_spikes.append(spike)
            return list_spikes
        else:
            return 1
        
    def compute_membrane_potential(self, time):
        if self.last_spike == -1:
            return resting_potential
        else:
            delay_since_last_input = time - self.last_spike 
            potential_decay = np.exp( - decay_constant*delay_since_last_input)
            decayed_potential_difference = (self.membrane_potential - resting_potential)*potential_decay 
            current_potential = decayed_potential_difference + resting_potential
            return current_potential
        
    def print_neuron(self):
        string = ' Neuron Id: '+str(self.id)
        string += '  Incomming ('+ str(len(self.incomming_synapses))+ '): '
        for syn in self.incomming_synapses:
            string += ' '+str(syn.presynaptic_neuron.id)
        string += '  Outgoing ('+ str(len(self.outgoing_synapses))+ '): '
        for syn in self.outgoing_synapses:
            string += ' '+str(syn.postsynaptic_neuron.id)
        print(string)
