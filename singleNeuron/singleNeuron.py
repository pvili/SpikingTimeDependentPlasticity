# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 19:51:18 2018

@author: user
"""
import math
import learning_singleNeuron as learn
import numpy as np
import random as rd

dt = 0.1

tau_membrane = 10.
refractory_period = 7.*dt #11.*dt
resting_potential = -70.0
firing_threshold = -50.0
spiking_voltage = -40
weight_noise = []
initial_spike = False


irresistible_spike = learn.irresistible_spike

def run_simulations(presynaptic_spikes, simulation_steps, spiking_allowed = True):
    V_list = []
    firing_list = []

    V = resting_potential
    spike_idx = 0
    t=0.
    t_ref = 0  
    time_out_simulation = simulation_steps*dt + refractory_period + 1.
    current_next_spike, t_next_spike = get_spike(presynaptic_spikes, spike_idx, time_out_simulation)
    
    for s in range(simulation_steps):    
        if t < t_ref:
            V = resting_potential
            while t_next_spike < t_ref:
                spike_idx += 1
                current_next_spike, t_next_spike = get_spike(presynaptic_spikes, spike_idx, time_out_simulation)
        elif (t -0.5*dt) <= t_next_spike and (t+0.5*dt) > t_next_spike:
            while (t+0.5*dt) > t_next_spike:   
                V = V + current_next_spike
                spike_idx += 1                
                current_next_spike, t_next_spike = get_spike(presynaptic_spikes, spike_idx, time_out_simulation)
                
            if V >= firing_threshold and spiking_allowed:
                
                firing_list.append(t + dt*0.5) #otherwise we get precision problems  
                V = spiking_voltage
                t_ref = t + refractory_period
            
        else:
            V = V - (V - resting_potential)/tau_membrane*dt
        
        t += dt
        V_list.append(V)
    return V_list, firing_list

def get_spike(presyn_spikes, spike_idx, time_out_simulation):
    if spike_idx < len(presyn_spikes):
        t_next = presyn_spikes[spike_idx][0]
        V_spike = presyn_spikes[spike_idx][1]
    else:
        t_next = time_out_simulation
        V_spike = 0
        
    return V_spike, t_next

def run_single_neuron(input_spikes, repetitions, simulation_steps,
                      stdp_function= learn.stdp, noise_level = weight_noise, initial_spike = initial_spike):#noise_time_interval = []):
    firing_times = []
    membrane_potential_list = []
    presynaptic_spikes = input_spikes[:]

    for r in range(repetitions):   
        V_list, firing_list = run_simulations(presynaptic_spikes, simulation_steps)  
        if initial_spike:
            firing_list.insert(0,0)
        stdp_function(presynaptic_spikes, firing_list)
        if initial_spike:
            firing_list.pop(0)
#        if noise_time_interval:
#            learn.add_noise(presynaptic_spikes, noise_time_interval)
        if noise_level:
            learn.add_gaussian(presynaptic_spikes, noise_level)
        membrane_potential_list.append(V_list)
        firing_times.append(firing_list)

    return membrane_potential_list, firing_times
    

def generate_random_spike_train(nE, nI, max_E, max_I, t_max):
    times = list(np.random.random(nE + nI)*t_max)
    times.sort()
    
    weights_E = list(np.random.random(nE)*max_E)
    weights_I = list(np.random.random(nI)*max_I*(-1))
    weights = weights_E + weights_I
    rd.shuffle(weights)
    
    spikes = [[t,w] for t, w in zip(times, weights)]
    
    return spikes
    
    