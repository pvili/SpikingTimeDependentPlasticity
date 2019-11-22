#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:51:19 2018

@author: aceituno
"""
import numpy as np
import singleNeuron as sN
import math
import learning_singleNeuron as learn

def test_Dt(time_window, nI, nE, max_E, max_I, learn_steps, num_samples = 1000,
            noise_level = [], initial_spike = False):
    initialy_one_postsyn = False
    simulation_steps = int(time_window/sN.dt + 1)
    
    deltaT_list = []
    spike_count_after_training = []
    t_final = 0
    countDecrease = 0.
    countIncrease = 0.
    latencyIncrease = 0.
    latencyDecrease = 0.
    
    for s in range(num_samples):
        while not initialy_one_postsyn:
            presynaptic_spikes = sN.generate_random_spike_train(nE, nI, max_E, max_I, time_window)
            V, firing_list = sN.run_simulations(presynaptic_spikes, simulation_steps)
            
            if len(firing_list) == 1:
                initialy_one_postsyn = True
                t_initial = firing_list[0]

        V, firing_list_lists = sN.run_single_neuron(presynaptic_spikes, learn_steps, simulation_steps,
                                                    learn.stdp, noise_level, initial_spike)
        firing_list = firing_list_lists[len(firing_list_lists)-1]
        spike_count_after_training.append(float(len(firing_list)))
        if len(firing_list) == 1:
            t_final = firing_list[0]
            if t_final < t_initial:
                latencyDecrease += 1.
        else:
            if t_final > t_initial:
                latencyIncrease += 1.
            else:
                if len(firing_list) == 0:
                    countDecrease += 1.
                else:
                    countIncrease += 1.
        
        deltaT_list.append(t_final - t_initial)
        initialy_one_postsyn = False
        
    print(" Simulations many spike trains: Avg Dt = "+str(sum(deltaT_list)/len(deltaT_list))
        + "  Frac latency Decrease: " + str(latencyDecrease/num_samples*100) + "%  Frac Lat Increase: "
        + str(latencyIncrease/num_samples*100) + "%  Count increase: "+ str(countIncrease/num_samples*100)
        + "%  Count decrease: "+ str(countDecrease/num_samples*100) + "%  Avg count change: " + 
        str(sum(spike_count_after_training)/num_samples))
        
        
    return deltaT_list

#def test_two_presynaptic_fixed_postsynaptic(t_1, t_2, weight_range, pts_per_spike, repetitions):
#    w_min = min(weight_range)
#    w_max = max(weight_range)
#    times = [t_1, t_2]
#    
#    list_weight_evolutions = []
#    
#    for w1 in np.linspace(w_min, w_max, pts_per_spike):
#        for w2 in np.linspace(w_min, w_max, pts_per_spike):
#            weights = [w1, w2]
#            w_evol = two_spikes_fixed_postsynaptic_weight_evolution(weights, times, repetitions)
#            list_weight_evolutions.append(w_evol)
#    
#    return list_weight_evolutions
#
#def two_spikes_fixed_postsynaptic_weight_evolution(weights, times, repetitions, simulation_steps = [], stdp_f = sN.learn.stdp):
#    if not simulation_steps:
#        simulation_steps = int(math.ceil( max(times)/sN.dt + 2))
#    
#    spike_list = create_2spikes(weights, times)
#    weight_evolution_list = [extract_weights(spike_list)]
#    spike_list.insert(0, [0, sN.irresistible_spike])
#    
#    for r in range(repetitions):
#        V_list, firing_list = sN.run_simulations(spike_list, simulation_steps)
#        stdp_f(spike_list, firing_list)
#        w_list = extract_weights([spike_list[1],spike_list[2]])
#        weight_evolution_list.append(w_list)
#    return weight_evolution_list


def test_two_spikes_evolution(time_distance, weight_range, pts_per_spike, repetitions):
    w_min = min(weight_range)
    w_max = max(weight_range)
    times = [0, time_distance]
    
    list_weight_evolutions = []
    
    for w1 in np.linspace(w_min, w_max, pts_per_spike):
        for w2 in np.linspace(w_min, w_max, pts_per_spike):
            weights = [w1, w2]
            w_evol = two_spikes_weight_evolution(weights, times, repetitions)
            list_weight_evolutions.append(w_evol)
    
    return list_weight_evolutions
 
    

def two_spikes_weight_evolution(weights, times, repetitions, simulation_steps = [], stdp_f = sN.learn.stdp):
    if not simulation_steps:
        simulation_steps = int(math.ceil( max(times)/sN.dt + 2))
    
    spike_list = create_2spikes(weights, times)
    weight_evolution_list = [extract_weights(spike_list)]
    
    for r in range(repetitions):
        V_list, firing_list = sN.run_simulations(spike_list, simulation_steps)
        stdp_f(spike_list, firing_list)
        w_list = extract_weights(spike_list)
        weight_evolution_list.append(w_list)
    return weight_evolution_list

def create_2spikes(weights, times):
    spike_list = [[times[0], weights[0]], [times[1], weights[1]]]
    return spike_list

def extract_weights(spike_list):
    weight_list = []
    for s in spike_list:
        w = s[1]
        weight_list.append(w)
    return weight_list


def line_2ndneuron_fires(w_max, delta_t, tau_mem, v_th = 20, linePts = 100):
    w1_val = np.ndarray.tolist(np.linspace(0, w_max, linePts))
    w2_val = []
    decay =  math.exp(-delta_t/tau_mem)
    for w1 in w1_val:
        if w1 < v_th:
            w2_val.append(v_th - w1*decay)
        else:
            w2_val.append(v_th)
    return w1_val, w2_val