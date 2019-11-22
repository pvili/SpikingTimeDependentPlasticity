# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 09:36:24 2018

@author: user
"""
import math
import numpy as np

tau_stdp = 10.
stdp_nu_E = 0.02
stdp_weight_min = 0.0 
stdp_excitatory_weight_max = 10.0 
ratio_ie = 4
stdp_learn_ratio = 1.5


irresistible_spike = 1000


def stdp_E_unbound(input_spikes, firing_times):
    for spike in input_spikes:
        spike_weight = spike[1]
        #for firing_list in firing_times:
        for t_f in firing_times:
            #spike_number = 0
            #for t_f in firing_list:
            spike_time = spike[0]
            delta_t = t_f - spike_time
            time_factor = math.exp(-abs(delta_t)/tau_stdp)
            weight_factor = A_fixed_stdp(spike_weight, delta_t)
            learn_rate = learning_rate(spike_weight, stdp_nu_E, stdp_nu_E*ratio_ie)
            #print(learn_rate)
            weight_change = learn_rate*weight_factor*time_factor                
            spike[1] = spike[1] + weight_change
            #spike_number += 1    

def A_soft_stdp(spike_weight, delta_t):
    if spike_weight == 0:
        return 0
    else:
        excitatory = spike_weight > 0
        stdp_reinforce = delta_t >= 0
        if stdp_reinforce:
            if excitatory:    
                #the max is there to avoid problems with numerical precition
                return max(0.,stdp_excitatory_weight_max - spike_weight)
            else:
                return -stdp_excitatory_weight_max*ratio_ie - abs(spike_weight)
                        #-max(0.,stdp_excitatory_weight_max*ratio_ie - abs(spike_weight))
        else:
            if excitatory:    
                return -max(0.,spike_weight) * stdp_learn_ratio
            else:
                return -min(0.,spike_weight) * stdp_learn_ratio

def A_fixed_stdp(spike_weight, delta_t):
    stdp_reinforce = delta_t >= 0
    if stdp_reinforce:
        return 1.
    else:
        return -1.
         
def stdp(input_spikes, firing_times, A_stdp=A_soft_stdp, 
         learning_rate_E = stdp_nu_E, learning_rate_I = stdp_nu_E*ratio_ie):
    for spike in input_spikes:
        for t_f in firing_times:
            single_stdp(spike, t_f, A_stdp, learning_rate_E, learning_rate_I )


def single_stdp(spike, firing_time, A_stdp=A_soft_stdp, 
         learning_rate_E = stdp_nu_E, learning_rate_I = stdp_nu_E*ratio_ie):

    spike_time = spike[0]
    spike_weight = spike[1]
    
    delta_t = firing_time - spike_time 
    time_factor = math.exp(-abs(delta_t)/tau_stdp)
    weight_factor = A_stdp(spike_weight, delta_t)
    learn_rate = learning_rate(spike_weight, learning_rate_E, learning_rate_I)
    if learn_rate != 0:
        weight_change = learn_rate*weight_factor*time_factor
        prospective_weight = spike[1] + weight_change
        if spike_weight >= 0:
            spike[1] = max(0, min(stdp_excitatory_weight_max, prospective_weight))
        else:
            spike[1] = min(0, max(-stdp_excitatory_weight_max*ratio_ie, prospective_weight))
                    
            
def learning_rate(spike_weight, learning_rate_E, learning_rate_I):
    if spike_weight >= 0:
        return learning_rate_E
    else:
        return learning_rate_I
    
def add_gaussian(spike_list, noise_level):
    for spike in spike_list:
        if spike[1] != irresistible_spike:
            noise = np.random.normal(0, noise_level)
            prospective_weight = spike[1] + noise
        if spike[1] >= 0:
            spike[1] = max(0, min(stdp_excitatory_weight_max, prospective_weight))
        else:
            spike[1] = min(0, max(-stdp_excitatory_weight_max*ratio_ie, prospective_weight))

def add_noise(spike_list, time_interval = [-10., 10]):
    for spike in spike_list:
        if spike[1] != irresistible_spike:
            firing_time = spike[0] + np.random.uniform(time_interval[0], time_interval[1])
            single_stdp(spike, firing_time)