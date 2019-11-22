#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 19:17:03 2018

@author: aceituno
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:57:20 2018

@author: aceituno
"""

import spiking_neural_networks as snn
import data_processing as dp
import numpy as np

#import pyspike as spk


discard_first = 0 


    
def test_basic(repetitions, neural_network, noise_per_input = 0., noise_time = [-10, 10]):
    list_sim_results =[]    
    cumulated_noise_exec = noise_per_input

    for r in range(repetitions + discard_first):
        neural_network.init_stimulus()
        sim_result = neural_network.run()
        while cumulated_noise_exec > 0:
            neural_network.execute_noise(noise_time)
            cumulated_noise_exec += -1.
        neural_network.reset_activity()
        cumulated_noise_exec += noise_per_input
        
        if r >= discard_first:
            list_sim_results.append(sim_result)
        
    return list_sim_results



def test_spikes_per_noise(repetitions, population_trials, noise_list=range(0,10), noise_time = [-10,10], init_noise=100):
#    list_avgLatencies = []
#    list_stdLatencies = []
    list_noise = []
    #list_synchrony = []
    
    for noise_Lvl in noise_list:
        list_trial =[]
        for trial in range(population_trials):
            SNN = snn.LIF_Network()
            population = range(1,SNN.count_neurons)
            
            run_noise(SNN, init_noise, noise_time)
            
            sim_results = test_basic(repetitions, SNN, noise_Lvl, noise_time)
            
            list_spike_times = dp.list_firing_times(sim_results[repetitions-1], population)
            list_trial.append(list_spike_times)
            
        list_noise.append(list_trial)
        
    return list_noise


#def test_noiseLevel(repetitions, population_trials, noise_list=range(0,10), noise_time = [-10,10], init_noise=100):
##    list_avgLatencies = []
##    list_stdLatencies = []
#    list_avgLatencies = []
#    list_countSpikes =[]
#    list_minLatencies = []
#    #list_synchrony = []
#    
#    for noise_Lvl in noise_list:
#        average_times = []
#        counts = []
#        minLat = []
#        for trial in range(population_trials):
#            SNN = snn.LIF_Network()
#            population = range(1,SNN.count_neurons)
#            
#            run_noise(SNN, init_noise, noise_time)
#            
#            sim_results = test_basic(repetitions, SNN, noise_Lvl, noise_time)
#            
#            list_spike_times = dp.list_firing_times(sim_results[repetitions-1], population)
#            avgLatency = np.average(list_spike_times)
#            average_times.append(avgLatency)
#            
#            countSpikes = dp.count_spikes_per_population(sim_results[repetitions-1], population)
#            counts.append(countSpikes)
#            
#            average_first_firing = dp.minimum_avgLatency(sim_results[repetitions-1], population)
#            minLat.append(average_first_firing)
##            sync = dp.measure_synchrony(sim_results[repetitions-1], population)
##            syncrony.append(sync)
#            
#        list_avgLatencies.append(average_times)
#        list_countSpikes.append(counts)
#        list_minLatencies.append(minLat)
##        list_synchrony.append(syncrony)
#        
#    return (list_avgLatencies, list_countSpikes )#,list_synchrony)

def run_noise(neural_network, noise_repetitions, noise_time = [-10., 10.]):
    for r in range(0,noise_repetitions):
        neural_network.execute_noise(noise_time)
        
