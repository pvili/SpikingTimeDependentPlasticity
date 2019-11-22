#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 19:57:47 2018

@author: aceituno
"""

import numpy as np
#import pyspike as spk

def count_firing_neurons(sim_results, population):
    population_set = set(population)
    spiking_neurons = set()
    for tup in sim_results:
        neuron_id = tup[0]
        if neuron_id in population_set:
            spiking_neurons.add(neuron_id)
    return len(spiking_neurons)

def count_spikes_per_population(sim_results, population):
    population_set = set(population)
    spikes_count = 0
    for tup in sim_results:
        neuron_id = tup[0]
        if neuron_id in population_set:
            spikes_count += 1
    return spikes_count

def list_firing_times(sim_results, population_idx):
    neurons_idx = set(population_idx)
    firing_times = []
    for tup in sim_results:
        neuron_id = tup[0]
        if neuron_id in neurons_idx:
            firing_times.append(tup[1])
    return firing_times

def list_firing_times_per_neuron(sim_results, number_neurons):
    list_spike_times = [[] for n in range(number_neurons)]
    for tup in sim_results:
        neuron_id = tup[0]
        time = tup[1]
        (list_spike_times[neuron_id - 1]).append(time)

    return list_spike_times

def minimum_avgLatency(sim_results, population_idx):
    neurons_idx = set(population_idx)
    neurons_that_fired = set()
    firing_times = []
    for tup in sim_results:
        neuron_id = tup[0]
        if neuron_id in neurons_idx:
            if neuron_id not in neurons_that_fired:
                firing_times.append(tup[1])
                neurons_that_fired.add(neuron_id)
    return np.average(firing_times)

#def measure_synchrony(sim_results, population_idx, ival = [0,100]):
#    cumulatedSimilarity = 0
#    countPairs = 0
#    
#    numberNeurons = len(population_idx)
#    spikesList = list_firing_times_per_neuron(sim_results, 100)
#    
#    for idx1 in range(0,numberNeurons -1):
#        spike_train_1 = spk.SpikeTrain(spikesList[idx1], ival)
#        for idx2 in range(idx1,numberNeurons -1):
#            spike_train_2 = spk.SpikeTrain(spikesList[idx2], ival)
#            if spike_train_1 and spike_train_2:
#                cumulatedSimilarity += spk.spike_sync(spike_train_1, spike_train_2, interval=ival)
#                countPairs += 1
#
#    return cumulatedSimilarity/countPairs
    
def spike_timing_evolution(list_sim_results, neurons_idxs):
    list_avg_latency = []
    list_std_latency = []
    
    for sim_results in list_sim_results:
        arrival_times = list_firing_times(sim_results, neurons_idxs)
        mean_latency = np.mean(arrival_times)
        std_latency = np.std(arrival_times)
        list_avg_latency.append(mean_latency)
        list_std_latency.append(std_latency)
        
    return list_avg_latency, list_std_latency
    

def spike_timing_evolution_quartiles(list_sim_results, neurons_idxs):
    list_med_latency = []
    list_low_latency = []
    list_up_latency = []
    
    for sim_results in list_sim_results:
        arrival_times = list_firing_times(sim_results, neurons_idxs)
        upQ_latency, med_latency, lowQ_latency = np.percentile(arrival_times, [75, 50, 25])
        list_med_latency.append(med_latency)
        list_up_latency.append(upQ_latency)
        list_low_latency.append(lowQ_latency)
        
    return list_med_latency, list_up_latency, list_low_latency

def spike_timing_evolution_percentiles(list_sim_results, neurons_idxs, percentages):
    list_perc_evolution = [[] for i in range(len(percentages))]
    
    
    for sim_results in list_sim_results:
        arrival_times = list_firing_times(sim_results, neurons_idxs)
        idx_perc = 0
        for perc in percentages:
            firing_percentile = np.percentile(arrival_times, perc)
            list_perc_evolution[idx_perc].append(firing_percentile)
            idx_perc += 1
        
    return list_perc_evolution