# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:33:58 2018

@author: aceituno
"""
#import network_structures as ns

import numpy as np
import spiking_neural_networks as snn
import random
from operator import itemgetter
import csv

countISI_merge = 1
data_dir = "./data/"

def list_neurons_in_population(list_neurons, population):
    set_common = set(list_neurons).intersection(population)
    return list(set_common)

def max_spikes_in_window(spike_list, population, windowLength):
    firing_times = list_firing_times(spike_list, population)
    
    pos_times = [(t,1) for t in firing_times]
    neg_times = [(t + windowLength,-1) for t in firing_times]
    
    signed_times = pos_times + neg_times
    signed_times.sort()
    
    S = 0
    S_max = 0
    t_max = 0
    
    for sig_t in signed_times:
        S = S + sig_t[1]
        if S_max < S:
            S_max = S
            t_max = sig_t[0]
    
    return (S_max, t_max)

def count_spikes_in_time_window(spike_list, time_window):
    s_idx = 0
    while (spike_list[s_idx])[1] < min(time_window):
        s_idx += 1
    
    spike_count = 1
    while (spike_list[s_idx])[1] < max(time_window):
        s_idx += 1
        spike_count += 1
        
    return spike_count
    

def list_spikes_per_population(sim_results, population):
    population_set = set(population)
    spikes = []
    for tup in sim_results:
        neuron_id = tup[0]
        if neuron_id in population_set:
            spikes.append(tup)
    return spikes

def list_firing_times(sim_results, population):
    neurons_idx = set(population)
    firing_times = []
    for tup in sim_results:
        neuron_id = tup[0]
        if neuron_id in neurons_idx:
            firing_times.append(tup[1])
    return firing_times

def firing_times_per_neuron(sim_results, population):
    firing_times = [[] for n in population]
    max_neuron = max(population)
    min_neuron = min(population)
    for tup in sim_results: # First zpike is the zero
        if tup[0] < max_neuron and tup[0]> min_neuron:
            firing_times[tup[0]].append(tup[1])
    return firing_times
   
def write_simulation_results(sim_results, filename, directory = data_dir):
    filename = directory+filename
    with open(filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(sim_results)

    
def read_simulation_results(filename, directory = data_dir):
    filename = directory + filename
    with open(filename, "r") as f:
        reader = csv.reader(f)
        str_sim_results = list(reader)
    from ast import literal_eval
    sim_results = []
    for list_str_sim in str_sim_results: 
        sim = []
        for str_spike in list_str_sim:
            sim.append(literal_eval(str_spike))
        sim_results.append(sim)  
    
    return sim_results

def list_neurons_spiked(sim_results, population):
    neurons = set(population)

    for tup in sim_results:
        neuron_idx = tup[0]
        if neuron_idx in neurons:
            neurons.add(neuron_idx) 
    
    return list(neurons)   

def interSpikeIntervals(sim_results, population, max_time = 200): #remove neurons firing after spike train
    L = len(population)+1
    spikesPerNeuron = [0 for l in range(L)]
    lastSpike = [0. for l in range(L)]
    ordered_ISI = []
    
    for tup in sim_results:
        neuron_idx = tup[0] - 1 #neuron zero does not count!
        if tup[0] != 0:            
            currentSpikeCount = spikesPerNeuron[neuron_idx]
            if (currentSpikeCount + 1) > len(ordered_ISI):
                ordered_ISI.append([])
            spikesPerNeuron[neuron_idx] = currentSpikeCount + 1
            spike_time = tup[1]
            ISI = spike_time - lastSpike[neuron_idx]
            ordered_ISI[currentSpikeCount].append(ISI)
            lastSpike[neuron_idx] = spike_time
            
    sorted_ISI = []
    for o_ISI in ordered_ISI:
        clean_ISI = [isi for isi in o_ISI if isi < 200]
        sorted_ISI.append(sorted(clean_ISI))
    
    return sorted_ISI
        
def mergeISI(spike_list, count_ISI = countISI_merge, neurons = 1000, t_ref = 2.):
    ordered_ISI = interSpikeIntervals(spike_list, range(1,neurons))
    combinedPreLearnedISI = []
    for n in range(1,count_ISI):
        correctedISI = [(o - t_ref) for o in ordered_ISI[n]]
        combinedPreLearnedISI = combinedPreLearnedISI + correctedISI
        
    combinedPreLearnedISI.sort()
    
    return combinedPreLearnedISI
  
    

def write_spike_count_table(initial_table, final_table, filename, time_intervals_count, rateVec, inputProbVec):
    
    table_latex = [[] for r in range(len(initial_table)+1)]
    
    table_latex[0].append('IR')
    table_latex[0].append('SP')
    
    for time_interval in time_intervals_count:
        header_str = str(time_interval) + ' ms'
        table_latex[0].append(header_str)
    
    for row in range(len(initial_table)):
        inpRate = rateVec[row/len(inputProbVec)]
        stimProb = inputProbVec[row % len(inputProbVec)]
        table_latex[row].append(str(inpRate))
        table_latex[row].append(str(stimProb))
        
        for col in range(len(initial_table[0])):
            #cellStr = '$'+str(initial_table[row][col])+ ' \\rightarrow ' + str(final_table[row][col]) + '$'
            cellVal = float(initial_table[row][col] - final_table[row][col])/float(final_table[row][col])*100
            cellStr = '$'+str(round(cellVal,1))+' \%$'
            table_latex[row].append(str(cellStr))
            
            
    with open(filename,'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(table_latex)


def extract_samples(list_spikes, neurons):
    set_first = set()#sets.set()
    set_second = set()#sets.set()
    list_first = [0 for n in range(neurons)]
    list_ISI = [0 for n in range(neurons)]
    
    for spike in list_spikes:
        neuron = spike[0] - 1
        time = spike[1]
        if neuron != 0:
            if neuron not in set_first:
                list_first[neuron] = time
                set_first.add(neuron)
            else:
                if neuron not in set_second:
                    list_ISI[neuron] = time - list_first[neuron]
                    set_second.add(neuron)
                
#    if len(set_first) < neurons or len(set_second) < neurons:
#        print('Out of '+str(neurons)+', '+str(len(set_first))+' fired the first spike and '+str(len(set_second))+' the second')
    
    clean_first = [e for e in list_first if e!= 0]
    clean_isi = [e for e in list_ISI if e!= 0]
    
    clean_first = [e for e in clean_first if e < 100]
    clean_isi = [e for e in clean_isi if e < 100]
    
    return [clean_first, clean_isi]


# FOR VERY LONG SPIKE TRAINS WE DO NOT SIMULATE THE WHOLE TRAIN, JUST SAMPLE ISI!
    
def probabilistic_spike_sampling_1neuron(samples_first, samples_ISI, time_max):
    spikes_time = []
    t = random.choice(samples_first)
    if t < time_max:
        spikes_time.append(t)
    while t < time_max:
        t = t + random.choice(samples_ISI)
        spikes_time.append(t)
        
    return spikes_time

def probabilistic_spike_sampling_population(samples_first, samples_ISI, time_max, neurons):
    spikes_time = []
    for n in range(neurons):
        times = probabilistic_spike_sampling_1neuron(samples_first, samples_ISI, time_max)
        spikes_time.extend(times)
    sorted_spikes_time = sorted(spikes_time)
    return sorted_spikes_time

