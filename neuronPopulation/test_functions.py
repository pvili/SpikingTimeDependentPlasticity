# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:57:20 2018

@author: aceituno
"""

import spiking_neural_networks as snn
import data_processing as dp
import random
import numpy.random as nprand
import copy

discard_first = 0 #discard the first 5 results


def test(repetitions, neural_network, input_freq = 1., weight_normalization = False):
    list_sim_results =[]    
    r = 0
    
    while r < repetitions:

        if random.random() > input_freq:
            neural_network.execute_noise()
        else:
            neural_network.init_stimulus()
            sim_result = neural_network.run()
            neural_network.reset_activity()
            if weight_normalization:
                neural_network.normalize_weights(sim_result)
            list_sim_results.append(sim_result)
            r = r + 1

    return list_sim_results


def test_basic(repetitions, neural_network, noise_per_input = 0.):
    list_sim_results =[]    
    cumulated_noise_exec = noise_per_input

    for r in range(repetitions + discard_first):
        neural_network.init_stimulus()
        sim_result = neural_network.run()
        while cumulated_noise_exec > 0:
            neural_network.execute_noise()
            cumulated_noise_exec += -1.
        neural_network.reset_activity()
        cumulated_noise_exec += noise_per_input
        
        if r >= discard_first:
            list_sim_results.append(sim_result)
        
    return list_sim_results

def test_noise(neural_network, repetitions, noise_list):
    list_results = []
    for noiseLevel in noise_list:
        neural_network_cpy = copy.deepcopy(neural_network)
        list_sim = test_basic(repetitions, neural_network_cpy, noiseLevel)
        list_results.append(list_sim[repetitions-1])
    return list_results

def test_spikeCount(inputRate, timeInterval, neurons, repetitions, inputProb):
    SNN = snn.LIF_Network(neurons, int(inputRate*timeInterval))
    SNN.initial_noise()
    sim_results = test(repetitions, SNN, inputProb)
    countSpikes = []
    for r in range(repetitions):
        countSpikes.append(len(sim_results[r]))
    return countSpikes

def test_input_level(inputRateList, timeInterval, neurons, inputProb):
    snn.neurons_per_population = neurons
    snn.delay_SP_max = timeInterval
    repetitions = 100
    initialSpikeCount = []
    finalSpikeCount = []
    
    for r in inputRateList:
        snn.synapses_per_neuron = r*timeInterval
        SNN = snn.LIF_Network(neurons, int(r*timeInterval))
        SNN.initial_noise()
        sim_results = test(repetitions, SNN, inputProb)
        initialSpikeCount.append(float(len(sim_results[0]))/neurons)
        finalSpikeCount.append(float(len(sim_results[repetitions - 1]))/neurons)
    
    return (initialSpikeCount, finalSpikeCount)




def test_signal_level(inputRate, timeInterval, neurons, repetitions, inputProb):
    SNN = snn.LIF_Network(neurons, int(inputRate*timeInterval))
    SNN.initial_noise()
    sim_results = test(repetitions, SNN, inputProb)
    signalLevelList = []
    for r in range(repetitions):
        (s_max,t_max) = dp.max_signal(sim_results[r],range(2,neurons))
        signalLevelList.append(s_max)
    return signalLevelList

def test_poissonian_input(repetitions, neural_network, timeInterval = [snn.delay_SP_min, snn.delay_SP_max]):
    list_sim_results =[]    

    for r in range(repetitions + discard_first):
        neural_network.init_stimulus()
        sim_result = neural_network.run()
        neural_network.rewire(timeInterval)
        neural_network.reset_activity()
        
        if r >= discard_first:
            list_sim_results.append(sim_result)
        
    return list_sim_results
#def test_multiple_network_single_input(instances, repetitions, number_of_layers, 
#                                       neurons_per_layer, synapses_per_neuron, fraction_negative):
#    list_sim = [] 
#    for i in range(instances):
#        rand_FF = ns.random_FeedForward(number_of_layers, neurons_per_layer, synapses_per_neuron, fraction_negative)
#        SNN = snn.LIF_Network(rand_FF)
#        sim_temp = []
#        sim_temp.extend(test_basic(repetitions, SNN, SNN.neurons[0:neurons_per_layer]))
#        list_sim.append(sim_temp)
#
#    median_per_layer = []
#    upQ_per_layer = []
#    lowQ_per_layer = []
#    
#    for l in range(number_of_layers-1):
#        median, upQ, lowQ = dp.medianQuartile_arrival_evolution(list_sim, l+1, neurons_per_layer)
#        median_per_layer.append(median)
#        upQ_per_layer.append(upQ)
#        lowQ_per_layer.append(lowQ)
#    return median_per_layer, upQ_per_layer, lowQ_per_layer
#
#def avg_delay_single_trial(neural_network, input_neurons):
#    neural_network.init_activity(input_neurons)
#    sim_result = neural_network.run()
#    neural_network.reset_activity()
#    arrival_times = dp.list_firing_times_per_layer(sim_result, ns.number_of_layers, ns.neurons_per_layer)
#    if len(arrival_times)> 0:
#        avg_arrival = sum(arrival_times)/len(arrival_times)
#    else:
#        avg_arrival = 0
#    return avg_arrival


#def test_two_inputs_multiple_prob(learning_steps, trials):
#    avg_arrival_times = []
#    input_probs = []
#    
#    test = 0
#    
#    while test < trials :
#        neural_network, input_list, input_freq = generate_two_inputs_setup()
#        
#        run_multiple_input_scale_weights(learning_steps, neural_network, input_list, input_freq)
#        neural_network.active_learning(False)
#        
#        sim_results_1 = test_single_input(1, neural_network, input_list[0])
#        sim_results_2 = test_single_input(1, neural_network, input_list[1])
#        
#        arrival_times_1 = dp.list_firing_times_per_layer(sim_results_1[0], ns.number_of_layers, ns.neurons_per_layer)
#        arrival_times_2 = dp.list_firing_times_per_layer(sim_results_2[0], ns.number_of_layers, ns.neurons_per_layer)
#        
#        if len(arrival_times_1)> 0 and len(arrival_times_2) > 0:
#            avg_arrival_1 = sum(arrival_times_1)/len(arrival_times_1)
#            avg_arrival_2 = sum(arrival_times_2)/len(arrival_times_2)
#            
#            avg_arrival_times.append(avg_arrival_1)
#            input_probs.append(input_freq[0])
#            avg_arrival_times.append(avg_arrival_2)
#            input_probs.append(input_freq[1])
#            
#            test += 1            
#            
#    return avg_arrival_times, input_probs
    
#def generate_two_inputs_setup():
#    rand_FF = ns.random_FeedForward(ns.number_of_layers, ns.neurons_per_layer, ns.synapses_per_neuron, ns.fraction_negative)
#    neural_network = snn.LIF_Network(rand_FF)  
#    input_neurons = list(neural_network.neurons[0:ns.neurons_per_layer])
#    random.shuffle(input_neurons)
#    input_1 = input_neurons[0:(ns.neurons_per_layer/2)]
#    input_2 = input_neurons[(ns.neurons_per_layer/2+1):ns.neurons_per_layer]
#    input_freq_1 = random.uniform(0.,1.)
#    input_freq_2 = 1 - input_freq_1
#    
#    return neural_network, [input_1, input_2], [input_freq_1, input_freq_2]
#    
#    
#    
#def test_inputs_final_speed_multiple_prob(number_inputs, neurons_per_input, learning_steps, trials):
#    avg_arrival_times = []
#    inputs_prob = []
#    for test in range(trials):
#        rand_FF = ns.random_FeedForward(ns.number_of_layers, ns.neurons_per_layer, ns.synapses_per_neuron, ns.fraction_negative)
#        neural_network = snn.LIF_Network(rand_FF)  
#        input_list = generate_inputs(neural_network, number_inputs, neurons_per_input, ns.neurons_per_layer)
#        input_freq = generate_frequencies(number_inputs)
#        
#        run_multiple_input_scale_weights(learning_steps, neural_network, input_list, input_freq)
#        neural_network.active_learning(False)
#        
#        
#        for i in range(number_inputs):
#            sim_result = test_single_input(1, neural_network, input_list[i])
#            arrival_times = dp.list_firing_times_per_layer(sim_result[0], ns.number_of_layers, ns.neurons_per_layer)
#            if len(arrival_times)> 0:
#                avg_arrival = sum(arrival_times)/len(arrival_times)
#                avg_arrival_times.append(avg_arrival)
#                inputs_prob.append(input_freq[i])
#            
#    return avg_arrival_times, inputs_prob
#                                                                    
#                                        
#
#def run_multiple_input_scale_weights(repetitons, neural_network, inputs_list, input_freq):
#    list_sim_results =[]
#    input_idx_list = range(len(inputs_list))
#    inputs_id = nprand.choice(input_idx_list, repetitons, p=input_freq)
#    number_neurons = len(neural_network.neurons)
#    
#    for r in range(repetitons):
#        input_idx = inputs_id[r]
#        input_neurons = inputs_list[input_idx]
#        neural_network.init_activity(input_neurons)
#        
#        sim_result = neural_network.run()
#        neural_network.reset_activity()
#        
#        list_sim_results.append(sim_result)
#        
#        if r % snn.nm.homeo_ratio_stdp_th == 0 and r > 0:
#            last_sim_results = list_sim_results[(r-snn.nm.homeo_ratio_stdp_th):r]
#            firing_rates = dp.get_firing_rates(last_sim_results, number_neurons)
#            neural_network.scale_weights(firing_rates)
#            
#    return list_sim_results, inputs_id
#  
#def run_multiple_inputs(repetitions, neural_network, inputs_list, input_freq):
#    list_sim_results =[]
#    input_idx_list = range(len(inputs_list))
#    inputs_id = nprand.choice(input_idx_list, repetitions, p=input_freq)
#    
#    for r in range(repetitions):
#        input_idx = inputs_id[r]
#        input_neurons = inputs_list[input_idx]
#        neural_network.init_activity(input_neurons)
#        
#        sim_result = neural_network.run()
#        neural_network.reset_activity()
#        
#        list_sim_results.append(sim_result)
#    return list_sim_results, inputs_id
#    
#def generate_inputs(neural_network, number_inputs, firing_count, number_neurons):
#    inputs = []
#    for k in range(number_inputs):
#        index_list = random.sample(range(number_neurons), firing_count)
#        initial_neurons = []
#        for idx in index_list:
#            initial_neurons.append(neural_network.neurons[idx])
#        inputs.append(list(initial_neurons))
#    return inputs
#    
#def generate_frequencies(count_inputs):
#    x =nprand.uniform(0.0,1.0,count_inputs)
#    freq = x/sum(x)
#    return freq
