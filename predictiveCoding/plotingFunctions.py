#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:27:45 2018

@author: aceituno
"""
import data_processing as dp
import matplotlib.pyplot as plt
import numpy as np


path_fig = './'

showPlots = True
savePlots = True


def plot_spiking_rate_with_learning(spikesInitial, spikesFinal, neurons, timePlot=50, bins=25,  saveFigName = []):
    bins = np.linspace(0, timePlot, bins)
    times = dp.list_firing_times(spikesInitial, range(2,neurons))
    plt.hist(times, bins, alpha=0.5, label='Before STDP')
    
    times = dp.list_firing_times(spikesFinal, range(2,neurons))
    plt.hist(times, bins, alpha=0.5, label='After STDP')
    plt.legend()
    
    if not showPlots:
        print('Print should work!')
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    if showPlots:
        plt.show()
    
    
def rasterPlotPop(list_sim, populations_neuron_idxs, colorList = ['b', 'r', 'g', 'k'],  saveFigName = []):
    plt.ioff()
    for pop_idx in range(0,len(populations_neuron_idxs)):
        low_idx = min(populations_neuron_idxs[pop_idx])
        high_idx = max(populations_neuron_idxs[pop_idx])
        repetitions = []
        times = []
        for repet in range(0,len(list_sim)):
            firing_times_per_repet = dp.list_firing_times(list_sim[repet], range(low_idx, high_idx))
            times.extend(firing_times_per_repet)
            repetitions.extend([repet]*len(firing_times_per_repet))
        plt.scatter(repetitions, times, color = colorList[pop_idx],marker='.')
            
    plt.title("Evolution of Median Latency")
    plt.xlabel("Stimulus repetition")
    plt.ylabel("Latency")
    plt.legend(loc = 'upper right')
    if savePlots:
        print('Print should work!')
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    if showPlots:
        plt.show()
        
        
        


#def plot_latencyVsNoise(list_times, noise_Lvls, saveFigName = []):
#    avg_avgLatency = []
#    std_avgLatency = []
#    for times in list_times:
#        avg_avgLatency.append(np.average(times))
#        std_avgLatency.append(np.std(times))
#    plt.errorbar(noise_Lvls,avg_avgLatency, yerr = std_avgLatency)
#    plt.plot(noise_Lvls,avg_avgLatency, 'bo', markersize=12)
#    
#    plt.title("Average Latency vs Input Frequency", fontsize=18)
#    plt.xlabel("Random Spikes Between Repetitions", fontsize=16)
#    plt.ylabel("Average Latency", fontsize=16)
#    if saveFigName:
#        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
#        
#    if showPlots:
#        plt.show()
#
#def plot_SpikeCountVsNoise(list_spikeCount, noise_Lvls, saveFigName = []):
#    avg_Count = []
#    std_Count = []
#    for counts in list_spikeCount:
#        avg_Count.append(np.average(counts))
#        std_Count.append(np.std(counts))
#    plt.errorbar(noise_Lvls,avg_Count, yerr = std_Count)
#    plt.plot(noise_Lvls,avg_Count, 'bo', markersize=12)
#    plt.title("Spike Count vs Input Frequency", fontsize=20)
#    plt.xlabel("Random Spikes Between Repetitions", fontsize=16)
#    plt.ylabel("Number of Spikes", fontsize=16)
#    if saveFigName:
#        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
#    if showPlots:
#        plt.show()

def plot_firing_times_population(list_sim_results, pop_idx,  saveFigName = []):
    repetitions = range(len(list_sim_results)) 
    
    
    neuron_idxs = range(pop_idx[0], pop_idx[1])
    avg_latency, sdt_latency = dp.spike_timing_evolution(list_sim_results, neuron_idxs)
    med_latency, upQ_latency, lowQ_latency = dp.spike_timing_evolution_quartiles(list_sim_results, neuron_idxs)
    
    
    plt.plot(repetitions, S2_to_Pop_max*np.ones(len(repetitions)), 'k', label = 'Stimulus-population latency')
    plt.plot(repetitions, S2_to_Pop_min*np.ones(len(repetitions)), 'k')
    
    plt.plot(repetitions, med_latency, label = 'Median Latency')
    plt.fill_between(repetitions, lowQ_latency, upQ_latency, alpha=0.5, label = 'Quartiles')
    plt.title("Evolution of Median Latency")
    plt.xlabel("Stimulus repetition")
    plt.ylabel("Latency")
    plt.legend(loc = 'upper right')
    if saveFigName:
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    if showPlots:
        plt.show()
        

def plot_firing_times_population_avg(list_sim_results, pop_idx,  saveFigName = []):
    repetitions = range(len(list_sim_results)) 
    
    
    neuron_idxs = range(pop_idx[0], pop_idx[1])
    avg_latency, sdt_latency = dp.spike_timing_evolution(list_sim_results, neuron_idxs)
    lowBar = [avg_latency[r] - sdt_latency[r] for r in range(0,len(avg_latency))]
    highBar = [avg_latency[r] + sdt_latency[r] for r in range(0,len(avg_latency))]
    
    plt.plot(repetitions, S2_to_Pop_max*np.ones(len(repetitions)), 'k', label = 'Stimulus-population latency')
    plt.plot(repetitions, S2_to_Pop_min*np.ones(len(repetitions)), 'k')
    
    plt.plot(repetitions, avg_latency, label = 'Median Latency')
    plt.fill_between(repetitions, highBar, lowBar, alpha=0.5, label = 'Quartiles')
    plt.title("Evolution of Median Latency")
    plt.xlabel("Stimulus repetition")
    plt.ylabel("Latency")
    plt.legend(loc = 'upper right')
    if saveFigName:
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    if showPlots:
        plt.show()
        

def plot_firing_times_percentiles(list_sim_results, population_idxs, percentiles, saveFigName = []):
    repetitions = range(len(list_sim_results)) 
    
    neuron_idxs = range(population_idxs[0], population_idxs[1])
    list_perc = dp.spike_timing_evolution_percentiles(list_sim_results, neuron_idxs, percentiles)
    
    for perc_idx in range(len(list_perc) - 1):
        label = str(percentiles[perc_idx]) + '% - ' + str(percentiles[perc_idx + 1])  +'% of neurons'
        plt.fill_between(repetitions, list_perc[perc_idx], list_perc[perc_idx + 1], label = label)
    plt.plot(repetitions, S2_to_Pop_max*np.ones(len(repetitions)), 'k', label = 'Stimulus time window')
    plt.plot(repetitions, S2_to_Pop_min*np.ones(len(repetitions)), 'k')
    plt.title("Evolution Average Latency")
    plt.xlabel("Stimulus repetition")
    plt.ylabel("Latency")
    plt.legend(loc = 'upper right')
    if saveFigName:
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    if showPlots:
        plt.show()
        

def plot_firing_times_line_percentiles(list_sim_results, population_idxs, percentiles, saveFigName = []):
    repetitions = range(len(list_sim_results)) 
    
    neuron_idxs = range(population_idxs[0], population_idxs[1])
    list_perc = dp.spike_timing_evolution_percentiles(list_sim_results, neuron_idxs, percentiles)
    
    for perc_idx in range(len(list_perc)):
        label = str(percentiles[perc_idx]) + 'th fastest neuron'
        plt.plot(repetitions, list_perc[perc_idx], label=label)
    plt.fill_between(repetitions, S2_to_Pop_max*np.ones(len(repetitions)), S2_to_Pop_min*np.ones(len(repetitions)), facecolor='black',alpha=0.2, label = 'Stimulus time window')
    plt.title("Evolution of Neuron Latency", fontsize=18)
    plt.xlabel("Stimulus repetition", fontsize=16 )
    plt.ylabel("Latency", fontsize=16)
    plt.legend(loc = 'upper right', fontsize = 12)
    if saveFigName:
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    if showPlots:
        plt.show()
        
