#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:27:45 2018

@author: aceituno
"""
import data_processing as dp
import matplotlib.pyplot as plt
import numpy as np


path_fig = '../figures'

showPlots = True
savePlots = True
timePlotRate = 50#200
timePlotCISI = 100

def plot_spiking_rate_with_learning(spikesInitial, spikesFinal, neurons, timePlot=timePlotRate, bins=40,  title =[], saveFigName = []):
    bins = np.linspace(0, timePlot, bins)
    times = dp.list_firing_times(spikesInitial, range(2,neurons))
    plt.hist(times, bins, alpha=0.5, label='Before STDP')
    
    times = dp.list_firing_times(spikesFinal, range(2,neurons))
    plt.hist(times, bins, alpha=0.5, label='After STDP')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('s(t)')
    plt.title(title)
    if savePlots:
        #print('Print should work!')
        plt.savefig(path_fig +'/'+ saveFigName +'.pdf', format = 'pdf')
    if showPlots:
        plt.show()

def plot_max_signal(sim_results, neurons = 1000,timeWindow = [10], saveFigName = []):
    repetitions = len(sim_results)
    for tw in timeWindow:
        maxSignal = []
        for r in range(repetitions):
            (s,t)=dp.max_spikes_in_window(sim_results[r], range(2,neurons),tw)
            maxSignal.append(s/tw)
        label = 'L = '+str(tw)
        plt.plot(maxSignal, label=label)
    plt.xlabel('Input Repetition')
    plt.ylabel("Maximum Signal")
    plt.legend()
    if savePlots:
        #print('Print should work!')
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    if showPlots:
        plt.show()


def plot_max_time(sim_results,  neurons = 1000, timeWindow = 10, saveFigName = []):
    repetitions = len(sim_results)
    timesMax = []
    for r in range(repetitions):
        (s,t)=dp.max_spikes_in_window(sim_results[r], range(2,neurons),timeWindow)
        timesMax.append(t)
    plt.plot(timesMax)
    plt.xlabel('Input Repetition')
    plt.ylabel("Max signal time")
    if savePlots:
        #print('Print should work!')
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    if showPlots:
        plt.show()

        
def plot_spikeCount(sim_results, neurons=1000, saveFigName = []):
    repetitions = len(sim_results)
    countSpikes = []
    for r in range(repetitions):
        countSpikes.append(len(sim_results[r]))
    plt.plot(countSpikes)
    plt.xlabel('Input Repetition')
    plt.ylabel('Spike Count')
    if savePlots:
        #print('Print should work!')
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    if showPlots:
        plt.show()
   
    
#def plot_spikeCountPerWindowLength(sim_results, windowLengths, neurons=1000, saveFigName = []):
#    for l in windowLengths:
        
    
def plot_cumulative_ISI(spike_list, neurons = 1000, count_ISI = 5, refISI = [], t_ref = 2., timePlot = timePlotCISI,  title =[], saveFigName = []):

    ordered_ISI = dp.interSpikeIntervals(spike_list, range(1,neurons))
    
    col = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    suf = lambda n: "%d%s"%(n,{1:"st",2:"nd",3:"rd"}.get(n if n<20 else n%10,"th"))
    
    firstSpike = [ts - 1 for ts in ordered_ISI[0]]
    percOfNeuronsFiring = 1#float(len(firstSpike))/float(neurons)
    stepY = np.linspace(0, percOfNeuronsFiring, len(firstSpike))
    plt.step(firstSpike, stepY, color = col[0], label=suf(1) + ' spike')
    
    for n in range(1,count_ISI):
        correctedISI = [(o - t_ref) for o in ordered_ISI[n]]
        #percOfNeuronsFiring = float(len(correctedISI))/float(neurons)
        stepY = np.linspace(0, percOfNeuronsFiring, len(correctedISI))
        plt.step(correctedISI, stepY , color = col[n], label= suf(n+1) + ' spike')
        
    if refISI:
        #percOfNeuronsFiring = float(len(refISI))/float(neurons)#/dp.countISI_merge
        stepY = np.linspace(0, percOfNeuronsFiring, len(refISI))
        plt.step(refISI, stepY , color = 'k', label = 'Before learning',linestyle='--')
        
    if len(refISI) >1:
        plt.legend()
    #plt.title("Time Between Spikes")
    plt.xlabel('Inter-Spike Interval')
    plt.ylabel('Cumulative Probability')
    plt.ylim(0,1)
    plt.xlim(0,timePlot)
    plt.title(title)
    if savePlots:
        #print('Print should work!')
        plt.savefig(path_fig +'/'+ saveFigName +'.pdf', format = 'pdf')
    if showPlots:
        plt.show()


def plot_spike_count_evoluiton(files_list, time_intervals, neurons, legend_list = [], saveFig = True):
    idx = 0
    for f in files_list:
        sim_results = dp.read_simulation_results(f)
        sc = spike_count_evolution(sim_results, time_intervals, neurons )
        if legend_list:
            plt.semilogx(time_intervals, sc[2], label = legend_list[idx])
        else:
            plt.semilogx(time_intervals, sc[2])
        plt.title('Evoluiton of the spike count')
        plt.xlabel('Time Interval (ms)')
        plt.ylabel('Spike Count Change(%)')
        plt.legend(fontsize="large")
        idx = idx + 1
    if saveFig:
        plt.savefig('./spikeCountEvolution.pdf', format = 'pdf')
    plt.show()