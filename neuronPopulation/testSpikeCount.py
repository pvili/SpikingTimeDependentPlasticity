import spiking_neural_networks as snn
import data_processing as dp
import plotingFunctions as pF
import test_functions as tF
import numpy as np
from math import factorial
import time


import matplotlib.pyplot as plt

repetitions =301 #501

print('Running File: Test Spike Implosion')

neurons = 1000#5000
t_ref = 2.
time_intervals_count = [50, 100, 500, 1000, 5000, 10000, 50000]
sync_tau = 10#10
max_time = 100
sync_window = [0, 50]





def estimateSpikeCount(samples_first, samples_ISI, time_intervals = time_intervals_count, neurons = neurons):
    time_interv = time_intervals[:]
    time_max = max(time_interv)
    spike_times = dp.probabilistic_spike_sampling_population(samples_first, samples_ISI, time_max, neurons)
    
    spikesCount = []
    count = 0
    t = 0
    next_time_limit = time_interv.pop(0)
    
    while time_interv:
        
        while t < next_time_limit:
            t = spike_times.pop(0)
            count = count + 1
        
        spikesCount.append(count)
        next_time_limit = time_interv.pop(0)
    
    while t < next_time_limit:
        t = spike_times.pop(0)
        count = count + 1
        
    spikesCount.append(count)
    
    return spikesCount

def computeSynchronization(spikes, neurons, window_count = sync_tau, time_interval = [0, 100]):
    (S_max, t_max) = dp.max_spikes_in_window(spikes, neurons, window_count)
    
    if time_interval:
        spikes_count = dp.count_spikes_in_time_window(spikes, time_interval)
        spikeRate = float(spikes_count)/float(max(time_interval) - min(time_interval))
    else:
        time_length = (spikes[-1])[1] #use the time of the last spike as proxy
        spikeRate = len(spikes)/time_length
        
    averageSpikesPerTau = spikeRate*window_count
    synch = float(S_max)/averageSpikesPerTau
    
    return synch
    
def calculateSynchEvolution(sim_results, neurons, tau = sync_tau, rep = [0,-1]):
    synch_list= []
    for r in rep:
        s = computeSynchronization(sim_results[r], neurons, tau)
        synch_list.append(s)
    
    return synch_list
    
def plot_synch_evolution(filename, neurons, tau, repetitions = range(0,300), label = []):
    sim_results = dp.read_simulation_results(filename)
    synch_list = calculateSynchEvolution(sim_results, neurons, tau, repetitions)
    plt.plot(repetitions, synch_list, label=label)

def print_synch_evoluiton(filename, neurons, tau):
    sim_results = dp.read_simulation_results(filename)
    synch_list = calculateSynchEvolution(sim_results, neurons, tau)
    print('Synchrony from ' + str(synch_list[0]) + ' to ' + str(synch_list[1]))

def spike_count_evolution(sim_results, time_intervals = time_intervals_count, neurons = 2000):
    [init_first, init_ISI] = dp.extract_samples(sim_results[0], neurons)
    [final_first, final_ISI] = dp.extract_samples(sim_results[-1], neurons)
    spikes_count_init = estimateSpikeCount(init_first, init_ISI, time_intervals , neurons)
    spikes_count_final = estimateSpikeCount(final_first, final_ISI, time_intervals , neurons)
    
    print('Init Count: '+str(spikes_count_init))
    print('Fin  Count: '+str(spikes_count_final))
    ratio = [round(float(fin - init)/init*100,1) for (init, fin) in zip(spikes_count_init, spikes_count_final)]
    print('Ratio (%): '+ str(ratio))
    
    return (spikes_count_init, spikes_count_final, ratio)


#
start = time.time()
print('Synch and Count Evolution')

rateVec = [0.1, 0.33, 1]
timeIntervalVec = [500, 200, 100, 50]
inputProbVec = [0.1, 0.33, 0.66, 1] #For simulations change to 0.9

table_initial_count = []
table_final_count = []

for rate_idx in range(len(rateVec)):
    inputRate = rateVec[rate_idx]
    timeInterval = timeIntervalVec[rate_idx]
    snn.delay_SP_max = timeInterval
    
    for inputProb in inputProbVec:
        nameFile = 'simR'+str(inputRate)+'P'+str(inputProb) 
        print('Plot for Rate: '+str(inputRate)+' and Input Prob: '+str(inputProb) )
        
        snn.delay_SP_max = timeInterval
        SNN = snn.LIF_Network(neurons, int(inputRate*timeInterval))
        SNN.initial_noise()
        sim_results = tF.test(repetitions, SNN, inputProb)
        
        dp.write_simulation_results(sim_results, nameFile)
#        sim_results = dp.read_simulation_results(nameFile)
        
        spikeCountEvolution = spike_count_evolution(sim_results, neurons =  neurons)   
        table_initial_count.append(spikeCountEvolution[0])
        table_final_count.append(spikeCountEvolution[1])

        pF.savePlots = False

        endMain = time.time()
        print('Time main: ' + str(int(endMain-start)/60) + ' minutes')
        start = endMain

dp.write_spike_count_table(table_initial_count, table_final_count, './tableSpikeCount.csv', time_intervals_count, rateVec, inputProbVec)
files_list = ['simR1P0.33', 'simR0.33P0.33', 'simR1P1', 'simR0.33P1']
legend_list = ['IR = 1 Hz, SP = 0.33','IR = 0.33 Hz, SP = 0.33','IR = 1 Hz, SP = 1','IR = 0.33 Hz, SP = 1']
time_intervals = [50*10**e for e in np.linspace(0,1.5,15)]
pF.plot_spike_count_evoluiton(files_list, time_intervals, neurons, legend_list = legend_list, saveFig = True)


print('Tau = 2, SP = 0.33, IR = 0.33')
print_synch_evoluiton('simR0.33P0.33', range(1,neurons), 2)
print('Tau = 5, P = 0.33, IR = 0.33')
print_synch_evoluiton('simR0.33P0.33', range(1,neurons), 5)
print('Tau = 10, P = 0.33, IR = 0.33')
print_synch_evoluiton('simR0.33P0.33', range(1,neurons), 10)
print('Tau = 2, P = 1, IR = 0.33')
print_synch_evoluiton('simR0.33P1', range(1,neurons), 2)
print('Tau = 5, P = 1, IR = 0.33')
print_synch_evoluiton('simR0.33P1', range(1,neurons), 5)
print('Tau = 10, P = 1, IR = 0.33')
print_synch_evoluiton('simR1P1', range(1,neurons), 10)

print('Tau = 2, SP = 0.33, IR = 1')
print_synch_evoluiton('simR1P0.33', range(1,neurons), 2)
print('Tau = 5, P = 0.33, IR = 1')
print_synch_evoluiton('simR1P0.33', range(1,neurons), 5)
print('Tau = 10, P = 0.33, IR = 1')
print_synch_evoluiton('simR1P0.33', range(1,neurons), 10)
print('Tau = 2, P = 1, IR = 1')
print_synch_evoluiton('simR1P1', range(1,neurons), 2)
print('Tau = 5, P = 1, IR = 1')
print_synch_evoluiton('simR1P1', range(1,neurons), 5)
print('Tau = 10, P = 1, IR = 1')
print_synch_evoluiton('simR1P1', range(1,neurons), 10)

