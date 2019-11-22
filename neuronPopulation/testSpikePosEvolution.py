import spiking_neural_networks as snn
import data_processing as dp
import plotingFunctions as pF
import test_functions as tF
import matplotlib.pyplot as plt
import numpy as np
import time

repetitions = 401 #401

print('Running File: Test Spike Implosion')
inputRate = 0.2#0.2
inputProb = 0.33#0.5
timeInterval = 1000#1000
snn.delay_SP_max = timeInterval
neurons = 4000#2000
t_ref = 2.
count_ISI = 3
merge_ISI = 2
dp.countISI_merge = merge_ISI


start = time.time()
print('Main text')
SNN = snn.LIF_Network(neurons, int(inputRate*timeInterval))
SNN.initial_noise()
sim_results = tF.test(repetitions, SNN, inputProb)
####
dp.write_simulation_results(sim_results, "simMain.csv")
#sim_results = dp.read_simulation_results("simMain.csv")
#
pF.plot_spiking_rate_with_learning(sim_results[0], sim_results[-1], neurons, title='Instantaneous Spike Rate', saveFigName = 'spikeRates')
##pF.plot_max_signal(sim_results, neurons, [10], saveFigName = 'maxSignal')
##pF.plot_max_time(sim_results, neurons, saveFigName = 'maxTime')
pF.plot_spikeCount(sim_results, neurons, saveFigName = 'spikeCount')
pF.plot_cumulative_ISI(sim_results[0], neurons , count_ISI, [],t_ref,title = 'CDF of ISI before STDP', saveFigName = 'cumulativeISI_beforeLearning')
befLearningISI = dp.mergeISI(sim_results[0], merge_ISI, neurons)
pF.plot_cumulative_ISI(sim_results[-1], neurons , count_ISI, befLearningISI, t_ref, title = 'CDF of ISI after STDP', saveFigName = 'cumulativeISI_afterLearning')
#
endMain = time.time()
print('Time main: ' + str(int(endMain-start)/60) + ' minutes')

