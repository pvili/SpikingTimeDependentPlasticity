#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:52:02 2019

@author: user
"""
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



print('Now testing recurrent neural networks')
count_ISI = 4


snn.recurrentConnections = True
snn.max_simulation_time = 200 # to prevent infinite runs
snn.rec_synapses_per_neuron = int(inputRate*timeInterval)/20
inputSpikes = int(inputRate*timeInterval) - snn.rec_synapses_per_neuron
SNN = snn.LIF_Network(neurons, int(inputRate*timeInterval))
SNN.initial_noise()
sim_results = tF.test(repetitions, SNN, inputProb)
##
dp.write_simulation_results(sim_results, "simMainRec.csv")
#sim_results = dp.read_simulation_results("simMainRec.csv")
##
pF.plot_spiking_rate_with_learning(sim_results[0], sim_results[repetitions - 1], neurons, timePlot= 100, title='Instantaneous Spike Rate', saveFigName = 'spikeRatesRec')
###pF.plot_max_signal(sim_results, neurons, [10], saveFigName = 'maxSignal')
###pF.plot_max_time(sim_results, neurons, saveFigName = 'maxTime')
pF.plot_spikeCount(sim_results, neurons, saveFigName = 'spikeCountRec')
pF.plot_cumulative_ISI(sim_results[0], neurons , count_ISI, [],t_ref,title = 'CDF of ISI before STDP', saveFigName = 'cumulativeISI_beforeLearningRec')
befLearningISI = dp.mergeISI(sim_results[0], merge_ISI, neurons)
pF.plot_cumulative_ISI(sim_results[repetitions-1], neurons , count_ISI, befLearningISI, t_ref, title = 'CDF of ISI after STDP', saveFigName = 'cumulativeISI_afterLearningRec')



snn.max_simulation_time = 1000000
snn.recurrentConnections = False

#
print('Now testing alternative learning rules')
count_ISI = 2
timeInterval = 300 #We only need to know the first and second
snn.delay_SP_max = timeInterval
#
##STDP inhibition longer constant than excitation
print('Tau_i > tau_e')
snn.nm.stdp_tau_i = snn.nm.stdp_tau_e * 2.0

SNN = snn.LIF_Network(neurons, int(inputRate*timeInterval))
SNN.initial_noise()
sim_results = tF.test(repetitions, SNN, inputProb)

dp.write_simulation_results(sim_results, "simTauI.csv")
sim_results = dp.read_simulation_results("simTauI.csv")


pF.plot_spiking_rate_with_learning(sim_results[0], sim_results[repetitions - 1], neurons, timePlot = 100,title = 'Instantaneous Spike Rate', saveFigName = 'spikeRates_tauI2')
pF.plot_spikeCount(sim_results, neurons, saveFigName = 'spikeCount_tauI2')
pF.plot_cumulative_ISI(sim_results[0], neurons , count_ISI, [],t_ref,  timePlot = 200,title = 'CDF of ISI before STDP', saveFigName = 'cumulativeISI_beforeLearning_tauI2')
befLearningISI = dp.mergeISI(sim_results[0], merge_ISI, neurons)
pF.plot_cumulative_ISI(sim_results[repetitions-1], neurons , count_ISI, befLearningISI, t_ref, timePlot = 200,title = 'CDF of ISI after STDP', saveFigName = 'cumulativeISI_afterLearning_tauI2')

snn.nm.stdp_tau_i = snn.nm.stdp_tau_e 

endInh = time.time()
print('Time Tau Inh: ' + str(int(endInh - endMain)/60) + ' minutes')

print('Tau_i > tau_e with modulated A')
snn.nm.stdp_tau_i = snn.nm.stdp_tau_e * 2.0
snn.nm.stdp_inhibitory_increase_nu = snn.nm.stdp_excitatory_increase_nu
snn.nm.stdp_inhibitory_decrease_nu = snn.nm.stdp_excitatory_decrease_nu
snn.nm.stdp_inhibitory_weight_max = snn.nm.stdp_excitatory_weight_max
#
SNN = snn.LIF_Network(neurons, int(inputRate*timeInterval))
SNN.initial_noise()
sim_results = tF.test(repetitions, SNN, inputProb)

dp.write_simulation_results(sim_results, "simTauI2A05.csv")
sim_results = dp.read_simulation_results("simTauI2A05.csv")


pF.plot_spiking_rate_with_learning(sim_results[0], sim_results[repetitions - 1], neurons, timePlot = 200,title = 'Instantaneous Spike Rate', saveFigName = 'spikeRates_tauI2A05')
pF.plot_spikeCount(sim_results, neurons, saveFigName = 'spikeCount_tauI2A05')
pF.plot_cumulative_ISI(sim_results[0], neurons , count_ISI, [],t_ref, timePlot = 200,title = 'CDF of ISI before STDP', saveFigName = 'cumulativeISI_beforeLearning_tauI2A05')
befLearningISI = dp.mergeISI(sim_results[0], merge_ISI, neurons)
pF.plot_cumulative_ISI(sim_results[repetitions-1], neurons , count_ISI, befLearningISI, t_ref, timePlot = 200,title = 'CDF of ISI after STDP', saveFigName = 'cumulativeISI_afterLearning_tauI2A05')

snn.nm.stdp_tau_i = snn.nm.stdp_tau_e 
snn.nm.stdp_inhibitory_increase_nu = snn.nm.stdp_excitatory_increase_nu*3
snn.nm.stdp_inhibitory_decrease_nu = snn.nm.stdp_excitatory_decrease_nu*3
snn.nm.stdp_inhibitory_weight_max = 40.

endInhA2 = time.time()
print('Time Tau Inh: ' + str(int(endInhA2 - endInh)/60) + ' minutes')



snn.nm.stdp_inhibitory_decrease_nu = snn.nm.stdp_excitatory_increase_nu*1.5

snn.nm.stdp_tau_i = snn.nm.stdp_tau_e

print(' EXCITATORY STDP')
# STDP with weight normalization
print('Weight Normalization')
weight_normalization = True
weight_decay_e = 0.1*snn.nm.stdp_excitatory_increase_nu

SNN = snn.LIF_Network(neurons, int(inputRate*timeInterval))
SNN.initial_noise()
sim_results = tF.test(repetitions, SNN, inputProb, weight_normalization)

dp.write_simulation_results(sim_results, "simWeightNorm.csv")
sim_results = dp.read_simulation_results("simWeightNorm.csv")

pF.plot_spiking_rate_with_learning(sim_results[0], sim_results[repetitions - 1], neurons, saveFigName = 'spikeRates_wNorm')
pF.plot_spikeCount(sim_results, neurons, saveFigName = 'spikeCount_wNorm')
pF.plot_cumulative_ISI(sim_results[0], neurons , count_ISI, [],t_ref,title = 'CDF of ISI before STDP', saveFigName = 'cumulativeISI_beforeLearning_wNorm')
befLearningISI = dp.mergeISI(sim_results[0], merge_ISI, neurons)
pF.plot_cumulative_ISI(sim_results[repetitions-1], neurons , count_ISI, befLearningISI, t_ref,title = 'CDF of ISI after STDP', saveFigName = 'cumulativeISI_afterLearning_wNorm')

weight_normalization = False
weight_decay_e = 0.

endWnorm= time.time()
print('Time weight normalization: ' + str(int(endWnorm - endInhA2 )/60) + ' minutes')





# Triplet rule for excitation
print('Triplet rule')
snn.nm.triplet_rule_active = True

SNN = snn.LIF_Network(neurons, int(inputRate*timeInterval))
SNN.initial_noise()
sim_results = tF.test(repetitions, SNN, inputProb)

dp.write_simulation_results(sim_results, "simTriplet.csv")
sim_results = dp.read_simulation_results("simTriplet.csv")

  
pF.plot_spiking_rate_with_learning(sim_results[0], sim_results[repetitions - 1], neurons, saveFigName = 'spikeRates_triple')
pF.plot_spikeCount(sim_results, neurons, saveFigName = 'spikeCount_triple')
pF.plot_cumulative_ISI(sim_results[0], neurons , count_ISI, [],t_ref,title = 'CDF of ISI before STDP', saveFigName = 'cumulativeISI_beforeLearning_triple')
befLearningISI = dp.mergeISI(sim_results[0], merge_ISI, neurons)
pF.plot_cumulative_ISI(sim_results[repetitions-1], neurons , count_ISI, befLearningISI, t_ref,title = 'CDF of ISI after STDP', saveFigName = 'cumulativeISI_afterLearning_triple')

snn.nm.triplet_rule_active = False

endTriplet = time.time()
print('Time Triplet: ' + str(int(endTriplet - endWnorm)/60) + ' minutes')




print('INHIBITORY STDP')

print('Reverse STDP')

snn.nm.inhibitory_kernel = 'r'
snn.nm.stdp_inhibitory_decrease_nu = snn.nm.stdp_inhibitory_increase_nu

SNN = snn.LIF_Network(neurons, int(inputRate*timeInterval))
SNN.initial_noise()
sim_results = tF.test(repetitions, SNN, inputProb)

dp.write_simulation_results(sim_results, "simInhRev.csv")
sim_results = dp.read_simulation_results("simInhRev.csv")

pF.plot_spiking_rate_with_learning(sim_results[0], sim_results[repetitions - 1], neurons, saveFigName = 'spikeRates_InhRev')
pF.plot_spikeCount(sim_results, neurons, saveFigName = 'spikeCount_InhRev')
pF.plot_cumulative_ISI(sim_results[0], neurons , count_ISI, [],t_ref,title = 'CDF of ISI before STDP', saveFigName = 'cumulativeISI_beforeLearning_InhRev')
befLearningISI = dp.mergeISI(sim_results[0], merge_ISI, neurons)
pF.plot_cumulative_ISI(sim_results[repetitions-1], neurons , count_ISI, befLearningISI, t_ref,title = 'CDF of ISI after STDP', saveFigName = 'cumulativeISI_afterLearning_InhRev')

snn.nm.stdp_inhibitory_decrease_nu = 1.5*snn.nm.stdp_inhibitory_increase_nu

endIR = time.time()
print('Time Inh Rev: ' + str(int(endIR - endTriplet)/60) + ' minutes')





print('Sombrero')
snn.nm.inhibitory_kernel = 's'
snn.nm.stdp_inhibitory_decrease_nu = snn.nm.stdp_inhibitory_increase_nu

SNN = snn.LIF_Network(neurons, int(inputRate*timeInterval))
SNN.initial_noise()
sim_results = tF.test(repetitions, SNN, inputProb)

dp.write_simulation_results(sim_results, "simInhSom.csv")
sim_results = dp.read_simulation_results("simInhSom.csv")

pF.plot_spiking_rate_with_learning(sim_results[0], sim_results[repetitions - 1], neurons, saveFigName = 'spikeRates_InhSomb')
pF.plot_spikeCount(sim_results, neurons, saveFigName = 'spikeCount_InhSomb')
pF.plot_cumulative_ISI(sim_results[0], neurons , count_ISI, [],t_ref,title = 'CDF of ISI before STDP', saveFigName = 'cumulativeISI_beforeLearning_InhSomb')
befLearningISI = dp.mergeISI(sim_results[0], merge_ISI, neurons)
pF.plot_cumulative_ISI(sim_results[repetitions-1], neurons , count_ISI, befLearningISI, t_ref,title = 'CDF of ISI after STDP', saveFigName = 'cumulativeISI_afterLearning_InhSomb')

snn.nm.stdp_inhibitory_decrease_nu = 1.5*snn.nm.stdp_inhibitory_increase_nu
snn.nm.inhibitory_kernel = 'c'


endSomb = time.time()
print('Time Inh sombrero: ' + str(int(endSomb - endIR)/60) + ' minutes')




## Test poissonian input (delays rewired)
print('Poissonian Input')

timeInterval = 200 #We only need to know the first and others

SNN = snn.LIF_Network(neurons, int(inputRate*timeInterval))
SNN.initial_noise()
sim_results = tF.test_poissonian_input(repetitions, SNN, [0,timeInterval])

dp.write_simulation_results(sim_results, "simPoisson.csv")
sim_results = dp.read_simulation_results("simPoisson.csv")

pF.plot_spikeCount(sim_results, neurons, saveFigName = 'spikeCountPoissonian')
pF.plot_cumulative_ISI(sim_results[0], neurons , count_ISI, [],t_ref, saveFigName = 'cumulativeISI_beforeLearning_poisson')
befLearningISI = dp.mergeISI(sim_results[0], merge_ISI, neurons)
pF.plot_cumulative_ISI(sim_results[repetitions-1], neurons , count_ISI, befLearningISI, t_ref, saveFigName = 'cumulativeISI_afterLearning_poisson')

endPoiss = time.time()
print('Time Poisson: ' + str(int(endPoiss - endIR)/60) + ' minutes')
