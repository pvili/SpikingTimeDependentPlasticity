# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:25:28 2018

@author: user
"""

import singleNeuron as sN
import plot_functions as pF
import test_functions as tF

saveFig = True

##
#3 NEURONS TOY MODEL
simulation_steps = 180
repetitions = 201
sN.learn.stdp_excitatory_weight_max = 30
sN.refractory_period = 10

inputs_two_neurons = [[1.0,15.], [10.0,15.]]
sN.learn.stdp_nu_E = 0.15

membrane_potential_list, firing_times = sN.run_single_neuron(inputs_two_neurons, repetitions, simulation_steps)

pF.plot_membrane_potentials(membrane_potential_list, [0, 50, 100, 150], saveFigName = '3Neurons_membranePot')

sN.learn.stdp_nu_E = 0.01
sN.learn.stdp_excitatory_weight_max = 10
sN.refractory_period = 0.7

sN.learn.ratio_ie = 2

inputs_EI = [[1.0,5.],[2.0,4.],[3.0,6.],[5.0,-7.], [6.0,-7.], [8.0,9.], [9.0,9.], [10.0,9.], [11.0,9.], [11.0,9.]]
membrane_potential_list_inh, firing_times_inh = sN.run_single_neuron(inputs_EI, repetitions, simulation_steps)

pF.plot_membrane_potentials(membrane_potential_list_inh, [0, 50, 100, 150], saveFigName = 'finputsEI_membranePot')

sN.learn.ratio_ie = 4

# Deptression must over-compenate potentiation
##LTD > LTP

interspike_time = 3.5
spike_strength = 5.55
spike_count = 45
simulation_steps = 2600
repetitions = 101

last_spike = [spike_count*interspike_time-3.+1,2]
inputs_regular = [[interspike_time*s+1, spike_strength] for s in range(spike_count)]
inputs_regular.append(last_spike)

membrane_potential_list, firing_times = sN.run_single_neuron(inputs_regular, repetitions, simulation_steps)

pF.plot_membrane_potentials(membrane_potential_list, [100, 70, 0] , saveFigName = 'LTPlessLTD_membranePot')
pF.plot_firing_time(firing_times, 90, saveFigName = 'LTPlessLTD_firingTimes')


#
## LTD = LTP
sN.learn.stdp_learn_ratio = 1.
interspike_time = 3.5
spike_strength = 5.55
spike_count = 45
simulation_steps = 2600
repetitions = 101

last_spike = [spike_count*interspike_time-3.+1,2]
inputs_regular = [[interspike_time*s+1, spike_strength] for s in range(spike_count)]
inputs_regular.append(last_spike)

membrane_potential_list, firing_times = sN.run_single_neuron(inputs_regular, repetitions, simulation_steps)

pF.plot_membrane_potentials(membrane_potential_list, [100, 70, 0], saveFigName = 'LTPeqLTD_membranePot')
pF.plot_firing_time(firing_times,90, saveFigName = 'LTPeqLTD_firingTimes')

sN.learn.stdp_learn_ratio = 1.5


# Disapperance of late spike


interspike_time = 5.
spike_strength = 7.5
spike_count = 15
simulation_steps = 1300
repetitions = 501
late_spike_w = 5

last_spike = [spike_count*interspike_time-3.+1,late_spike_w]
force_post = [0.5,1000]
inputs_regular = [[interspike_time*s+1, spike_strength] for s in range(spike_count)]
inputs_regular.append(last_spike)
inputs_regular.insert(0,force_post)

membrane_potential_list, firing_times = sN.run_single_neuron(inputs_regular, repetitions, simulation_steps)

pF.plot_membrane_potentials(membrane_potential_list, [300, 70, 0] , saveFigName = 'presynSaturation_membranePot')
pF.plot_firing_time(firing_times, 300, saveFigName = 'presynSaturation_firingTimes')
#NOTE: IT MIGHT TAKE A FEW REPETITIONS!
interspike_time = 3.5
spike_strength = 5.55
spike_count = 15
simulation_steps = 2600
repetitions = 501
noise_time_interval = [-5, 5]
late_spike_w = 2
noise_level = 0.3

last_spike = [spike_count*interspike_time-3.+1,late_spike_w]
force_post = [0.5,1000]
inputs_regular = [[interspike_time*s+1, spike_strength] for s in range(spike_count)]
inputs_regular.append(last_spike)
inputs_regular.insert(0,force_post)

membrane_potential_list, firing_times = sN.run_single_neuron(inputs_regular, repetitions, simulation_steps, noise_level = noise_level)

pF.plot_membrane_potentials(membrane_potential_list, [300, 10, 0] , saveFigName = 'postsynDisapp_membranePot')
pF.plot_firing_time(firing_times, 300, saveFigName = 'postsynDisapp_firingTimes')


## example many spikes disappear into one
pattern_times = [10., 13., 16., 20]#[14., 16., 18., 20.]
pattern_weights = [9., 9., 9., -0.1] #[ 8., 8., 8., -0.1]
pattern_repetitions = 50
period = max(pattern_times)
simulation_steps = int(pattern_repetitions*period/sN.dt + 1)
noise_level = 0.
repetitions = 101#10001

input_spikes = []
for r in range(pattern_repetitions):
    delay = r*period
    times = [t + delay for t in pattern_times]
    spikes = [[t, w] for (t,w) in zip(times, pattern_weights)]
    input_spikes.extend(spikes)

membrane_potential_list, firing_times = sN.run_single_neuron(input_spikes, repetitions, simulation_steps, noise_level = noise_level)

pF.plot_membrane_potentials(membrane_potential_list, [100, 50, 1], saveFigName = 'manySpikesTo1_membranePot')
pF.plot_firing_time(firing_times, repetitions, saveFigName = 'manySpikesTo1_firingTimes')

# with noise
noise_level = 1.
membrane_potential_list, firing_times = sN.run_single_neuron(input_spikes, repetitions, simulation_steps, noise_level = noise_level)

pF.plot_membrane_potentials(membrane_potential_list, [100, 50, 1], saveFigName = 'manySpikesTo1_membranePotNoise')
pF.plot_firing_time(firing_times, repetitions, saveFigName = 'manySpikesTo1_firingTimesNoise')

#
### Evolution of weights
#
#max_W = 27
#sN.learn.stdp_excitatory_weight_max = max_W
#delta_t = 20
#we = tF.test_two_spikes_evolution(delta_t, [0,max_W], 18, 5)
#wt = []#tF.two_spikes_weight_evolution([0.5,max_W*0.9], [0, delta_t], 5000)
#pF.plot_2spikes_weight_evolution(we, wt, delta_t,saveFigName = 'TwoWeightsEvolutionTwoFixedPts')
#sN.learn.stdp_excitatory_weight_max = 10
#
#max_W = 23
#sN.learn.stdp_excitatory_weight_max = max_W
#delta_t = 20
#we = tF.test_two_spikes_evolution(delta_t, [0,max_W], 18, 5)
#wt = []#tF.two_spikes_weight_evolution([0.5,max_W*0.9], [0, delta_t], 5000)
#pF.plot_2spikes_weight_evolution(we, wt, delta_t, saveFigName = 'TwoWeightsEvolutionOneFixedPt')
#

### Compute evolution of random spike trains
time_window = 40
nI = 2
nE = 8
max_E = 10
max_I = 40
learn_steps = 100
samples = 1000
noise_level = 1.
#

sN.learn.ratio_ie = 4.

print("Statistics for no noise and E+I")
deltaT = tF.test_Dt(time_window, nI, nE, max_E, max_I, learn_steps, samples)

print("Statistics for noise and E+I")
deltaT = tF.test_Dt(time_window, nI, nE, max_E, max_I, learn_steps, samples, noise_level)

print("Statistics for no noise and E")
sN.weight_noise = []
sN.learn.ratio_ie = 0.
deltaT = tF.test_Dt(time_window, nI, nE, max_E, max_I, learn_steps, samples, 0.)

print("Statistics for noise and E")
sN.learn.ratio_ie = 0.
deltaT = tF.test_Dt(time_window, nI, nE, max_E, max_I, learn_steps, samples, noise_level)

sN.learn.ratio_ie = 4.

print(' ')
print(" Fixing a spike at t=0")
sN.initial_spike = True
init_spike = True
print("Statistics for no noise and E+I")
deltaT = tF.test_Dt(time_window, nI, nE, max_E, max_I, learn_steps, samples, [], init_spike)

print("Statistics for noise and E+I")
deltaT = tF.test_Dt(time_window, nI, nE, max_E, max_I, learn_steps, samples, noise_level, init_spike)

print("Statistics for no noise and E")
sN.weight_noise = []
sN.learn.ratio_ie = 0.
deltaT = tF.test_Dt(time_window, nI, nE, max_E, max_I, learn_steps, samples, init_spike)

print("Statistics for noise and E")
sN.learn.ratio_ie = 0.
deltaT = tF.test_Dt(time_window, nI, nE, max_E, max_I, learn_steps, samples, noise_level, init_spike)

sN.initial_spike = True
#pF.plot_Dt_histogram(deltaT, saveFigName = 'DeltaTHist')