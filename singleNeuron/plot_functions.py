# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 21:07:40 2018

@author: user
"""

import matplotlib.pyplot as plt
import learning_singleNeuron as learn
import singleNeuron as sN
import numpy as np
import math

arrowMod = 0.8
path_fig = '../figures/'

def plot_membrane_potentials(membrane_potentials, repetitions,
                             threshold = -50., dt=0.1, saveFigName = []):
    times = [s*dt for s in range(len(membrane_potentials[0]))]
    for r in repetitions:
        label = 'Repetition '+str(r)
        plt.plot(times, membrane_potentials[r], label= label)
        
    threshold_line = [threshold]*len(membrane_potentials[0])
    plt.plot(times,threshold_line, 'k--')
    plt.title('Single neuron membrane potential', fontsize=18)
    plt.xlabel('time (ms)', fontsize=16)
    plt.ylabel('Membrane Potential (mV)', fontsize=16)
    plt.legend()
    
    if saveFigName:
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    plt.show()
    
def plot_firing_time(firing_times, xlimit = [], saveFigName = []):
    repetitions = []
    firings = []
    r=0
    for firings_list in firing_times:
        for f_t in firings_list:
            firings.append(f_t)
            repetitions.append(r)
        r += 1
    
    axes = plt.gca()
    if len(firings)>0:
        axes.set_ylim([0,max(firings) + 10.])
    if xlimit:
        axes.set_xlim([0,xlimit])
    plt.plot(repetitions,firings,'.')
    plt.title('Single neuron firing time', fontsize=18)
    plt.xlabel('Input Repetitons', fontsize=16)
    plt.ylabel('Firing Time (ms)', fontsize=16)
    plt.legend()
    
    if saveFigName:
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    plt.show()
    
  
def plot_Dt_histogram(deltaT, bins = 20, saveFigName = []):
    fig = plt.hist(deltaT, bins)
    plt.title('Change in Firing Times', fontsize=18)
    plt.xlabel(r'$\Delta t_f$', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    if saveFigName:
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    plt.show()
    
def plot_2spikes_weight_evolution(list_weight_evolution, weight_trajectory = [], delta_t = 10, v_th = 20, saveFigName = []):
    w_max = learn.stdp_excitatory_weight_max
    
#    arrow_linefrac = [0.1, 0.5, 0.9]
#    
    x = []
    y = []
    u = []
    v = []
    
    for we in list_weight_evolution:
        w1_list, w2_list = convert_list_of_pairs_to_pair_of_lists(we)
        if w1_list[0] == w1_list[2]:
            plt.scatter(w1_list[0], w2_list[0], color = 'blue', s = 0.5)
        else:
            #weight_evol, = plt.plot(w1_list, w2_list, color = 'blue', linewidth=0.5)
            
            x0, y0, Dx, Dy = calculate_arrow(w1_list, w2_list, arrowMod)
            x.append(x0)
            y.append(y0)
            u.append(Dx)
            v.append(Dy)
            #x0, y0, Dx, Dy = calculate_arrow(w1_list, w2_list, arrowMod)
            #weight_evol = plt.arrow(x0, y0, Dx, Dy, color = 'blue')

        #ADD ARROWS
    weight_evol = plt.quiver(x, y, u, v, color = 'blue')
    
    if weight_trajectory:
        w1_list, w2_list = convert_list_of_pairs_to_pair_of_lists(weight_trajectory)
        weight_tr,= plt.plot(w1_list, w2_list, 'green', linewidth = 2, label = 'Postsynaptic Trajectory')
    
    w1_fire, w2_fire = line_2ndneuron_fires(w_max, delta_t, sN.tau_membrane)
    firing_th, = plt.plot(w1_fire, w2_fire, color = 'black', linestyle = '-.', linewidth = 3)
    plt.plot([v_th, v_th], [0, w_max], color = 'black', linestyle = '-.',linewidth = 3)
    
    learnRatio = learn.stdp_learn_ratio
    tau_s = learn.tau_stdp
    w2_nullcline = w_max/(learnRatio*math.exp(-delta_t/tau_s) + 1)
    nullcline, = plt.plot([v_th, w_max], [w2_nullcline, w2_nullcline], color = 'red', linestyle = ':')
    plt.plot([0, v_th], [w_max, w_max], color = 'red', linestyle = ':')
    plt.plot([w_max, w_max], [0, w_max], color = 'red', linestyle = ':')
    
    plt.scatter([w_max, w_max], [0, w2_nullcline], color = 'red', s= 1, label = 'Fixed points')
    
    plt.text(w_max*0.4,w_max*0.5,'a', fontsize=20, bbox=dict(facecolor='white', alpha=1, linewidth=1))
    plt.text(w_max*0.6,w_max*0.85,'b', fontsize=20, bbox=dict(facecolor='white', alpha=1, linewidth=1))
    plt.text(w_max*0.9,w_max*0.5,'c', fontsize=20, bbox=dict(facecolor='white', alpha=1, linewidth=1))
    plt.text(w_max*0.9,w_max*0.95,'d', fontsize=20, bbox=dict(facecolor='white', alpha=1, linewidth=1))
    
    
    plt.title('Weights of two spikes evolution', fontsize=18)
    plt.xlabel('W1', fontsize=16)
    plt.ylabel('W2', fontsize=16)
    
    if weight_trajectory:
        plt.legend((weight_tr, weight_evol, firing_th, nullcline), 
                   ('Postsynaptic Trajectory', 'Weight evolution', 'Firing threshold', 'Nullcline'), loc='lower left', fontsize=14)
    
    else:
        plt.legend((weight_evol, firing_th, nullcline), 
                   ('Weight evolution', 'Firing threshold', 'Nullcline'), loc='lower left', fontsize=14)
    
    if saveFigName:
        plt.savefig(path_fig + saveFigName +'.pdf', format = 'pdf')
    plt.show()
    
    
    
def calculate_arrow(line_x, line_y, modulus):
    idx = 1
    Didx = 1
    x0 = line_x[idx]
    y0 = line_y[idx]
    Dx = line_x[idx+Didx] - line_x[idx]
    Dy = line_y[idx+Didx] - line_y[idx]
    
    arrow_length = math.sqrt(Dx*Dx + Dy*Dy)
    if arrow_length > 0:
        Dx = Dx*modulus/arrow_length
        Dy = Dy*modulus/arrow_length
    else:
        Dx = 0
        Dy = 0
    
    return x0, y0, Dx, Dy

def line_2ndneuron_fires(w_max, delta_t, tau_mem, v_th = 20, linePts = 100):
    w1_val = np.ndarray.tolist(np.linspace(0, w_max, linePts))
    w2_val = []
    decay =  math.exp(-delta_t/tau_mem)
    for w1 in w1_val:
        if w1 < v_th:
            w2_val.append(v_th - w1*decay)
        else:
            w2_val.append(v_th)
    return w1_val, w2_val
        
    
def convert_list_of_pairs_to_pair_of_lists(list_pairs):
    list_1 = []
    list_2 = []
    for p in list_pairs:
        list_1.append(p[0])
        list_2.append(p[1])
        
    return list_1, list_2