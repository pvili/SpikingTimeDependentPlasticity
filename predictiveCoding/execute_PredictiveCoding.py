import spiking_neural_networks as snn
#import data_processing as dp
import plotingFunctions as pF
import test_functions as tF
import numpy as np
import csv

##
######PREDICTIVE CODING
#repetitions = 201
#noise = 4.
#pop_counts = [50, 50, 50]
#source_pop1 = [0, 1, 25, [0, 1000], [0., 10.]]
#source_pop2 = [0, 2, 25, [1000, 2000], [0., 10.]]
#source_pop3 = [0, 3, 25, [2000, 3000], [0., 10.]]
#pop1_pop1 = [1, 1, 10, [5, 45], [0., 1.]]
#pop2_pop2 = [2, 2, 10, [5, 45], [0., 1.]]
#pop3_pop3 = [3, 3, 2, [5, 15], [0., 1.]]
#pop1_pop2 = [1, 2, 15, [5, 45], [0., 5.]]
#pop2_pop1 = [2, 1, 15, [5, 45], [0., 5.]]
#pop2_pop3 = [2, 3, 15, [5, 45], [0., 5.]]
#pop3_pop2 = [3, 2, 15, [5, 45], [0., 5.]]
#pop1_pop3 = [1, 3, 15, [5, 45], [0., 5.]]
#pop3_pop1 = [3, 1, 15, [5, 45], [0., 5.]]
#connections = [source_pop1, source_pop2, source_pop3, pop1_pop1, pop2_pop2,
#               pop3_pop3, pop1_pop2, pop2_pop1, pop2_pop3, pop3_pop2, 
#               pop1_pop3, pop3_pop1]
#connections = [source_pop1, source_pop2, pop1_pop2, pop2_pop1, source_pop3, pop2_pop3, pop3_pop2, pop1_pop3, pop3_pop1]
#
#SNN = snn.LIF_Network(pop_counts, connections)
#SNN.initial_noise()
#SNN.initial_noise()
#list_sim = tF.test_basic(repetitions, SNN, noise)
#
#
#
##with open("spikes_predictiveCoding.csv","w") as f:
##    wr = csv.writer(f)
##    wr.writerows(list_sim)
#
#pF.rasterPlotPop(list_sim, [[1,100],[101,200], [201,300]], ['b', 'r', 'g', 'k'], "testPrediction")

#
#
###### Simple Predictive
repetitions = 101
noise = 2.
pop_counts = [50, 50]
source_pop1 = [0, 1, 15, [0, 500], [0., 10.]]
source_pop2 = [0, 2, 15, [400, 900], [0., 10.]]
pop1_pop2 = [1, 2, 15, [1, 5], [0., 1.]]
connections = [source_pop1, source_pop2, pop1_pop2]
SNN = snn.LIF_Network(pop_counts, connections)

list_sim = tF.test_basic(repetitions, SNN, noise)

pF.rasterPlotPop(list_sim, [[1,50],[51,100]], ['b', 'r', 'g', 'k'], "testPrediction")

###### 3 Predictions
repetitions = 201
noise = 2.
pop_counts = [50, 50, 50]
source_pop1 = [0, 1, 8, [0, 500], [0., 10.]]
source_pop2 = [0, 2, 7, [500, 1000], [0., 10.]]
source_pop3 = [0, 3, 7, [1000, 1500], [0., 10.]]
pop1_pop2 = [1, 2, 6, [1, 5], [0., 0.1]]
pop1_pop3 = [1, 3, 6, [1, 5], [0., 0.1]]
pop2_pop3 = [2, 3, 6, [1, 5], [0., 0.1]]
pop2_pop1 = [1, 2, 6, [1, 5], [0., 0.1]]
pop3_pop1 = [1, 3, 6, [1, 5], [0., 0.1]]
pop3_pop2 = [2, 3, 6, [1, 5], [0., 0.1]]
connections = [source_pop1, source_pop2, source_pop3, pop1_pop2, pop2_pop1, pop2_pop3, pop3_pop2, pop3_pop1]
SNN = snn.LIF_Network(pop_counts, connections)

list_sim = tF.test_basic(repetitions, SNN, noise)

pF.rasterPlotPop(list_sim, [[1,50],[51,100], [100,150]], ['b', 'r', 'g', 'k'], "testPrediction")
