import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import wntr
import scipy.sparse as sp
import networkx as nx
import numpy.linalg as la
import time
from scipy.optimize import curve_fit
import re
import datetime
from matplotlib import gridspec
from example3_build_general_24hr_2hrts_pumpschedule_optimizationproblem import *
from textwrap import wrap

t1 = time.time()

no_ts = 12
ob_fn = 'diurnal'
inp_file = 'Networks/Net3_delBP_2hr.inp'

wn = wntr.network.WaterNetworkModel(inp_file)
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~diurnal~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Here we run an optimization problem with a "realistic" diurnal objective function
basic_res = run_gurobi(inp_file, no_ts, ob_fn = 'diurnal',  
                       num_pipe_seg =1, num_pump_seg = 5, 
                       obj = True, penalty = True)
QP_TS_diur = basic_res['QP_TS']
HT_TS_diur = basic_res['HT_TS']

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~static~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Here we run an optimization problem with a static, unchanging objective function 
basic_res2 = run_gurobi(inp_file, no_ts, ob_fn ='static',  
                       num_pipe_seg =1, num_pump_seg = 5, 
                       obj = True, penalty = True)
QP_TS_stat = basic_res2['QP_TS']
HT_TS_stat = basic_res2['HT_TS']

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~reverse~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Here we run an optimization problem with an unrealistic objective function
basic_res3 = run_gurobi(inp_file, no_ts, ob_fn = 'reverse',  
                       num_pipe_seg =1, num_pump_seg = 5, 
                       obj = True, penalty = True)
QP_TS_rev = basic_res3['QP_TS']
HT_TS_rev = basic_res3['HT_TS']

time_step = wn.options.time.report_timestep

# In[]

# Cost functions
diurnals = []
statics = []
reverses = []
penalties = []

for pump_name in [wn.pump_name_list[0]]:
    power = results.link['flowrate'].loc[:,pump_name]*-results.link['headloss'].loc[:,pump_name]
    mean_power = power[power>0].mean()*9.81*3600/1000
    
    off_peak =  0.01870  
    mid_peak = 0.02877 
    on_peak = 0.05898
    
    diurnal_cost_function = [off_peak * mean_power] * 3 + [mid_peak * mean_power] * 4 + [on_peak * mean_power] * 2 + [mid_peak * mean_power] * 2 + [off_peak * mean_power] * 1
    
    reverse_diurnal_cost_function = [on_peak * mean_power] * 3 + [mid_peak * mean_power] * 4 + [off_peak * mean_power] * 2 + [mid_peak * mean_power] * 2 + [on_peak * mean_power] * 1
    
    penalty = [0.5 * mean_power * np.mean([off_peak,mid_peak,on_peak])] * (no_ts-1)
    
    static_cost = np.mean(diurnal_cost_function) 
    
    static_cost_function = [static_cost] * no_ts
    
    diurnals.append(diurnal_cost_function)
    statics.append(static_cost_function)
    reverses.append(reverse_diurnal_cost_function)
    penalties.append(penalty)


start_node = 0

x_plot = np.linspace(0,no_ts-1,no_ts)
x_plot_tank = np.linspace(0,no_ts, no_ts+1)
hc = '#2BAE66FF'
fc = '#2DA8D8FF'
grey = '#949398FF'
black = '#2A2B2DFF'
red = '#D9514EFF'
newcol = 'navy'
newcol2 = 'royalblue'
grey = 'navy'
lw = 4
fs = 18
color_list = [grey, fc, black]

from matplotlib.lines import Line2D

fig, ax = plt.subplots(3,1,figsize = (8,18))
for i in range(wn.num_pumps):
    ax[0].plot(2*x_plot, 3600*QP_TS_diur[i].T, lw = lw, color = color_list[i], label = 'Pump {}'.format(i))
ax2 = ax[0].twinx()
ax2.step(2*x_plot, diurnal_cost_function,where='mid', linestyle = '--', color = red, label = '$c_1$', lw = lw)
ax2.set_ylabel('Cost structure [$/hr]', fontsize = fs)
ax[0].set_ylabel('Pump flow rate [m$^3$/hr]', fontsize = fs)
ax[0].legend(frameon=False, loc = 'upper left', fontsize = 12)

for i in range(wn.num_pumps):
    ax[1].plot(2*x_plot, 3600*QP_TS_stat[i].T,  lw = lw, color = color_list[i], label = 'Pump {}'.format(i))
ax3 = ax[1].twinx()
ax3.step(2*x_plot, static_cost_function, linestyle = '--', color = red, label = '$c_2$', lw = lw)
ax3.set_ylabel('Cost structure [$/hr]', fontsize = fs)
ax[1].set_ylabel('Pump flow rate [m$^3$/hr]', fontsize = fs)


for i in range(wn.num_pumps):
    ax[2].plot(2*x_plot, 3600*QP_TS_rev[i].T, lw = lw, color = color_list[i], label = 'Pump {}'.format(i))
ax4 = ax[2].twinx()
ax4.step(2*x_plot, reverse_diurnal_cost_function, where='mid',linestyle = '--', color = red, label = '$c_1$', lw = lw)
ax4.set_ylabel('Cost structure [$/hr]', fontsize = fs)
ax[2].set_ylabel('Pump flow rate [m$^3$/hr]', fontsize = fs)
ax[2].set_xlabel('Time [hr]', fontsize = fs)

custom_lines = [Line2D([0], [0], color=color_list[i], lw=3) for i in range(wn.num_pumps)] + [Line2D([0], [0], color=red, linestyle = '--',lw=3)]

ax[0].legend(custom_lines, ['Pump {} flow rate'.format(i) for i in range(wn.num_pumps)] + ['Cost structure'], frameon = False, fontsize = 12)



custom_lines = [Line2D([0], [0], color=fc, lw=3),
                Line2D([0], [0], color=newcol, linestyle = ':',lw=3),
                Line2D([0], [0], color=newcol2, linestyle = '--',lw=3),
                Line2D([0], [0], color=red, linestyle = '--',lw=3)   ]


# In[]

print('c1 OF: ', basic_res['of'])
print('c2 OF: ', basic_res2['of'])
print('c3 OF: ', basic_res3['of'])

print('c1 problem run time: ', basic_res['Reduction time'])
print('c2 problem run time: ', basic_res2['Reduction time'])
print('c3 problem run time: ', basic_res3['Reduction time'])

