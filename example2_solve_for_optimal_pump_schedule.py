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
from example2_set_up_pump_scheduling import *
from textwrap import wrap

no_ts = 24

inp_file = 'Networks/Net1_casestudy2.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~diurnal~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

basic_res = run_gurobi(inp_file, no_ts, ob_fn = 'diurnal')
QP_TS_stat = basic_res['QP_TS']
HT_TS_stat = basic_res['HT_TS']
time_step = wn.options.time.report_timestep

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~static~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
basic_res2 = run_gurobi(inp_file, no_ts, ob_fn = 'static')
QP_TS_diur = basic_res2['QP_TS']
HT_TS_diur = basic_res2['HT_TS']

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~reverse~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
basic_res3 = run_gurobi(inp_file, no_ts, ob_fn = 'reverse')
QP_TS_rev = basic_res3['QP_TS']
HT_TS_rev = basic_res3['HT_TS']

# In[] Power approximation and cost functions

q_pump = results.link['flowrate'].loc[:,'9']
h_pump = results.node['head'].loc[:,'10']-results.node['head'].loc[:,'9']
power = q_pump*h_pump
mean_power = np.mean(power)*9.81*3600/1000
off_peak =  0.01870  
mid_peak = 0.02877 
on_peak = 0.05898

diurnal_cost_function = [off_peak * mean_power] * 7 + [mid_peak * mean_power] * 8 + [on_peak * mean_power] * 3 + [mid_peak * mean_power] * 4 + [off_peak * mean_power] * 2

reverse_diurnal_cost_function = [on_peak * mean_power] * 7 + [mid_peak * mean_power] * 8 + [off_peak * mean_power] * 3 + [mid_peak * mean_power] * 4 + [on_peak * mean_power] * 2

penalties = [0.5 * mean_power * np.mean([off_peak,mid_peak,on_peak])] * (no_ts-1)

static_cost = np.mean(diurnal_cost_function) 

static_cost_function = [static_cost] * no_ts

# In[] Plot optimized pump flow for the different cost functions

x_plot = np.linspace(0,no_ts-1,no_ts)
x_plot_tank = np.linspace(0,no_ts, no_ts+1)
fc = '#2DA8D8FF'
red = '#D9514EFF'
lw = 3
fs = 18

from matplotlib.lines import Line2D

fig, ax = plt.subplots(2,2,figsize = (12,8))

ax[0,1].plot(x_plot, 3600*QP_TS_stat.T, fc, lw = lw)
ax2 = ax[0,1].twinx()
ax2.step(x_plot, diurnal_cost_function, linestyle = '--', color = red, label = '$c_1$', lw = lw)
ax2.set_ylabel('Cost structure [$/hr]', fontsize = fs)
ax2.set_yticks([0,2,4,6,8,10,12,14])
lr = ax[0,1].get_ylim()
ax[0,0].set_xticks([0,4,8,12,16,20,24])

ax[0,0].text(12,400,'no objective function', fontsize = 14,color = red, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3, 'edgecolor':red})
ax[0,1].text(19,400,'$c_1$', fontsize = 16,color = red, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2, 'edgecolor':red})
ax[1,0].text(19,400,'$c_2$', fontsize = 16,color = red, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2, 'edgecolor':red})
ax[1,1].text(19,400,'$c_3$', fontsize = 16,color = red, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2, 'edgecolor':red})


custom_lines = [Line2D([0], [0], color=fc, lw=3),
                Line2D([0], [0], color=red, linestyle = '--',lw=3)]

ax[0,0].plot(x_plot,3600*results.link['flowrate'].loc[:(no_ts-1)*time_step, wn.pump_name_list[0]], fc, lw = lw)
ax[0,0].legend()  
ax[0,0].set_ylabel('Pump flow rate [m$^3$/hr]', fontsize = fs)
ax[0,0].set_ylim(lr[0],lr[1])
ax2.set_xticks([0,4,8,12,16,20,24])
ax[0,0].legend(custom_lines, ['Pump flow rate', 'Cost structure'], loc = 'lower left', frameon = False, fontsize = 12)


ax[1,0].plot(x_plot,  3600*QP_TS_diur.T, fc, lw = lw)  
ax3 = ax[1,0].twinx()
ax3.step(x_plot, static_cost_function, linestyle = '--', color = red, label = '$c_2$', lw = lw)
ax[1,0].set_xlabel('Time [hr]', fontsize = fs)
ax[1,0].set_ylabel('Pump flow rate [m$^3$/hr]', fontsize = fs)
ax3.set_xticks([0,4,8,12,16,20,24])
ax3.set_yticks([0,2,4,6,8,10,12,14])


ax[1,1].plot(x_plot,  3600*QP_TS_rev.T, fc, lw = lw)    
ax4 = ax[1,1].twinx()
ax4.step(x_plot, reverse_diurnal_cost_function, linestyle = '--', color = red, label = '$c_3$', lw = lw)
ax[1,1].set_xlabel('Time [hr]', fontsize = fs)
ax4.set_ylabel('Cost structure [$/hr]', fontsize = fs)
ax4.set_xticks([0,4,8,12,16,20,24])
ax4.set_yticks([0,2,4,6,8,10,12,14])
fig.tight_layout()


print('Diurnal Cost:', sum(diurnal_cost_function*np.array([1,1,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])))
print('Objective function value:', basic_res['of'])
print('Running time:', basic_res['Running time'])

print('Static Cost:', sum(static_cost_function*np.array([1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0])))
print('Objective function value:', basic_res2['of'])
print('Running time:', basic_res2['Running time'])

print('Reverse Diurnal Cost:', sum(reverse_diurnal_cost_function*np.array([1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0])))
print('Objective function value:', basic_res3['of'])
print('Running time:', basic_res3['Running time'])
