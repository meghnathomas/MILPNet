import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import wntr
import networkx as nx
import time
from scipy.optimize import curve_fit
import re
import datetime
from matplotlib import gridspec
from set_up_milpnet import *
from matplotlib.ticker import FormatStrFormatter

#################################### USER INPUTS #########################################################
inp_file = 'Networks/ANET.inp'   # network name
no_ts = 10              # simulation duration
##########################################################################################################

# Run WNTR hydraulic simulation and store EPANET results
wn = wntr.network.WaterNetworkModel(inp_file)
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()

##########################################################################################################################
# use the run_gurobi function from set_up_milpnet.py to obtain a dictionary with MILPNet results
# in this example we only obtain the heads of the tanks and flow rates in the pump

milpnet_results = run_gurobi(inp_file, no_ts, pump_on = False)
milpnet_tank = milpnet_results['HT_TS']
milpnet_pump = milpnet_results['QP_TS']

##########################################################################################################################
# plot tank heads and pump flow rates for MILPNet (solid line) vs EPANET (dashed line)
    
time_step = wn.options.time.report_timestep
x_plot = np.linspace(0,no_ts-1,no_ts)
x_plot_tank = np.linspace(0,no_ts, no_ts+1)
colors = ['k', 'b', 'r', 'g', 'orange', 'grey', 'pink']
lw = 3
fs = 16

fig, ax = plt.subplots(1,2,figsize = (12,6))
for i in range(len(wn.tank_name_list)):
    ax[0].plot(x_plot_tank, milpnet_tank[i].T, color = colors[i], label = 'MILPNet ' + wn.tank_name_list[i], lw = lw)
    ax[0].plot(x_plot_tank,results.node['head'].loc[:(no_ts)*time_step, wn.tank_name_list[i]],
      color = colors[i], linestyle = '--', label = 'EPANET ' + wn.tank_name_list[i], lw = lw)   
ax[0].set_ylabel('Tank head [m]', fontsize = fs)
ax[0].set_xlabel('Time [hr]', fontsize = fs)
ax[0].legend()


for i in range(len(wn.pump_name_list)):
    ax[1].plot(x_plot, 3600*milpnet_pump[i,:], color = colors[i], label = 'MILPNet ' + wn.tank_name_list[i], lw = lw)  
    ax[1].plot(x_plot, 3600*results.link['flowrate'].loc[:(no_ts-1)*time_step, wn.pump_name_list[i]], 
      color = colors[i], linestyle = '--', label = 'EPANET ' + wn.tank_name_list[i], lw = lw) 
ax[1].set_ylabel('Pump flow rate [m$^3$/hr]', fontsize = fs)
ax[1].legend()
ax[1].set_xlabel('Time [hr]', fontsize = fs)
fig.tight_layout()

##########################################################################################################################
# plot junction heads and pipe flow rates for MILPNet (solid line) vs EPANET (dashed line)

milpnet_heads = milpnet_results['H_TS']
milpnet_flows = milpnet_results['Q_TS']


# junction heads

n_junc = wn.num_junctions
n_cols = 4
n_rows = int(np.ceil(n_junc // n_cols))

fig, ax = plt.subplots(n_rows, n_cols, figsize=(12,  12 * n_rows / n_cols))

junction_names = wn.junction_name_list

for i in range(n_rows * n_cols):
    if i < len(junction_names) - 1:
        ax.flat[i].plot(x_plot,milpnet_heads[i].T, color = colors[0], lw = lw)
        ax.flat[i].plot(x_plot,results.node['head'].loc[:(no_ts-1)*time_step, wn.junction_name_list[i]],
               color = colors[2], linestyle = '--', lw = lw)
        ax.flat[i].set_title('Node ' + junction_names[i])
        ax.flat[i].yaxis.set_major_formatter(FormatStrFormatter('%d'))
        if (i % ax.shape[1]) == 0:
            ax.flat[i].set_ylabel('Head [m]')
        if (i >= ax.shape[1] * (ax.shape[0] - 1)):
            ax.flat[i].set_xlabel('Time [hr]')
    else:
        ax.flat[i].set_visible(False)
    
ax.flat[0].legend(['MILPNet','EPANET'])
plt.tight_layout()

# pipe flow rates

n_superlinks = wn.num_pipes
n_cols = 4
n_rows = int(np.ceil(n_superlinks // n_cols))

fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 12 * n_rows / n_cols))

pipe_names = wn.pipe_name_list

for i in range(n_rows * n_cols):
    if i < len(junction_names) - 1:
        ax.flat[i].plot(x_plot,milpnet_flows[i].T, color = colors[0], lw = lw)
        ax.flat[i].plot(x_plot,results.link['flowrate'].loc[:(no_ts-1)*time_step, wn.pipe_name_list[i]],
               color = colors[2], linestyle = '--', lw = lw)
        ax.flat[i].set_title('Pipe ' + pipe_names[i])
        ax.flat[i].ticklabel_format(useOffset=False)
        if (i % ax.shape[1]) == 0:
            ax.flat[i].set_ylabel('Flow [m3/s]')
        if (i >= ax.shape[1] * (ax.shape[0] - 1)):
            ax.flat[i].set_xlabel('Time [hr]')
    else:
        ax.flat[i].set_visible(False)
    
ax.flat[0].legend(['MILPNet','EPANET'])
plt.tight_layout()