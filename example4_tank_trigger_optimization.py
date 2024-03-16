# GENERAL CODE

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import wntr
import scipy.sparse as sp
import networkx as nx
import numpy.linalg as la
import time
from scipy.optimize import curve_fit
import re
import datetime


def calc_K(L, D, R):
    
    alpha = 10.66
    e1 = 1.852
    e2 = 4.87
    
    return (alpha*L)/((R**e1)*(D**e2))    

def one_point_linear(Q_op, dH_op, K):
    
    alpha = 10.66
    e1 = 1.852
    e2 = 4.87
    
    # np.sign(a) * (np.abs(a)) ** (1 / 3)
    # slope = K * e1 * Q_op**(1-e1)
    slope = K * e1 * abs(Q_op) ** (e1-1)
    intercept = dH_op - Q_op*slope
    
    return slope, intercept

def two_point_linear(Q_op, dH_op, K, range_Q):
    
    alpha = 10.66
    e1 = 1.852
    e2 = 4.87

    Q_1 = Q_op*(1-range_Q)
    Q_2 = Q_op*(1+range_Q)
    
    dH_1 = np.sign(Q_1) * K * abs(Q_1)**e1
    dH_2 = np.sign(Q_2)* K * abs(Q_2)**e1
    
    slope = (dH_2 - dH_1)/(Q_2 - Q_1)
    intercept = dH_2 - Q_2*slope
    
    return slope, intercept
    
    
# define function to calculate and store new K~ value for each pipe

def extract_params(wn):
    
    e1 = 1.852
    e2 = 4.87
    alpha = 10.67
    
    #create name list variables
    node_names = wn.node_name_list
    pipe_names = wn.pipe_name_list
    link_names = []

    #number of each element - to demarcate sections of A21
    num_nodes = wn.num_nodes
    num_junc = wn.num_junctions
    num_pipes = wn.num_pipes
    
    #find incidence matrix and A21 matrix
    pipe_tp =[]
    pipe_tp2 = []
    
    for link_name, link in wn.pipes():
        pipe_tp.append(link.start_node_name)
        pipe_tp2.append(link.end_node_name)
        link_names.append(link_name)
    
    new_pipes_list = tuple(zip(pipe_tp, pipe_tp2))
    
    for link_name, link in wn.pumps(): # added pumps to inc_mat
        pipe_tp.append(link.start_node_name)
        pipe_tp2.append(link.end_node_name)
        link_names.append(link_name)
        
    for link_name, link in wn.valves(): # added pumps to inc_mat
        pipe_tp.append(link.start_node_name)
        pipe_tp2.append(link.end_node_name)
        link_names.append(link_name)
        
    link_list = tuple(zip(pipe_tp, pipe_tp2))
        
      
    #extract A21 matrix
    G = wn.get_graph()
    
    inc_mat = sp.csr_matrix.toarray(nx.incidence_matrix(G,nodelist = node_names, edgelist = link_list, oriented=True))
    A21 = inc_mat[0:num_junc,:]

    K = np.zeros((num_pipes,1))
    
    for i in range(num_pipes):
        L = wn.get_link(pipe_names[i]).length
        D = wn.get_link(pipe_names[i]).diameter
        R = wn.get_link(pipe_names[i]).roughness
        K[i] = (alpha*L)/((R**e1)*(D**e2))   
        
    return A21, K, inc_mat, node_names, link_names

def extract_d_h(wn, results, inc_mat, mult_ts):
       
    #number of each element - to demarcate sections of A21
    num_nodes = wn.num_nodes
    junc_names = wn.junction_name_list
    num_junc = wn.num_junctions
    
    #retrieve known heads and demands from inp file
    special_heads = np.zeros((num_nodes-num_junc,1))
    heads_count = 0
    ts = wn.options.time.report_timestep
        
    #returns list of junction demands
    junc_demands = np.zeros((num_junc,1))
    for i in range(num_junc):
        junc_demands[i] = results.node['demand'].loc[ts*mult_ts, junc_names[i]]
    
    #stores head of reservoirs
    for res_name, res in wn.reservoirs():
        special_heads[heads_count] = results.node['head'].loc[ts*mult_ts, res_name]
        heads_count = heads_count+1
    
    #stores head of tanks
    for tank_name, tank in wn.tanks():
        special_heads[heads_count] = results.node['head'].loc[ts*mult_ts, tank_name]
        heads_count = heads_count+1
        
    special_heads_vec = -inc_mat.T[:,num_junc:num_nodes] @ (special_heads)

    return junc_demands, special_heads_vec

def build_K_star(wn, results, inc_mat, mult_ts, K, op, range_Q, mid_point):
    
    ts = wn.options.time.report_timestep
    num_pipes = wn.num_pipes
    num_links = wn.num_links

    K_star = np.zeros((num_links,1))
    K_int = np.zeros((num_links,1))
    Q_test = results.link['flowrate'] # .loc[ts*mult_ts,:]
    heads = results.node['head'] # .loc[ts*mult_ts,:]
    
    Q_op = [] 
    dH_op = []
    if mid_point == True:
        for i in range(num_pipes):
            Q_op.append(0.5*(np.max(Q_test.loc[:,wn.pipe_name_list[i]]) + np.min(Q_test.loc[:,wn.pipe_name_list[i]])))
            dH_op.append(np.sign(Q_op[i])*K[i]*abs(Q_op[i])**1.852)
    else:
        Q_op = Q_test.loc[op*ts, :]
        dH_op = - inc_mat.T @ heads.loc[op*ts, :]
        
    if range_Q == None:
        for i in range(num_pipes):
            K_s, K_i = one_point_linear(Q_op[i], dH_op[i], K[i])
            K_star[i] = (K_s)
            K_int[i] = (K_i)
    else:
        for i in range(num_pipes):
            K_s, K_i = two_point_linear(Q_op[i], dH_op[i], K[i], range_Q)
            K_star[i] = (K_s)
            K_int[i] = (K_i)
        
    return K_star, K_int

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    # if array[idx] < value:
    #     idx = idx + 1
    return array[idx], idx

def pump_func(x, a, b):
    return a - b * x**2

def piecewise_pump(p1, pump):
    
    if len(pump.get_pump_curve().points) == 1:
        A1, B1 = pump.get_pump_curve().points[0][0], pump.get_pump_curve().points[0][1]
        A2, B2 = 0, 4*B1/3
        A3, B3 = 2*A1, 0

    if len(pump.get_pump_curve().points) == 3:
        A1, B1 = pump.get_pump_curve().points[0][0], pump.get_pump_curve().points[0][1]
        A2, B2 = pump.get_pump_curve().points[1][0], pump.get_pump_curve().points[1][1]
        A3, B3 = pump.get_pump_curve().points[2][0], pump.get_pump_curve().points[2][1]
    
    xdata = [A1, A2, A3]
    ydata = [B1, B2, B3]
    popt, pcov = curve_fit(pump_func, xdata, ydata)
    A, B = popt[0], popt[1]
    
    brackets = np.linspace(0,np.sqrt(A/B),p1+1)
    H= A - B*brackets**2
    return brackets, H

def piecewise_pipe(p2, results, pipe, K, range_Qp = None):
    largest_q = np.max(results.link['flowrate'].loc[:,pipe])
    smallest_q = np.min(results.link['flowrate'].loc[:,pipe])
    q_bound = max(abs(largest_q), abs(smallest_q))
    if range_Qp != None:
        q_bound = q_bound*(1+range_Qp)
        
    brackets = np.linspace(-q_bound, q_bound, p2+1)
    if abs(brackets[1]-brackets[0]) < 10**(-6):
        q_bound = np.mean(np.mean(results.link['flowrate']))
        brackets = np.linspace(-q_bound, q_bound, p2+1)
    H = np.sign(brackets)*K*abs(brackets)**1.852
    plt_brackets = np.linspace(-1.5*q_bound, 1.5*q_bound, 100)
    
    return brackets, H

# In[] Store pump cuve points to be used in PWL function

def run_gurobi(inp_file, no_ts = 3, num_pipe_seg = 3, num_pump_seg = 5,
               RELAX = 1, pump_on = False, pump_off = False, gate_valves = [], 
               gv_on = False, gv_off = False,
               ob_fn = 'static', obj = True, penalty = False, mipval = 0.0001):
       
    p1 = num_pump_seg
    p2 = num_pipe_seg
    
    pipe_valves = gate_valves
    
    # Run hydraulic simulation and store results
    wn = wntr.network.WaterNetworkModel(inp_file)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    junc_names = wn.junction_name_list
    node_names = wn.node_name_list
    pipe_names = wn.pipe_name_list
    num_junc = wn.num_junctions
    num_pipes = wn.num_pipes
    num_res = len(wn.reservoir_name_list)
    num_tanks = len(wn.tank_name_list)
    num_pumps = len(wn.pump_name_list)
    num_valves = len(wn.valve_name_list)
    num_links = len(wn.link_name_list)
    
    # Time configuration
    x_val = len(results.node['head'].loc[:, junc_names[0]])
    time_step = wn.options.time.report_timestep
    mult_ts = 0
    report_time_step = wn.options.time.report_timestep
    if no_ts == None:
        no_ts = int(wn.options.time.duration/report_time_step)
    
    # Empty arrays to hold the optimization results
    Q_TS = np.zeros((num_pipes, no_ts))
    QP_TS = np.zeros((num_pumps, no_ts))
    QV_TS = np.zeros((num_valves, no_ts))
    H_TS = np.zeros((num_junc, no_ts))
    HPi_TS = np.zeros((num_pipes, no_ts))
    HP_TS = np.zeros((num_pumps, no_ts))
    HV_TS = np.zeros((num_valves, no_ts))
    HT_TS = np.zeros((num_tanks, no_ts+1))
    
    pump_x = []
    pump_y = []
    pipe_x = []
    pipe_y = []
    valve_x = []
    valve_y = []
    tank_elv = []
    valve_dict = {}
    
    A21, K, inc_mat, node_names, link_names = extract_params(wn)
    junc_demands, special_heads_vec = extract_d_h(wn, results, inc_mat, mult_ts)

    for _, pump in wn.pumps():
        x,y = piecewise_pump(p1, pump)
        pump_x.append(x)
        pump_y.append(y)
        
    for i in range(num_pipes):
        pipe = wn.pipe_name_list[i]
        x,y = piecewise_pipe(p2, results, pipe, K[i])
        pipe_x.append(x)
        pipe_y.append(y)
        
    for _, tank in wn.tanks():
        tank_elv.append(tank.elevation)
        
    for valve_name, valve in wn.valves():
        valve_dict[valve_name] = {'Start node': valve.start_node_name, 
                                  'End node': valve.end_node_name,
                                  'Setting': valve.setting,
                                  'Type': valve.valve_type,
                                  'Minor loss': valve.minor_loss,
                                  'Start node index': wn.junction_name_list.index(valve.start_node_name),
                                  'End node index': wn.junction_name_list.index(valve.end_node_name)}
        
    
    
    # In[] Extract control rules and store in a dictionary
    
    time_controls_dict = {}
    event_controls_dict = {}
    time_controls_compiled = {}
    events_controls_pairs = {}
    time_controls_link_list = []
    event_controls_link_list = []
    
    for i in range(len(wn.control_name_list)):
        pattern = "'(.*?)'" 
        cont = wn.get_control(wn.control_name_list[i])
        cont_str = str(cont)
        contr_list = cont_str.split()
        
        # time-based controls
        if 'TIME' in cont_str:
            link_name  = contr_list[7]      # link name
            sim_time = contr_list[4]                                            # time
            stat = contr_list[8]                          # status? setting?
            stat_val = contr_list[10]
                       
            ctrl_name = 'Control {}'.format(i)
            time_controls_dict[ctrl_name] = {'Link': link_name, 
                                             'Time': sim_time, 
                                             'Stat': stat, 
                                             'Stat val': stat_val}
            if link_name not in time_controls_link_list:
                time_controls_link_list.append(link_name)
            
        # event-based controls
        else:
            node_name = contr_list[2]
            link_name = contr_list[8]
            node_level = contr_list[5]
            sense = contr_list[4]
            stat = contr_list[9]
            stat_val = contr_list[11]
            
            event_controls_dict['Control {}'.format(i)] = {'Link': link_name, 
                                                           'Node': node_name, 
                                                           'Sense': sense, 
                                                           'Level': node_level, 
                                                           'Stat': stat, 
                                                           'Stat val': stat_val}
            if link_name not in event_controls_link_list:
               event_controls_link_list.append(link_name)
     

    # store all time controls in one dict
    for link in time_controls_link_list:
        times_list_stat = [1- int(results.link['status'].loc[0,link])]
        times_list_time = [0]
    
        for key in list(time_controls_dict.keys()):
            if time_controls_dict[key]['Link'] == link:
                dt_list = time_controls_dict[key]['Time'].split(":")
                dt_time = int(dt_list[0])*3600 + int(dt_list[1])*60 + int(dt_list[2])
                time_cont = float(dt_time)/report_time_step
                if time_cont <= no_ts:
                    if time_controls_dict[key]['Stat'] == 'status' or time_controls_dict[key]['Stat'] == 'STATUS':
                        if time_controls_dict[key]['Stat val'] == 'Open' or time_controls_dict[key]['Stat val'] == 'OPEN':  # check if other stat vals are possible
                            times_list_stat.append(0)
                        else:
                            times_list_stat.append(1)
                    times_list_time.append(int(time_cont))  
                    
        stat_array = np.zeros((no_ts))
                                                             
        for i in range(len(times_list_stat)-1):
            stat_array[times_list_time[i]:times_list_time[i+1]] = times_list_stat[i]
        stat_array[times_list_time[-1]:] = times_list_stat[-1]
        time_controls_compiled[link] = {'Status':stat_array}
    
    # store coupled event-based controls together 
    ev_pair_count = 0
    ev_pair_list = []
    
    above_list = ['>', 'ABOVE']
    below_list = ['<', 'BELOW']
    closed_list = ['Closed', 'CLOSED']
    open_list = ['OPEN', 'Opened', 'Open']
    
    for key in list(event_controls_dict.keys()):
        for key_2 in list(event_controls_dict.keys()):
            if key != key_2:
                
                dict1 = event_controls_dict[key]
                dict2 = event_controls_dict[key_2]
                
                link, node = dict1['Link'], dict1['Node']
                link_2, node_2 = dict2['Link'], dict2['Node']
            
                if link == link_2 and node == node_2:
                    if (link,node) not in ev_pair_list:
                        # add other possibilites? setting for each kind of valve
                        
                        if dict1['Stat'] == 'status' or dict1['Stat'] == 'STATUS':
                            
                            val1 = wn.get_node(node).elevation + float(dict1['Level'])
                            val2 = wn.get_node(node).elevation + float(dict2['Level'])
                            init = str(wn.get_link(link).initial_status) # results.link['status'].loc[0, link]
                            if init == 'Closed' or init :
                                init_stat =  1
                            if init == 'Open':
                                init_stat = 0
                            
                            if dict1['Sense'] in above_list and dict1['Stat val'] in closed_list:
                                events_controls_pairs[ev_pair_count] = {'Link': link,
                                                                        'Node': node,
                                                                        'Upper lim': val1,
                                                                        'Lower lim': val2,
                                                                        'Upper lim stat': 1, # this is the value that will go to y
                                                                        'Lower lim stat': 0,
                                                                        'Link initial status': init_stat} # this is the value that will go to y0
                                
                            if dict1['Sense'] in above_list and dict1['Stat val'] in open_list: #unlikely
                                events_controls_pairs[ev_pair_count] = {'Link': link,
                                                                        'Node': node,
                                                                        'Upper lim': val1,
                                                                        'Lower lim': val2,
                                                                        'Upper lim stat': 0, # this is the value that will go to y
                                                                        'Lower lim stat': 1,
                                                                        'Link initial status': init_stat} # this is the value that will go to y0
                                
                            if dict1['Sense'] in below_list and dict1['Stat val'] in closed_list: #unlikely
                                events_controls_pairs[ev_pair_count] = {'Link': link,
                                                                        'Node': node,
                                                                        'Upper lim': val2,
                                                                        'Lower lim': val1,
                                                                        'Upper lim stat': 0, # this is the value that will go to y
                                                                        'Lower lim stat': 1,
                                                                        'Link initial status': init_stat}
                                
                            if dict1['Sense'] in below_list and dict1['Stat val'] in open_list: 
                                events_controls_pairs[ev_pair_count] = {'Link': link,
                                                                        'Node': node,
                                                                        'Upper lim': val2,
                                                                        'Lower lim': val1,
                                                                        'Upper lim stat': 1, # this is the value that will go to y
                                                                        'Lower lim stat': 0,
                                                                        'Link initial status': init_stat}
                                
                        ev_pair_count += 1
                        ev_pair_list.append((link,node))
    # In[] compile lists of special links and their indices
    # pumps, tank inlet, valves, other links in controls
    
    special_link_list = []
    special_link_index_list = []
    unique_ctrl_link_list = []
    unique_ctrl_ind_list = []
    unique_pipe_valve_list = []
    pipe_valve_ind_list = []
    
    u_tank_count = 0
    u_tank_link_list = []
    u_tank_ind_list = []
    
    # pumps
    for pump_name,_ in wn.pumps():
        special_link_list.append(pump_name)
        special_link_index_list.append(link_names.index(pump_name))
        
    # tank inlet pipes
    for i in range(len(link_names)):
        for j in range(num_tanks):
            tank = wn.tank_name_list[j]
            link = wn.get_link(link_names[i])
            if link.start_node_name == tank or link.end_node_name == tank:
                u_tank_count += 1
                u_tank_link_list.append(link_names[i])
                u_tank_ind_list.append(i)
                if link_names[i] not in special_link_list:
                    special_link_list.append(link_names[i])
                    special_link_index_list.append(link_names.index(link_names[i]))
         
    # valves
    for valve_name,_ in wn.valves():
        if valve_name not in special_link_list:
            special_link_list.append(valve_name)
            special_link_index_list.append(link_names.index(valve_name))
    
    # time based controls
    for link in time_controls_link_list:
        if link not in special_link_list:
            special_link_list.append(link)
            special_link_index_list.append(link_names.index(link))
            unique_ctrl_link_list.append(link)
    
    # event based controls        
    for link in event_controls_link_list:
        if link not in special_link_list:
            special_link_list.append(link)
            special_link_index_list.append(link_names.index(link))
            unique_ctrl_link_list.append(link)
            unique_ctrl_ind_list.append(link_names.index(link))
            
    # pipe valves
    for link in pipe_valves:
        if link not in special_link_list:
            special_link_list.append(link)
            special_link_index_list.append(link_names.index(link))
            unique_pipe_valve_list.append(link)
            pipe_valve_ind_list.append(link_names.index(link))
    
    num_unique_ctrl_links = len(unique_ctrl_link_list)
    num_pipe_valves = len(unique_pipe_valve_list)
    
    # In[]
    # Build matrices 
    A12 = A21.T
    
    KA11 = np.zeros((num_links,num_links)) # not populated by K star
    
    #build energy conservation rhs
    energy_rhs = special_heads_vec         # no K int here!
    
    #build A_link submatrix      
    A_link = np.identity(num_links)
    for i in range(num_pipes, num_links):
        A_link[i,i] = -1
                   
    u_count = num_pumps + u_tank_count + num_valves + num_unique_ctrl_links + num_pipe_valves
    u_block = np.zeros((num_links, u_count))
    
    for i in range(num_pumps):
        j = num_pipes + i
        u_block[j][i] = 1           # we do this to avoid problems that might arise because of names (eg. there might be both a pipe 1 and a valve 1)
    
    for i in range(u_tank_count):
        j = u_tank_ind_list[i]
        u_block[j][num_pumps + i] = 1
                
    for i in range(num_valves):
        j = num_pipes + num_pumps + i
        u_block[j][num_pumps + u_tank_count + i] = 1
    
    for i in range(num_unique_ctrl_links):
        j = unique_ctrl_ind_list[i]
        u_block[j][num_pumps + u_tank_count + num_valves + i] = 1
        
    for i in range(num_pipe_valves):
        j = pipe_valve_ind_list[i]
        u_block[j][num_pumps + u_tank_count + num_valves + num_unique_ctrl_links] = 1
    
    #build w block matrix
    w_block = np.zeros((num_links, num_valves))
    for i in range(num_valves):
        j = num_pipes + num_pumps + i
        w_block[j][i] = -1
    
    # Empty arrays to hold the optimization results
    up_TS = np.zeros((num_pumps, no_ts))
    ut_TS = np.zeros((u_tank_count, no_ts))
    uv_TS = np.zeros((num_valves, no_ts))
    uu_TS = np.zeros((num_unique_ctrl_links, no_ts))
    upv_TS = np.zeros((num_pipe_valves, no_ts))
    
    #build tank volume continuity equation
    Vt = np.zeros((num_tanks, num_links))
    Htanks = np.zeros((num_tanks, 1))
    
    for j in range(len(wn.tank_name_list)):
        tank = wn.get_node(wn.tank_name_list[j])
        tank_ind = num_junc + num_res + j
        prev_tank_head = np.array(inc_mat[tank_ind,:])[np.newaxis] @ special_heads_vec
        diam = tank.diameter
        Htanks[j] = - prev_tank_head
        Vt[j,:] = np.array(inc_mat[tank_ind,:])[np.newaxis]*(time_step/(np.pi*(diam/2)**2)) #check sign
        HT_TS[j,0] = -prev_tank_head
        
    # build A submatrix
    A_sub = np.block([[KA11,        A12,                            A_link,                             u_block,                            w_block,                            np.zeros((num_links,num_tanks))],   # delta H = H1 - H2 + u ...
                      [A21,         np.zeros((num_junc,num_junc)),  np.zeros((num_junc,num_links)),     np.zeros((num_junc, u_count)),      np.zeros((num_junc, num_valves)),   np.zeros((num_junc, num_tanks))],   # sum of Q = d  
                      [-Vt,         np.zeros((num_tanks,num_junc)), np.zeros((num_tanks,num_links)),    np.zeros((num_tanks, u_count)),     np.zeros((num_tanks, num_valves)),  np.identity(num_tanks)       ]])    # tank storage
    # build b submatrix
    b_sub = np.vstack((energy_rhs, junc_demands, Htanks)) #check sign
    
    A = A_sub   # just in case there's only one time step
    b = b_sub
    
    # Build master matrices
    for i in range(1,no_ts):   
       
        # Update RHS vector
        mult_ts = i
          
        num_nodes = wn.num_nodes
        num_junc = wn.num_junctions
        num_pipes = wn.num_pipes
        
        A_dim0 = A.shape[0]
        A_dim1 = A.shape[1]
        
        special_heads = np.zeros((num_res,1))
        heads_count = 0
            
        junc_demands = np.zeros((num_junc,1))
        for j in range(num_junc):
            junc_demands[j] = results.node['demand'].loc[i*time_step, junc_names[j]] # update junctions
        
        #stores head of reservoirs
        for res_name, res in wn.reservoirs():
            special_heads[heads_count] = results.node['head'].loc[i*time_step, res_name]
            heads_count = heads_count+1
        
        special_heads_vec = - inc_mat.T[:, num_junc : num_junc + num_res] @ (special_heads)
    
        # build energy conservation rhs
        energy_rhs = special_heads_vec
              
        Vt = np.zeros((num_tanks, num_links))
        Htanks = np.zeros((num_tanks, 1))
        A_tank = inc_mat[num_nodes - num_tanks : num_nodes,:].T
        
        for j in range(len(wn.tank_name_list)):
            tank = wn.get_node(wn.tank_name_list[j])
            tank_ind = num_junc + num_res + j
            diam = tank.diameter
            Vt[j,:] = np.array(inc_mat[tank_ind,:])[np.newaxis]*(time_step/(np.pi*(diam/2)**2))
               
        
        # build A submatrix
        A_tank_block = np.vstack((A_tank, np.zeros((num_junc, num_tanks)), -np.identity(num_tanks)))
        A_sub1 = np.hstack((A_tank_block, A_sub))
        
        # build b submatrix
        b_sub1 = np.vstack((energy_rhs, junc_demands, np.zeros((num_tanks,1)))) 
         
        # define A and b 
        A = np.block([[A,                                                       np.zeros((A_dim0, A_sub1.shape[1] - num_tanks))],
                      [np.zeros((A_sub1.shape[0], A_dim1 - num_tanks)),         A_sub1]])
        
        b = np.vstack((b, b_sub1))
    
      
    # In[] Build Gurobi optimization problem
    
    # Build right hand side of matrix describing relationship between mass balance at nodes and head loss in links
    rhs = []
    for i in range(len(b)):
        rhs.append(b[i][0])
    rhs = np.array(rhs)
    
    no_cons = A.shape[1]        # should be (num_junc + 2 * num_links + num_tanks) * no_ts
    
    Q_pipes, Q_pumps, Q_valves = {}, {}, {}                                             # flowrates
    H_nodes, H_pipes, H_pumps, H_valves = {}, {}, {}, {}                                # heads and head differences
    u_pumps, u_tanks, u_valves, u_conts, u_pvalves, w_valves = {}, {}, {}, {}, {}, {}   # u slack values
    H_tanks = {}                                                                        # tank heads
    y_pumps, y_tanks, y_controls, y_pvalves = {}, {}, {}, {}                            # on/off variables
    z1_cont, z2_cont, z3_cont, z4_cont = {}, {}, {}, {}                                 # control rules binary variables
    z1_inlet, z2_inlet, z3_inlet, z4_inlet = {}, {}, {}, {}                             # tank inlet pipe binary variables
    v1, v2, v3 = {}, {}, {}                                                             # valve status binary variables              
    
    ##############################################################################################################
    ##############################################################################################################
    
    # 1. Here is where we introduce new variables
    upper_lim, lower_lim = {}, {}                                                       # upper and low tank trigger levels
    y_pumps_proxy, diff_y, consecutive_y = {}, {}, {}                                   # pump variables for objective function
    
    ##############################################################################################################  
    ##############################################################################################################
                                
    
    # Define bounds
    lbounds_qpipes, ubounds_qpipes = [], []
    lbounds_qpumps, ubounds_qpumps = [], []
    lbounds_qvalves, ubounds_qvalves = [], []
    lbounds_hnodes, ubounds_hnodes = [], []
    lbounds_hpipes, ubounds_hpipes = [], []
    lbounds_hpumps, ubounds_hpumps = [], []
    lbounds_hvalves, ubounds_hvalves = [], []
    lbounds_upumps, ubounds_upumps = [], []
    lbounds_utanks, ubounds_utanks = [], []
    lbounds_uvalves, ubounds_uvalves = [], []
    lbounds_ucont, ubounds_ucont = [], []
    lbounds_upvalves, ubounds_upvalves = [], []
    lbounds_wvalves, ubounds_wvalves = [], []
    lbounds_htanks, ubounds_htanks = [], []
        
    largest_q = max(np.max(results.link['flowrate']))
    smallest_q = min(np.min(results.link['flowrate']))
    q_bound = max(abs(largest_q), abs(smallest_q))
    h_bound = max(np.max(results.node['head']))
    
    for i in range(num_pipes):
        lbounds_qpipes.append(-2 * q_bound)
        ubounds_qpipes.append(2 * q_bound)
        
    for i in range(num_pumps):
        lbounds_qpumps.append(0)
        ubounds_qpumps.append(pump_x[i][len(pump_x[0])-1])
    
    for i in range(num_valves):
        lbounds_qvalves.append(0)
        ubounds_qvalves.append(q_bound)  
    
    for i in range(num_junc):
        lbounds_hnodes.append(wn.get_node(wn.junction_name_list[i]).elevation)  # forcing pressures to be >= 0
        ubounds_hnodes.append(2 * h_bound)
        
    for i in range(num_pipes):
        lbounds_hpipes.append(-2 * h_bound)
        ubounds_hpipes.append(2 * h_bound)    
        
    for i in range(num_pumps):
        lbounds_hpumps.append(0)
        ubounds_hpumps.append(pump_y[i][0]) 
        
    for i in range(num_valves):
        lbounds_hvalves.append(-2 * h_bound)
        ubounds_hvalves.append(2 * h_bound) 
    
    for i in range(num_pumps):
        lbounds_upumps.append(-h_bound)
        ubounds_upumps.append(h_bound)  
        
    for i in range(u_tank_count):
        lbounds_utanks.append(-h_bound)
        ubounds_utanks.append(h_bound)
        
    for i in range(num_valves):
        lbounds_uvalves.append(-h_bound)
        ubounds_uvalves.append(h_bound)  
        
    for i in range(num_unique_ctrl_links):
        lbounds_ucont.append(-h_bound)
        ubounds_ucont.append(h_bound)  
        
    for i in range(num_pipe_valves):
        lbounds_upvalves.append(-h_bound)
        ubounds_upvalves.append(h_bound) 
        
    for i in range(num_valves):
        lbounds_wvalves.append(0)
        ubounds_wvalves.append(h_bound)  
                                     
    for i in range(num_tanks):
        tank = wn.get_node(wn.tank_name_list[i])
        lbounds_htanks.append(tank.elevation + tank.min_level)
        ubounds_htanks.append(tank.elevation + tank.max_level)
        
    ##############################################################################################################
    ##############################################################################################################
    
    # 2. Here is where we define bounds for our new variables
    lbounds_upperlim, ubounds_upperlim = [], []
    lbounds_lowerlim, ubounds_lowerlim = [], []
    
    for j in range(ev_pair_count):                                                                          
        
        cont_dict = events_controls_pairs[j]       
        link = ev_pair_list[j][0]
        node = ev_pair_list[j][1]
        if node in wn.tank_name_list:   # have to write a whole other case if junc
            node_ind = wn.tank_name_list.index(node)
            tank = wn.get_node(node)
            
            lbounds_upperlim.append(tank.elevation + tank.min_level + 0.5 )         # adding a buffer so the problem is not too tight
            ubounds_upperlim.append(tank.elevation + tank.max_level - 0.5)              # adding a buffer so the problem is not too tight
            
            lbounds_lowerlim.append(tank.elevation + tank.min_level + 0.5 )         # adding a buffer so the problem is not too tight
            ubounds_lowerlim.append(tank.elevation + tank.max_level - 0.5)              # adding a buffer so the problem is not too tight
    
    ##############################################################################################################   
    ##############################################################################################################
        
        
    # Define some constants
    eps_flow = 0.000001 
    eps_head = 0.1 
    M = h_bound * 1.1 
    # M_q = q_bound * 1.1
    
        
    try:
        t1 = time.time()

        #####################################################################################################################################################################################
        # Create a new model
        m = gp.Model("Net1")
        
        m.reset()
        
        #####################################################################################################################################################################################
        # Create variables   
        for i in range(no_ts):
            Q_pipes[i] = m.addMVar(shape = num_pipes, lb=lbounds_qpipes, ub=ubounds_qpipes, vtype=GRB.CONTINUOUS, name="Q_pipes {}".format(i))
            
            if num_pumps > 0:
                Q_pumps[i] = m.addMVar(shape = num_pumps, lb=lbounds_qpumps, ub=ubounds_qpumps, vtype=GRB.CONTINUOUS, name="Q_pumps {}".format(i))
                
            if num_valves > 0:
                Q_valves[i] = m.addMVar(shape = num_valves, lb=lbounds_qvalves, ub=ubounds_qvalves, vtype=GRB.CONTINUOUS, name="Q_valves {}".format(i))
            
            H_nodes[i] = m.addMVar(shape = num_junc, lb=lbounds_hnodes, ub=ubounds_hnodes, vtype=GRB.CONTINUOUS, name="H_nodes {}".format(i))
            
            H_pipes[i] = m.addMVar(shape = num_pipes, lb=lbounds_hpipes, ub=ubounds_hpipes, vtype=GRB.CONTINUOUS, name="H_pipes {}".format(i))
            
            if num_pumps > 0:
                H_pumps[i] = m.addMVar(shape = num_pumps, lb=lbounds_hpumps, ub=ubounds_hpumps, vtype=GRB.CONTINUOUS, name="H_pumps {}".format(i))
                
            if num_valves > 0:
                H_valves[i] = m.addMVar(shape = num_valves, lb=lbounds_hvalves, ub=ubounds_hvalves, vtype=GRB.CONTINUOUS, name="H_valves {}".format(i))
                
            if num_pumps > 0:
                u_pumps[i] = m.addMVar(shape = num_pumps, lb=lbounds_upumps, ub=ubounds_upumps, vtype=GRB.CONTINUOUS, name="u_pump {}".format(i))
            
            u_tanks[i] = m.addMVar(shape = u_tank_count, lb=lbounds_utanks, ub=ubounds_utanks, vtype=GRB.CONTINUOUS, name="u_tanks {}".format(i))
            
            if num_valves > 0:
                u_valves[i] = m.addMVar(shape = num_valves, lb=lbounds_uvalves, ub=ubounds_uvalves, vtype=GRB.CONTINUOUS, name="u_valves {}".format(i))
            
            if num_unique_ctrl_links > 0:
                u_conts[i] = m.addMVar(shape = num_unique_ctrl_links, lb=lbounds_ucont, ub=ubounds_ucont, vtype=GRB.CONTINUOUS, name="u_conts {}".format(i))
                
            if num_pipe_valves > 0:
                u_pvalves[i] = m.addMVar(shape = num_pipe_valves, lb=lbounds_upvalves, ub=ubounds_upvalves, vtype=GRB.CONTINUOUS, name="u_pvalves {}".format(i))
            
            if num_valves > 0:            
                w_valves[i] = m.addMVar(shape = num_valves, lb=lbounds_wvalves, ub=ubounds_wvalves, vtype=GRB.CONTINUOUS, name="w_valves {}".format(i))
            
            H_tanks[i] = m.addMVar(shape = num_tanks, lb=lbounds_htanks, ub=ubounds_htanks, vtype=GRB.CONTINUOUS, name="H_tanks {}".format(i+1))
            
        
        #####################################################################################################################################################################################
        # Load all current variable into one vector x_A
        m.update()
    
        x_A = gp.MVar(m.getVars())
        
        m.update()
        
        #####################################################################################################################################################################################
        # MATRIX CONSTRAINTS (head loss, mass balance, tank storage)
        m.addConstr(A @ x_A == rhs, name = "block")   # for mass balance, head loss, tank storage constraints
    
        m.update()
        
        ##############################################################################################################
        ##############################################################################################################
        
        # 3. Here is where we introduce our new variables to the Gurobi optimization problem
        if ev_pair_count > 0:  
            upper_lim = m.addMVar(shape = ev_pair_count, lb=lbounds_upperlim, ub=ubounds_upperlim, vtype=GRB.CONTINUOUS, name="upper_lim_{}".format(i))
            lower_lim = m.addMVar(shape = ev_pair_count, lb=lbounds_lowerlim, ub=ubounds_lowerlim, vtype=GRB.CONTINUOUS, name="lower_lim_{}".format(i))
        
        ##############################################################################################################   
        ##############################################################################################################
        
        #####################################################################################################################################################################################
        # PWL PIPE HEAD LOSS CONSTRAINTS
        for i in range(no_ts):
            for j in range(num_pipes):
                m.addGenConstrPWL(Q_pipes[i][j], H_pipes[i][j], pipe_x[j], pipe_y[j])
        
        #####################################################################################################################################################################################
        # PWL PUMP HEAD GAIN CONSTRAINTS 
        if num_pumps > 0:
            for i in range(no_ts):
                for j in range(num_pumps):
                    m.addGenConstrPWL(Q_pumps[i][j], H_pumps[i][j], pump_x[j], pump_y[j])
        
        
        #####################################################################################################################################################################################
        # ALLOW PUMP TO DISCONNECT
        if num_pumps > 0:
            for i in range(no_ts):
                y_pumps[i] = m.addMVar(num_pumps, vtype = GRB.BINARY, name = "y_pumps {}".format(i))
             
            # Initial conditions for the pump and tank inlet pipe 
#            for i in range(num_pumps):
#                stat = str(wn.get_link(wn.pump_name_list[i]).initial_status)
##                print('stat', stat)
#                if stat == 'Open' or stat == 0:
#                    m.addConstr(y_pumps[0][i] == 0)
#                    m.addConstr(u_pumps[0][i] == 0)
#                else:
#                    m.addConstr(y_pumps[0][i] == 1)
#                    m.addConstr(Q_pumps[0][i] == 0)
            
            for i in range(no_ts):
                for j in range(num_pumps):
                    
                    # Define big M value
                    M = max(ubounds_upumps) + 5
                    m.addConstr(-Q_pumps[i][j] - M*(1-y_pumps[i][j]) <= 0)
                    m.addConstr(Q_pumps[i][j] - M*(1-y_pumps[i][j]) <= 0)
                    m.addConstr((y_pumps[i][j] == 0) >> (u_pumps[i][j] == 0))
                    m.addConstr((y_pumps[i][j] == 1) >> (Q_pumps[i][j] == 0))
                    
            ## forcing pump 1 to be open the whole time
            if pump_on == True:
                for i in range(no_ts):
                    m.addConstr(y_pumps[i][0] == 0)
                    
            ## forcing pump 1 to be open the whole time
            if pump_off == True:
                for i in range(no_ts):
                    m.addConstr(y_pumps[i][0] == 1)
            
        #####################################################################################################################################################################################    
        # TANK STATUS CHECK: FORCE TANK INLET PIPE TO CLOSE ONLY IF TANK LEVEL HITS MAX OR MIN
                
        for i in range(no_ts):
            y_tanks[i] = m.addMVar(u_tank_count, vtype = GRB.BINARY, name = "y tanks {}".format(i))
            z1_inlet[i] = m.addMVar(u_tank_count, vtype = GRB.BINARY, name = "z1_inlet {}".format(i))
            z2_inlet[i] = m.addMVar(u_tank_count, vtype = GRB.BINARY, name = "z2_inlet {}".format(i))
            z3_inlet[i] = m.addMVar(u_tank_count, vtype = GRB.BINARY, name = "z3_inlet {}".format(i))
            z4_inlet[i] = m.addMVar(u_tank_count, vtype = GRB.BINARY, name = "z4_inlet {}".format(i))
        
        for i in range(u_tank_count):
            m.addConstr(y_tanks[0][i] == 0) ####### edit this at some point! to reflect actual starting
            m.addConstr(u_tanks[0][i] == 0)
            m.addConstr(z1_inlet[0][i] == 0)
            m.addConstr(z2_inlet[0][i] == 1)
            m.addConstr(z3_inlet[0][i] == 1)
        
        for i in range(1,no_ts):
            for j in range(num_tanks):
                ind = u_tank_ind_list[j]
                
                up_bd = ubounds_htanks[j]
                lo_bd = lbounds_htanks[j]
                M_tank = up_bd - lo_bd + 500
                
                # I tank level greater than or equal to upper bound
                m.addConstr(H_tanks[i-1][j] >= up_bd - M_tank*(1-z1_inlet[i][j]))
                m.addConstr(H_tanks[i-1][j] + eps_head <= up_bd + M_tank*z1_inlet[i][j])
                
                # II tank level greater than lower lim
                m.addConstr(H_tanks[i-1][j] >= lo_bd - M_tank*(1-z2_inlet[i][j]) + eps_head)
                m.addConstr(H_tanks[i-1][j] <= lo_bd + M_tank*z2_inlet[i][j])
                m.addConstr((z2_inlet[i][j] == 1) >> (z3_inlet[i][j] == 0))    
                
                # III tank level between lower and upper lim
                m.addConstr(y_tanks[i][j] == gp.or_(z1_inlet[i][j],z3_inlet[i][j]), name = "III {}".format(i))                    # is this right?
    
                # IV putting it all together           
                m.addConstr(-Q_pipes[i][ind] - M_tank*(1-y_tanks[i][j]) <= 0)
                m.addConstr(Q_pipes[i][ind] - M_tank*(1-y_tanks[i][j]) <= 0)                                                        ##### try to change M val here
                m.addConstr((y_tanks[i][j] == 0) >> (u_tanks[i][j] == 0))
                m.addConstr((y_tanks[i][j] == 1) >> (Q_pipes[i][ind] == 0))
        
        #####################################################################################################################################################################################    
                
        # VALVE STATUS RULES
        if num_valves > 0:
            for i in range(no_ts):
                v1[i] = m.addMVar(num_valves, vtype = GRB.BINARY, name = "v1 {}".format(i))
                v2[i] = m.addMVar(num_valves, vtype = GRB.BINARY, name = "v2 {}".format(i))
                v3[i] = m.addMVar(num_valves, vtype = GRB.BINARY, name = "v3 {}".format(i))
            
            for i in range(no_ts):
                for j in range(num_valves):
                    valve_name = wn.valve_name_list[j]
                    
                    M = M_tank
                    
                    # PRV
                    if valve_dict[valve_name]['Type'] == 'PRV':
        
                        start_ind = valve_dict[valve_name]['Start node index']
                        end_ind = valve_dict[valve_name]['End node index']
                        Hset_start = wn.get_node(valve_dict[valve_name]['End node']).elevation + valve_dict[valve_name]['Setting']  
                        Hset_end = wn.get_node(valve_dict[valve_name]['End node']).elevation + valve_dict[valve_name]['Setting']
                          
                        #1
                        m.addConstr(v1[i][j] + v2[i][j] + v3[i][j] == 1, name = 'Valve status')
                        
                        #2
                        m.addConstr(-M*v3[i][j] <= u_valves[i][j])
                        m.addConstr(u_valves[i][j] <= M*v3[i][j])
                                                
                        #3
                        m.addConstr(H_nodes[i][start_ind] - H_nodes[i][end_ind] >= -M*v3[i][j])
                        #4
                        m.addConstr(H_nodes[i][start_ind] - H_nodes[i][end_ind] + eps_head* v3[i][j] <= M*(1-v3[i][j]))
                        
                        
                        #5
                        m.addConstr(Hset_start - H_nodes[i][start_ind] >= -M*(1-v2[i][j]) - M*v1[i][j] + eps_head*v2[i][j])   
                        #6
                        m.addConstr(Hset_start - H_nodes[i][start_ind] <= M*(1-v1[i][j]) + M*v2[i][j]) 
                                    
                        #7
                        m.addConstr(w_valves[i][j] <= M*v1[i][j] )
                        
                        #8
                        m.addConstr(Hset_end*v1[i][j] - M*v2[i][j] - M*v3[i][j] <= H_nodes[i][end_ind])               
                        m.addConstr(H_nodes[i][end_ind] <= Hset_end*v1[i][j] + M*v2[i][j] + M*v3[i][j])
                        
                       #9
                        m.addConstr((v1[i][j] == 1) >> (Q_valves[i][j] >= eps_flow))
                        m.addConstr((v1[i][j] == 1) >> (u_valves[i][j] == 0))
                        m.addConstr((v2[i][j] == 1) >> (Q_valves[i][j] >= eps_flow))
                        m.addConstr((v2[i][j] == 1) >> (u_valves[i][j] == 0))
                        m.addConstr((v3[i][j] == 1) >> (Q_valves[i][j] == 0))
                       
                        if valve_dict[valve_name]['Minor loss'] == 0:
                            m.addConstr((v2[i][j] == 1) >> (H_nodes[i][start_ind] == H_nodes[i][end_ind]))
                           
                       
                       # else: flesh this out later!
                       
        #####################################################################################################################################################################################
        # ALLOW GATE VALVE TO DISCONNECT
        if num_pipe_valves > 0:
            for i in range(no_ts):
                y_pvalves[i] = m.addMVar(num_pipe_valves, vtype = GRB.BINARY, name = "y_pvalve{}".format(i))
             
            # Initial conditions for the pump and tank inlet pipe 
            for i in range(num_pipe_valves):
                stat = str(wn.get_link(unique_pipe_valve_list[i]).initial_status)
#                print('Stat', stat)
                if stat == 'Open' or stat == 0:
                    m.addConstr(y_pvalves[0][i] == 0)
                    m.addConstr(u_pvalves[0][i] == 0)
                else:
                    m.addConstr(y_pvalves[0][i] == 1)
                    m.addConstr(Q_pipes[0][pipe_valve_ind_list[i]] == 0)
                    
            
            for i in range(no_ts):
                for j in range(num_pipe_valves):
                    
                     # Define big M value
                    M = max(ubounds_upvalves) + 5
                    ind = pipe_valve_ind_list[j]
                   
                    m.addConstr(-Q_pipes[i][ind] - M*(1-y_pvalves[i][j]) <= 0)
                    m.addConstr(Q_pipes[i][ind] - M*(1-y_pvalves[i][j]) <= 0)
                    m.addConstr((y_pvalves[i][j] == 0) >> (u_pvalves[i][j] == 0))
                    m.addConstr((y_pvalves[i][j] == 1) >> (Q_pipes[i][ind] == 0))
                    
            ## forcing pipe valve to be open the whole time
            if gv_on == True:
                for i in range(no_ts):
                    m.addConstr(y_pvalves[i][0] == 0)               
                    
            ## forcing pipe valve to be open the whole time
            if gv_off == True:
                for i in range(no_ts):
                    m.addConstr(y_pvalves[i][0] == 1)  
        
        #####################################################################################################################################################################################
        
        ##############################################################################################################
        ##############################################################################################################
        
        # 4. In this example pump scheduling optimization problem, we are setting tank trigger settings to be VARIABLE instead of CONSTANT. 
        # Therefore, we have to modify the EVENT-BASED CONTROL RULES constraints shown here.
        # We are looking only at pre-existing combinations of tank-pump pairs (for which control rules have previously been defined) 
        
        
        # EVENT_BASED CONTROL RULES
        for i in range(no_ts):
            z1_cont[i] = m.addMVar(ev_pair_count, vtype = GRB.BINARY, name = "z1_cont{}".format(i))
            z2_cont[i] = m.addMVar(ev_pair_count, vtype = GRB.BINARY, name = "z2_cont{}".format(i))
            z3_cont[i] = m.addMVar(ev_pair_count, vtype = GRB.BINARY, name = "z3_cont{}".format(i))
            z4_cont[i] = m.addMVar(ev_pair_count, vtype = GRB.BINARY, name = "z4_cont{}".format(i))
            y_controls[i] = m.addMVar(ev_pair_count, vtype = GRB.BINARY, name = "y_cont{}".format(i))
            
        for j in range(ev_pair_count):                                                                          
            
            cont_dict = events_controls_pairs[j]
    
            if cont_dict['Link initial status'] == 0:
                m.addConstr(y_controls[0][j] == 0)
            else:
                m.addConstr(y_controls[0][j] == 1)
            
            link = ev_pair_list[j][0]
            node = ev_pair_list[j][1]
            if node in wn.tank_name_list:   # have to write a whole other case if junc
                node_ind = wn.tank_name_list.index(node)
                
                upper_lim_net = cont_dict['Upper lim']                                                                          # note: we no longer rely on this upper limit
                lower_lim_net = cont_dict['Lower lim']                                                                          # note: we no longer rely on this lower limit
                M_cont = upper_lim_net - lower_lim_net + 1
                M_cont_final = 11
                eps_cont = 0.001
                                          
                # if node > x then close link, if node < y then open link
                if cont_dict['Upper lim stat'] == 1 and cont_dict['Lower lim stat'] == 0:
                    for i in range(1,no_ts):
                        # I tank level greater than or equal to upper lim
                        m.addConstr(H_tanks[i-1][node_ind] >= upper_lim[j] - M_cont*(1-z1_cont[i][j]))               # note: change upper_lim (a constant) to upper_lim[node_ind] (variable corresponding to tank)
                        m.addConstr(H_tanks[i-1][node_ind] + eps_cont <= upper_lim[j] + M_cont*z1_cont[i][j])        # note: change upper_lim (a constant) to upper_lim[node_ind] (variable corresponding to tank)
                        m.addConstr((z1_cont[i][j] == 0) >> (z2_cont[i][j] == 1)) 
                        m.addConstr((z1_cont[i][j] == 1) >> (z2_cont[i][j] == 0))      # check if this line is necessary, the fewer constraints we add the better        
                        
                        # II tank level greater than lower lim
                        m.addConstr(H_tanks[i-1][node_ind] >= lower_lim[j] - M_cont*(1-z3_cont[i][j]) + eps_cont)    # note: change lower_lim (a constant) to lower_lim[node_ind] (variable corresponding to tank)
                        m.addConstr(H_tanks[i-1][node_ind] <= lower_lim[j] + M_cont*z3_cont[i][j])                   # note: change lower_lim (a constant) to lower_lim[node_ind] (variable corresponding to tank)
                        
                        # III tank level between lower and upper lim
                        m.addConstr(z4_cont[i][j] == gp.and_(z2_cont[i][j],z3_cont[i][j]), name = "III {}".format(i))                    # is this right?
            
                        # IV putting it all together
                        m.addConstr(y_controls[i-1][j] + 2*z1_cont[i][j] + z4_cont[i][j] >= 2 - M_cont_final*(1-y_controls[i][j]), name = "IV a {}".format(i))
                        m.addConstr(y_controls[i-1][j] + 2*z1_cont[i][j] + z4_cont[i][j] + eps_cont <= 2 + M_cont_final*(y_controls[i][j]), name = "IV b {}".format(i))   
                 
                # if node > x then open link, if node < y then close link
                elif cont_dict['Upper lim stat'] == 0 and cont_dict['Lower lim stat'] == 1:
                                         
                    for i in range(1,no_ts):
                        # I tank level greater than or equal to upper lim
                        m.addConstr(H_tanks[i-1][node_ind] >= upper_lim[j] - M_cont*(1-z1_cont[i][j]))               # note: change upper_lim (a constant) to upper_lim[node_ind] (variable corresponding to tank)
                        m.addConstr(H_tanks[i-1][node_ind] + eps_cont <= upper_lim[j] + M_cont*z1_cont[i][j])        # note: change upper_lim (a constant) to upper_lim[node_ind] (variable corresponding to tank)
                        m.addConstr((z1_cont[i][j] == 0) >> (z2_cont[i][j] == 1)) 
                        m.addConstr((z1_cont[i][j] == 1) >> (z2_cont[i][j] == 0))      # check if this line is necessary, the fewer constraints we add the better        
                        
                        # II tank level greater than lower lim
                        m.addConstr(H_tanks[i-1][node_ind] >= lower_lim[j] - M_cont*(1-z3_cont[i][j]) + eps_cont)    # note: change lower_lim (a constant) to lower_lim[node_ind] (variable corresponding to tank)
                        m.addConstr(H_tanks[i-1][node_ind] <= lower_lim[j] + M_cont*z3_cont[i][j])                   # note: change lower_lim (a constant) to lower_lim[node_ind] (variable corresponding to tank)
                        
                        # III tank level between lower and upper lim
                        m.addConstr(z4_cont[i][j] == gp.and_(z2_cont[i][j],z3_cont[i][j]), name = "III {}".format(i))                    # is this right?
            
                        # IV putting it all together
                        m.addConstr(y_controls[i-1][j] + 2*z4_cont[i][j] + 3*(1-z3_cont[i][j]) >= 3 - M_cont_final*(1-y_controls[i][j]), name = "IV a {}".format(i))
                        m.addConstr(y_controls[i-1][j] + 2*z4_cont[i][j] + 3*(1-z3_cont[i][j]) + eps_cont <= 3 + M_cont_final*(y_controls[i][j]), name = "IV b {}".format(i))   
                    
                for i in range(no_ts):
                    # V translating y to u
                    if link in wn.pump_name_list:
                        link_ind = wn.pump_name_list.index(link)
#                        print(link_ind)
                        m.addConstr((y_controls[i][j] == 0) >> (y_pumps[i][link_ind] == 0))
                        m.addConstr((y_controls[i][j] == 1) >> (y_pumps[i][link_ind] == 1))
                        
                    elif link in u_tank_link_list:
                        inlet_ind = u_tank_link_list.index(link)
                        link_ind = u_tank_ind_list[inlet_ind]
                        m.addConstr((y_controls[i][j] == 0) >> (u_tanks[i][inlet_ind] ==0))
                        m.addConstr((y_controls[i][j] == 1) >> (Q_pipes[i][link_ind] == 0))
                        
                    elif link in wn.valve_name_list:
                        link_ind = wn.valve_name_list.index(link)
                        m.addConstr((y_controls[i][j] == 0) >> (u_valves[i][link_ind] == 0))    
                        m.addConstr((y_controls[i][j] == 1) >> (Q_valves[i][link_ind] == 0))
                        
                    elif link in unique_ctrl_link_list:
                        spl_ind = unique_ctrl_link_list.index(link)
                        link_ind = wn.link_name_list.index(link)
                        m.addConstr((y_controls[i][j] == 0) >> (u_conts[i][spl_ind] == 0))    
                        m.addConstr((y_controls[i][j] == 1) >> (Q_pipes[i][link_ind] == 0))
                        
        ##############################################################################################################   
        ##############################################################################################################
                        
        #####################################################################################################################################################################################    
        # TIME-BASED CONTROL RULES
        for link in list(time_controls_compiled.keys()):
            
            # pumps
            if link in wn.pump_name_list:
                ind = wn.pump_name_list.index(link)
                
                for i in range(no_ts):
                    stat = int(time_controls_compiled[link]['Status'][i])
                    m.addConstr(y_pumps[i][ind] == stat)
            
            # tank inlet pipes
            if link in u_tank_link_list:
                ind = u_tank_ind_list[u_tank_link_list.index(link)]
                
                for i in range(no_ts):
                    stat = time_controls_compiled[link]['Status'][i]
                    m.addConstr(y_tanks[i][ind] == stat)
                
              
        ##############################################################################################################
        ##############################################################################################################
        
        # 5. Add additional constraints and binary variables here
        
        # These  constraints ensure that the upper lim will always be > lower lim + arbitrary storage value
        
        for j in range(ev_pair_count):
            m.addConstr(upper_lim[j] >= lower_lim[j] + 5) # must be at least 5 m difference between upper and lower limit
        
        
        # The following constraints model pump triggers (turning on/off) and pump switches
        for i in range(no_ts):
             y_pumps_proxy[i] = m.addMVar(num_pumps, vtype = GRB.BINARY, name = "y_proxy_{}".format(i))
             if i != no_ts-1:
                 diff_y[i] = m.addMVar(num_pumps, vtype = GRB.BINARY, name = "y_diff_{}".format(i))
         
        m.addConstrs(y_pumps_proxy[i][j] == 1 - y_pumps[i][j] for i in range(no_ts) for j in range(num_pumps))
         
        m.addConstrs(diff_y[i][j] >= y_pumps[i+1][j] - y_pumps[i][j] for i in range(no_ts-1) for j in range(num_pumps))
        m.addConstrs(diff_y[i][j] >= y_pumps[i][j] - y_pumps[i+1][j] for i in range(no_ts-1) for j in range(num_pumps))    
        m.addConstrs(diff_y[i][j] <= y_pumps[i][j] + y_pumps[i+1][j] for i in range(no_ts-1) for j in range(num_pumps))    
        m.addConstrs(2 - diff_y[i][j] >= y_pumps[i][j] + y_pumps[i+1][j] for i in range(no_ts-1) for j in range(num_pumps)) 
               
        ##############################################################################################################   
        ############################################################################################################## 
       
        m.update()
         
        
        # In[] Objective function
        
        ##############################################################################################################
        ##############################################################################################################
        
        # 6. Set up objective function
        
        diurnals = []
        statics = []
        reverses = []
        penalties = []
        
        num_pumps_in_OF = wn.num_pumps
        
        for pump_name in wn.pump_name_list:
            # Calculate mean power at pump based on WNTR simulation results
            power = results.link['flowrate'].loc[:,pump_name]*-results.link['headloss'].loc[:,pump_name]
            mean_power = power[power>0].mean()*9.81*3600/1000
            
            # Determine prices for different times of day
            off_peak =  10*0.01870  
            mid_peak = 10*0.02877 
            on_peak = 10*0.05898
            
            # Build time-varying cost functions based on mean power and different price schemes
            diurnal_cost_function = [off_peak * mean_power] * 3 + [mid_peak * mean_power] * 4 + [on_peak * mean_power] * 2 + [mid_peak * mean_power] * 2 + [off_peak * mean_power] * 2
            
            reverse_diurnal_cost_function = [on_peak * mean_power] * 3 + [mid_peak * mean_power] * 4 + [off_peak * mean_power] * 2 + [mid_peak * mean_power] * 2 + [on_peak * mean_power] * 2
            
            # This will need some customization for different networks and purposes - calculate a penalty term to penalize pump switching on/off 
            penal = [2 * mean_power * np.mean([off_peak,mid_peak,on_peak])] * (no_ts-1)
            
            static_cost = np.mean(diurnal_cost_function) 
            
            static_cost_function = [static_cost] * no_ts
            
            diurnals.append(diurnal_cost_function)
            statics.append(static_cost_function)
            reverses.append(reverse_diurnal_cost_function)
            penalties.append(penal)
        
        if ob_fn == 'static':
            cost_function = statics
            
        if ob_fn == 'diurnal':
            cost_function = diurnals
       
        if ob_fn == 'reverse':
            cost_function = reverses 
        
        ##############################################################################################################
        ##############################################################################################################
        
        ####################################################################################################################################################################################
        # Assign objective function to optimization problem
        if obj == True:
            if penalty == True:
                print('true obj penalty')
                obj = gp.quicksum(cost_function[0][i] * y_pumps_proxy[i][j] for i in range(no_ts) for j in range(num_pumps_in_OF)) + gp.quicksum(penalties[0][i] * diff_y[i][j] for i in range(no_ts-1) for j in range(1))
        
            else:
                print('true obj')
                obj = gp.quicksum(cost_function[0][i] * y_pumps_proxy[i][j] for i in range(no_ts) for j in range(num_pumps_in_OF))
        
            m.setObjective(obj, GRB.MINIMIZE)
        
        else:
            print('no obj')
            obj = np.zeros((1,no_cons))
            m.setObjective(obj @ x_A, GRB.MINIMIZE)
              
           
        # Set MIP gap
        m.setParam('MIPGap', mipval)
        
        #####################################################################################################################################################################################
        
        m.update()
        
        # Final A matrix solve by gurobi
        A_final = m.getA().toarray()
        rhs_final  = m.getAttr('RHS',m.getConstrs())
        rhs_senses = m.getAttr('Sense', m.getConstrs())
        # obj_final  = m.getAttr('Obj', m.getVars())
        
        # Optimize model
        m.optimize()

#        m.write("case study lp.lp")
                
        # Print stats
        m.printStats()

                
        m.write("milpnet.lp")
            
   
        # Exit flag 
        exit_flag = m.Status
        print('Exit flag:', exit_flag)
        
        relax_vars = 0
        relax_cons = 0
                      
        # Infeasibility assessments
        if RELAX and exit_flag in [3,4]:
            print('_______________________________________________________________________')
            print('-----------------------------------------------------------------------')
            print('The model is infeasible; relaxing the constraints')
            print('-----------------------------------------------------------------------')
            orignumvars = m.NumVars
            m.feasRelaxS(0, False, False, True)
            m.optimize()
            status = m.Status
            
            relax_vars = m.NumVars
            relax_cons = m.NumConstrs
            
            print('\nSlack values:')
            slacks = m.getVars()[orignumvars:]
            for sv in slacks:
                if sv.X > 1e-6:
                    print('%s = %g' % (sv.VarName, sv.X))
            if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
                print('The relaxed model cannot be solved \
                        because it is infeasible or unbounded')
        
            
            if status != GRB.OPTIMAL:
                print('Optimization was stopped with status %d' % status)
                # solution vector
                x_tot = gp.MVar(m.getVars())
                # A_final = model.getA().toarray()
                x = x_tot.X
        
        # exit flag and computing IIS
        # if computeIIS:
        #     print('Exit flag: ' + str(exit_flag))
        #     if exit_flag == 3 or exit_flag == 4:
        #         m.computeIIS()
                
        #         for c in m.getConstrs():
        #             if c.IISConstr:
        #                 print('%s' % c.ConstrName)
        
        # print(x.X)
        print('Obj: %g' % m.ObjVal)
        
        print('Solution, reduced costs, ranges')
        for v in m.getVars():
            print('%s %g' % (v.VarName, v.X), v.obj, v.RC, round(v.SAObjUp),v.SAObjLow)
        print('\nShadow prices and ranges')
        for c in m.getConstrs():
            lhsVal = m.getRow(c).getValue()
            print(c.ConstrName, lhsVal, c.Pi, c.RHS, c.SARHSUp, c.SARHSLow)
        
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
                
    
    t2 = time.time()
    print('Run time:', t2-t1, 's')
    
    
    # Solutions
    x = gp.MVar(m.getVars())
    # A_final = model.getA().toarray()
    x_tot = x.X
    x_dict={}
    for v in m.getVars():
        x_dict[str(v.varName)] = v.X
        
    for i in range(no_ts):
        start = int(i*no_cons/no_ts)
        
        for j in range(num_pipes):          # Qpipes
            Q_TS[j, i] = x.X[start + j]
            
        if num_pumps > 0:
            for j in range(num_pumps):      # Qpumps
                QP_TS[j, i] = x.X[start + num_pipes + j]    
                
        if num_valves > 0:
            for j in range(num_valves):
                QV_TS[j, i] =  x.X[start + num_pipes + num_pumps + j]    
            
        for j in range(num_junc):           # Hnodes 
            H_TS[j, i] = x.X[start + num_links + j]
            
        for j in range(num_pipes):          # Hpipes
            HPi_TS[j, i] = x.X[start + num_links + num_junc + j] 
            
        if num_pumps > 0:
            for j in range(num_pumps):      # Qpumps
                HP_TS[j, i] = x.X[start + num_links + num_junc + num_pipes + j] 
                
        if num_valves > 0:
            for j in range(num_valves):
                HV_TS[j, i] = x.X[start + num_links + num_junc + num_pipes + num_pumps + j] 
                                      
        if num_pumps > 0:
            up_TS[j, i] = x.X[start + 2 * num_links + num_junc + j]
                
        for j in range(u_tank_count):
            ut_TS[j, i] = x.X[start + 2 * num_links + num_junc + num_pumps + j]
            
        if num_valves > 0:
            for j in range(num_valves):
                uv_TS[j, i] = x.X[start + 2 * num_links + num_junc + num_pumps + u_tank_count + j] 
                
        if num_unique_ctrl_links > 0:
            for j in range(num_unique_ctrl_links):
                uu_TS[j,i] = x.X[start + 2 * num_links + num_junc + num_pumps + u_tank_count + num_valves + j] 
                
        if num_pipe_valves > 0:
            for j in range(num_pipe_valves):
                upv_TS[j,i] = x.X[start + 2 * num_links + num_junc + num_pumps + u_tank_count + num_valves + num_unique_ctrl_links + j] 
                
        for j in range(num_tanks):          # Htanks
            HT_TS[j, i+1] = x.X[start + 2 * num_links + num_junc + num_pumps + u_tank_count + 2*num_valves + num_unique_ctrl_links + num_pipe_valves + j]  ## w_TS!!
        
        
    all_vars = m.getVars()
    values = m.getAttr("X", all_vars)
    names = m.getAttr("VarName", all_vars)
    y_pump = []
    y_diff = []
    y_proxy = []
    
    ##############################################################################################################
    ##############################################################################################################
    
    # 7. Extract trigger levels optimal trigger results from optimization problem
    
    upper_lim_res = []
    lower_lim_res = []
    
    for name, val in zip(names, values):
        if "upper_lim" in name:
            upper_lim_res.append((name,val))
        if "lower_lim" in name:
            lower_lim_res.append((name,val))
        
        if "y_pump" in name:
            y_pump.append((name,val))
        if "y_diff" in name:
            y_diff.append((name,val))
        if "y_proxy" in name:
            y_proxy.append((name,val))
            
    
    
    ##############################################################################################################
    ##############################################################################################################

    
    num_total_var = m.NumVars
    num_bin_var = m.NumBinVars
    num_pwl_vars = m.NumPWLObjVars
    num_lin_constr = m.NumConstrs
    num_gen_constr = m.NumGenConstrs
    num_sos = m.NumSOS
    obj = m.getObjective()
    
    dict_res = {'INP': inp_file,
                'Num lin constr': num_lin_constr,
                'Num gen constr': num_gen_constr,
                'Num total var': num_total_var,
                'Num bin var': num_bin_var,
                'Num pwl': num_pwl_vars,
                'Num SOS': num_sos,
                'Num relax var': relax_vars,
                'Num relax const': relax_cons, 
                'Reduction time': t2-t1,
                'Q_TS': Q_TS,
                'QP_TS': QP_TS,
                'QV_TS': QV_TS,
                'H_TS': H_TS,
                'HPi_TS': HPi_TS, 
                'HP_TS': HP_TS,
                'HV_TS': HV_TS,
                'HT_TS': HT_TS,
                'of': obj.getValue(),
                
                ##############################################################################################################
                ##############################################################################################################
                
                # 8. Save trigger level results in results dictionary
                'upper lim': upper_lim_res,
                'lower lim': lower_lim_res
                
                ##############################################################################################################
                ##############################################################################################################
                
                
                }
    
    return dict_res, ev_pair_list, events_controls_pairs

# In[] Run an optimization problem

t1 = time.time()

no_ts = 12
ob_fn = 'diurnal'

inp_file = 'Networks/ANET.inp'

penalty = False

wn = wntr.network.WaterNetworkModel(inp_file)
wn.options.time.report_timestep = 7200
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~diurnal~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Here we run an optimization problem with a "realistic" diurnal objective function
basic_res, ev_pair_list, ev_dict = run_gurobi(inp_file, no_ts, ob_fn = 'diurnal',  
                        num_pipe_seg =1, num_pump_seg = 5, 
                        obj = True, penalty = penalty)
QP_TS_diur = basic_res['QP_TS']
HT_TS_diur = basic_res['HT_TS']

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~static~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Here we run an optimization problem with a static, unchanging objective function 
basic_res2, _, _ = run_gurobi(inp_file, no_ts, ob_fn ='static',  
                       num_pipe_seg =1, num_pump_seg = 5, 
                       obj = True, penalty = penalty)
QP_TS_stat = basic_res2['QP_TS']
HT_TS_stat = basic_res2['HT_TS']

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~reverse~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Here we run an optimization problem with an unrealistic objective function
basic_res3, _, _ = run_gurobi(inp_file, no_ts, ob_fn = 'reverse',  
                        num_pipe_seg =1, num_pump_seg = 5, 
                        obj = True, penalty = penalty)
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


# In[] Plot tank heads

fig, ax = plt.subplots(3,wn.num_tanks,figsize = (11,11))
for i, tank_name in enumerate(wn.tank_name_list):
    tank = wn.get_node(tank_name)
    ax.flat[i].plot(2*x_plot_tank, HT_TS_diur[i].T, lw = lw, color = 'g')
    ax.flat[i].plot(2*x_plot_tank, results.node['head'].loc[:no_ts*7200, tank_name], lw = lw, color = 'r')
    ax.flat[i].set_ylabel('Tank level [m]', fontsize = fs)
    ax.flat[i].axhline(y = tank.elevation + tank.max_level, c = 'k')
    ax.flat[i].axhline(y = tank.elevation + tank.min_level, c = 'k')
    for j, pair in enumerate(ev_pair_list):
        if tank_name in pair:
            ax.flat[i].axhline(y = basic_res['upper lim'][j][1], c = 'g', linestyle = 'dashed', label = 'MILPNet')
            ax.flat[i].axhline(y = basic_res['lower lim'][j][1], c = 'g', linestyle = 'dashed')
            ax.flat[i].axhline(y = ev_dict[j]['Upper lim'], c = 'r', linestyle = 'dashed', label = 'Original')
            ax.flat[i].axhline(y = ev_dict[j]['Lower lim'], c = 'r', linestyle = 'dashed')
    if i == 0:
        ax.flat[i].legend()
            
for i, tank_name in enumerate(wn.tank_name_list):
    tank = wn.get_node(tank_name)
    ax.flat[i+wn.num_tanks].plot(2*x_plot_tank, HT_TS_stat[i].T, lw = lw, color = 'g')
    ax.flat[i+wn.num_tanks].plot(2*x_plot_tank, results.node['head'].loc[:no_ts*7200, tank_name], lw = lw, color = 'r')
    ax.flat[i+wn.num_tanks].set_ylabel('Tank level [m]', fontsize = fs)
    ax.flat[i+wn.num_tanks].axhline(y = tank.elevation + tank.max_level, c = 'k')
    ax.flat[i+wn.num_tanks].axhline(y = tank.elevation + tank.min_level, c = 'k')
    for j, pair in enumerate(ev_pair_list):
        if tank_name in pair:
            ax.flat[i+wn.num_tanks].axhline(y = basic_res2['upper lim'][j][1], c = 'g', linestyle = 'dashed')
            ax.flat[i+wn.num_tanks].axhline(y = basic_res2['lower lim'][j][1], c = 'g', linestyle = 'dashed')
            ax.flat[i+wn.num_tanks].axhline(y = ev_dict[j]['Upper lim'], c = 'r', linestyle = 'dashed', label = 'Original')
            ax.flat[i+wn.num_tanks].axhline(y = ev_dict[j]['Lower lim'], c = 'r', linestyle = 'dashed')

for i, tank_name in enumerate(wn.tank_name_list):
    tank = wn.get_node(tank_name)
    ax.flat[i+2*wn.num_tanks].plot(2*x_plot_tank, HT_TS_rev[i].T, lw = lw, color = 'g')
    ax.flat[i+2*wn.num_tanks].plot(2*x_plot_tank, results.node['head'].loc[:no_ts*7200, tank_name], lw = lw, color = 'r')
    ax.flat[i+2*wn.num_tanks].set_ylabel('Tank level [m]', fontsize = fs)
    ax.flat[i+2*wn.num_tanks].axhline(y = tank.elevation + tank.max_level, c = 'k')
    ax.flat[i+2*wn.num_tanks].axhline(y = tank.elevation + tank.min_level, c = 'k')
    for j, pair in enumerate(ev_pair_list):
        if tank_name in pair:
            ax.flat[i+2*wn.num_tanks].axhline(y = basic_res3['upper lim'][j][1], c = 'g', linestyle = 'dashed')
            ax.flat[i+2*wn.num_tanks].axhline(y = basic_res3['lower lim'][j][1], c = 'g', linestyle = 'dashed')
            ax.flat[i+2*wn.num_tanks].axhline(y = ev_dict[j]['Upper lim'], c = 'r', linestyle = 'dashed', label = 'Original')
            ax.flat[i+2*wn.num_tanks].axhline(y = ev_dict[j]['Lower lim'], c = 'r', linestyle = 'dashed')

# In[]

print('c1 OF: ', basic_res['of'])
print('c2 OF: ', basic_res2['of'])
print('c3 OF: ', basic_res3['of'])

print('c1 problem run time: ', basic_res['Reduction time'])
print('c2 problem run time: ', basic_res2['Reduction time'])
print('c3 problem run time: ', basic_res3['Reduction time'])

print(basic_res['upper lim'], basic_res['lower lim'])
print(basic_res2['upper lim'], basic_res2['lower lim'])
print(basic_res3['upper lim'], basic_res3['lower lim'])

