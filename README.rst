=========
MILPNet 
=========

.. image:: https://zenodo.org/badge/537493014.svg
   :target: https://zenodo.org/badge/latestdoi/537493014

MILPNet is a Mixed-Integer Linear Programming framework for water distribution system (WDS) optimization problems. MILPNet models the following aspects of WDSs:

1. system hydraulics (i.e., mass balance equations and energy conservation equations)
2. hydraulic devices (e.g., storage tanks, pumps, pressure-reducing valves (PRVs), and gate valves (GVs))
3. status checks (e.g., preventing flow into/out of a tank if it is full/empty)
4. event-based and time-based control rules

MILPNet extracts WDS network topology and system boundary conditions (e.g., reservoir heads, junction demands, initial tank level, etc.) from a .INP file of a WDS model using the Python package **WNTR**. The **Gurobi (v 9.5.1) Python API** is used to formulate and solve an extended period optimization problem for WDSs. 

Checking MILPNet modeling accuracy
----------------------------------

The file :code:`example1_set_up_milpnet.py` builds an optimization problem using the `run_gurobi` function with the following components:

decision variables: heads of junctions and tanks, flow rates of pipes, pumps, GVs, and PRVs, status of pumps, GVs, and tank-links (open/closed), status of PRVs (active/open/closed)

objective function: 0

constraints: system hydraulics, hydraulic devices, status checks, control rules

The non-linear pipe head loss equation (based on the Hazen-Williams equation) and pump head gain equations are approximated using piece-wise linear segments. 

MILPNet employs Gurobi's in-built piece-wise linearization, indicator, and AND/OR/XOR constraint functionalities to model constraints.

The file :code:`example1_apply_milpnet.py` takes in a .INP file of a WDS network model and the number of time steps in a simulation duration (other inputs to the `run_gurobi` function can be specified as well if desired) and solves the feasibility optimization problem. Plots comparing the MILPNet optimization results for junction pressure heads and flows in links to EPANET results (through WNTR) are displayed to highlight modeling accuracy.

Demonstrating application of MILPNet to a optimization problem
--------------------------------------------------------------

Here, we show an example of how MILPNet can be used to build and solve a pump scheduling optimization problem.  The optimization problems involve determining the operational status and flow rates supplied by the pumps to minimize the energy costs under different cost structures while subject to system hydraulics. The file :code:`example2_set_up_pump_scheduling.py` builds an optimization problem using the `run_gurobi` function with the following components:

decision variables: heads of junctions and tanks, flow rates of pipes, pumps, GVs, and PRVs, status of pumps, GVs, and tank-links (open/closed), status of PRVs (active/open/closed), pump switches (binary variables relating pump status between consecutive time steps *t* amd *t+1*)

objective function: minimize cost of pump operation and number of pump switches

constraints: system hydraulics, hydraulic devices, status checks, control rules

The file :code:`example2_solve_for_optimal_pump_scheduling.py` builds and solves the optimization problem for a modified version of network ANET/Net1. We generate plots displaying the MILPNet optimization results for pump flow rates under different cost structures.

The Networks folder includes 8 benchmark networks for testing and validation (+ 2 networks to demonstrate pump scheduling examples). The original network names, modifications, and sources are as follows:

.. list-table:: 
   :header-rows: 1

   * - Network
     - Original Name
     - Modification
     - Source
   * - ANET
     - Net1
     - None
     - `Rossman et al (2020)`_
   * - BNET
     - Net2
     - None
     -  `Rossman et al (1994)`_ 
   * - CNET
     - Net2
     - Added open PRV
     -  `Rossman et al (1994)`_ 
   * - DNET
     - Net2
     - Added active + closed PRV
     -  `Rossman et al (1994)`_ 
   * - ENET
     - CA1
     - None
     -  `Rossman and Boulos (1996)`_
   * - FNET
     - PA2
     - None
     -  `Vasconcelos et al. (1997)`_
   * - GNET
     - Net3
     - Created a reduced model using `MAGNets`_ (max_nodal_degree =2)    
     -  `Clark et al. (1995)`_
   * - HNET
     - ky6
     - Created a fully reduced model using `MAGNets`_ and replaced power pump with head pump  
     -  `Jolly et al. (2014)`_
   * - Net1_casestudy2
     - Net1
     - Removed control rules, modified demand pattern
     - `Rossman et al (2020)`_
 
.. _`Rossman et al (2020)`: https://cfpub.epa.gov/si/si_public_record_Report.cfm?dirEntryId=348882&Lab=CESER
.. _`Rossman et al (1994)`: https://ascelibrary.org/doi/abs/10.1061/(ASCE)0733-9372(1994)120:4(803)
.. _`Rossman and Boulos (1996)`: https://ascelibrary.org/doi/abs/10.1061/(ASCE)0733-9496(1996)122:2(137)
.. _`Vasconcelos et al. (1997)`: https://awwa.onlinelibrary.wiley.com/doi/full/10.1002/j.1551-8833.1997.tb08259.x
.. _`Clark et al. (1995)`: https://ascelibrary.org/doi/abs/10.1061/(ASCE)0733-9496(1995)121:6(423)
.. _`MAGNets`: https://ascelibrary.org/doi/full/10.1061/JWRMD5.WRENG-5486
.. _`Jolly et al. (2014)`: https://ascelibrary.org/doi/full/10.1061/%28ASCE%29WR.1943-5452.0000352

Cite Us
-------

To cite MILPNet, please use the following publication: `A Mixed-Integer Linear Programming Framework for Optimization of Water Network Operations Problems`_

.. _`A Mixed-Integer Linear Programming Framework for Optimization of Water Network Operations Problems`: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023WR034526

::

    @article{title={A Mixed-Integer Linear Programming Framework for Optimization of Water Network Operations Problems},
             author={Thomas, Meghna and Sela, Lina},
             journal={Water Resources Research},
             volume={60},
             number={2},
             pages={e2023WR034526},
             year={2024},
             publisher={Wiley Online Library}
             }

Contact
-------
Meghna Thomas - meghnathomas@utexas.edu

Lina Sela - linasela@utexas.edu
