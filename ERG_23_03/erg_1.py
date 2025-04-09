from trajERG_obs import ExplicitReferenceGovernor
import os
import numpy as np
import pydot
import time
import csv
from scipy.spatial.transform import Rotation as R
from IPython.display import SVG, display
from pydrake.common import temp_directory
from pydrake.geometry import StartMeshcat
from pydrake.math import RotationMatrix, RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization
from pydrake.systems.framework import LeafSystem
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.all import Variable, MakeVectorVariable
from helper.dynamics import CalcRobotDynamics
import matplotlib.pyplot as plt
from pydrake.systems.primitives import LogVectorOutput
from pydrake.all import AbstractValue
from pydrake.all import ContactResults
from pydrake.systems.primitives import VectorLogSink

from dyncontroller import Controller
T = 0.2  # Prediction horizon
dt = 1e-2
trajInit_=[np.cos(0), 0.0, 0.0, np.sin(0), 0, 0, 0.138]
##reset files
files=["controller_q","q_v_erg","tau_erg","tau_pred","state_dyn","x_next","state_pred","navfield"]
for i in files:
    fil=i+".csv"
    
    with open (fil,mode="w",newline="") as file:
        pass

# Function to get the path relative to the script's directory
def csv_writing_file(name:str,vector):
    # Define file name
    filename = name+".csv"
    # Data to be written (list of lists)
    for i in range(vector.shape[0]):
        data=[vector]
        #print(data)
    
    # Open the file in write mode
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write each row
        writer.writerows(data)
    
def use_logger(logger,legend,simulator):
    log_x = logger.FindLog(simulator.get_context())
    time_data = log_x.sample_times()  # Get timestamps
    state_data = log_x.data()
    # Number of states
    num_states = state_data.shape[0]
    # Plot each state over time
    plt.figure(figsize=(10, 6))

    for i in range(num_states):
        plt.plot(time_data, state_data[i, :])#, label=legend[i])

    plt.xlabel("Time [s]")
    plt.ylabel("State Values")
    plt.title("State Evolution Over Time")
    plt.legend()
    plt.grid()
    plt.show()
def get_relative_path(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, path))


# Start the visualizer and clean up previous instances
meshcat = StartMeshcat()
visualize = False # Bool to switch the viszualization and simulation
meshcat.Delete()
meshcat.DeleteAddedControls()

# Set the path to your robot model:
robot_path = get_relative_path("../../models/descriptions/robots//husky_description/urdf/husky.urdf")
scene_path = get_relative_path("../../models/objects&scenes/scenes/floor.sdf")
box_object = get_relative_path("../../models/objects&scenes/objects/cylinder.sdf")


######################################################################################################
#                             ########Define PD+G Controller as a LeafSystem #######   
######################################################################################################




##################################################################################################
# Function to Create Simulation Scene



num_positions=11

####################################
#Explicite reference governor      #
####################################

class ERG(LeafSystem):
    def __init__(self):
        super().__init__()  # Don't forget to initialize the base class.

        self._state_port = self.DeclareVectorInputPort(name="state", size=21) # input is the husky state
        self._tau_port = self.DeclareVectorInputPort(name="tau", size=4)
        self._qr_port = self.DeclareVectorInputPort(name="q_r", size=7)
        self._osb_port = self.DeclareVectorInputPort(name = "obs_1", size=13)

        self.DeclareVectorOutputPort(name="q_v_filtered", size=7, calc=self.refrence)
        
        self.q_v_ = self.DeclareDiscreteState(7)  # Assuming q_v_ has 7 elements

        
        self.q_v_n = trajInit_ 
        self.erg = ExplicitReferenceGovernor(sim_time_step,  #put the deltas to 0 
          robust_delta_tau_=0.01, kappa_tau_=5/32.08,
          robust_delta_q_=0, kappa_q_=5/20, robust_delta_dq_=0, kappa_dq_=5/4,
          robust_delta_dp_EE_=0, kappa_dp_EE_=5/4, kappa_terminal_energy_=7.5)
        self.first_update = True

        
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/100,  # One millisecond time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.refrence) # Call the Update method defined below.  #
        
    def refrence(self, context, discrete_state):
        # Evaluate the input ports
        state = self._state_port.Eval(context)
        q = state[:num_positions]
        dq = state[num_positions:]
        tau = self._tau_port.Eval(context)
        q_r = self._qr_port.Eval(context)
        q_o = self._osb_port.Eval(context)

        if self.first_update:
            self.q_v_ = trajInit_  # or an appropriate initial value    
            self.first_update = False
        self.q_v_ = self.erg.get_qv(q, dq, tau, q_r, self.q_v_,q_o)
        csv_writing_file("q_v_erg",self.q_v_)
        csv_writing_file("tau_erg",tau)
        
        # Write into the output vector.
        discrete_state.set_value(self.q_v_)


        


##################################################################################################
# Function to Create Simulation Scene
def create_sim_scene(sim_time_step):   
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    parser = Parser(plant)
    parser.AddModelsFromUrl("file://" + robot_path)
    parser.AddModelsFromUrl("file://" + scene_path)
    parser.AddModelsFromUrl("file://" + box_object)
    plant.Finalize()
    plant_context = plant.CreateDefaultContext()
    
    #Initial rotation angle(z axis) of the robot 
    init_angle_deg = -45; 
    rotation_angle = init_angle_deg/180*np.pi
    
    ## --> print(plant.GetStateNames())
    #new state have the form ;
    # [qw,qx,qy,qz,x,y,z,front_left,front_right,rear_left,rear_right,cyllinder_qw, cyllinder_qx, cyllinder_qy, cyllinder_qz, 
    # cylinder_x, cylinder_y, cylinder_z, 10 velocities, cylinder_wx, cylinder_wy cylinder_wz, cylinder_vx, cylinder_vz, cylinder_vz]



    # Set the initial position of the robot
    plant.SetDefaultPositions([np.cos(rotation_angle/2), 0.0, 0.0, np.sin(rotation_angle/2), 0, 0, 0.138, 0.0, 0.0, 0.0, 0.0, np.cos(rotation_angle/2), 0.0, 0.0, np.sin(rotation_angle/2), 2, -2, 0.4]) #(quaternions), translations, wheel rotations
    

    # Add visualization to see the geometries in MeshCat
    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    # Add a PD controller to regulate the robot
    controller = builder.AddNamedSystem("PD controller", Controller(plant, plant_context))

    #Add an ERG on the top of the system 
    erg=builder.AddNamedSystem("erg",ERG())
    
    # Create a constant source for desired positions
    desired_rotation_dedg = 0
    desired_rotation_angle = init_angle_deg/180*np.pi 


    obstacle_model_instance = plant.GetModelInstanceByName("cylinder")
    obstacle_state_port = plant.get_state_output_port(obstacle_model_instance) #dimension 13


    despos_ = [np.cos(desired_rotation_angle/2), 0.0, 0.0, np.sin(desired_rotation_angle/2),3,-3.5,0]
    des_pos = builder.AddNamedSystem("Desired position", ConstantVectorSource(despos_))
    
    builder.Connect(plant.GetOutputPort("cylinder_state"),erg.GetInputPort("obs_1"))
    builder.Connect(plant.GetOutputPort("husky_state"),erg.GetInputPort("state"))
    builder.Connect(controller.GetOutputPort("tau_u"),erg.GetInputPort("tau"))
    builder.Connect(des_pos.get_output_port(), erg.GetInputPort("q_r"))


    # Connect systems: plant outputs to controller inputs, and vice versa
    builder.Connect(plant.GetOutputPort("husky_state"), controller.GetInputPort("Current_state")) 
    builder.Connect(controller.GetOutputPort("tau_u"), plant.get_actuation_input_port())
    builder.Connect(erg.GetOutputPort("q_v_filtered"),controller.GetInputPort("Desired_state"))
    


    #builder.Connect(plant.get_contact_results_output_port(), controller.GetInputPort("contact_results"))
    


    logger_x = LogVectorOutput(erg.GetOutputPort("q_v_filtered"), builder) #state
    logger_tau = LogVectorOutput(controller.GetOutputPort("tau_u"),builder)
    logger_q = LogVectorOutput(plant.get_state_output_port(),builder)
    logger_husky = LogVectorOutput(plant.GetOutputPort("husky_state"),builder)
    logger_cylinder = LogVectorOutput(plant.GetOutputPort("cylinder_state"),builder)

    

    

    
    # Build and return the diagram
    diagram = builder.Build()
    diagram.set_name("diagram")
    diagram_context = diagram.CreateDefaultContext()
    return diagram, diagram_context,logger_x, logger_tau,erg,logger_q,despos_,logger_husky, logger_cylinder

# Create a function to run the simulation scene and save the block diagram:
def run_simulation(sim_time_step):
    diagram,diagram_context,logger_q_v_,logger_tau,erg,logger_q, despos_,logger_husky, logger_cylinder = create_sim_scene(sim_time_step)
    simulator = Simulator(diagram,diagram_context)
    simulator.Initialize()
    #time.sleep(5); #Add 5s delay so the simulation is fluid
    simulator.set_target_realtime_rate(1.)
    
    # Save the block diagram as an image file
    svg_data = diagram.GetGraphvizString(max_depth=2)
    graph = pydot.graph_from_dot_data(svg_data)[0]
    image_path = "block_diagram_husky.png"  # Change this path as needed
    graph.write_png(image_path)
    
    print(f"Block diagram saved as: {image_path}")
    
    # Run simulation and record for replays in MeshCat
    meshcat.StartRecording()
    simulator.AdvanceTo(5)  # Adjust this time as needed
    meshcat.PublishRecording() #For the replay
    


    size=len(erg.erg.DSM_tau_list)
    DSM_tau_list=np.array(erg.erg.DSM_tau_list)
    DSM_q_list=np.array(erg.erg.DSM_q_list)
    DSM_dq_list = np.array(erg.erg.DSM_qd_list)
    DSM_=np.array(erg.erg.DSM_list)
    navfield = np.array(erg.erg.navfield_list)
    rep_field = np.array(erg.erg.rep_field_list)
    att_field= np.array(erg.erg.att_field_list)
    
    vec_x=np.arange(0,size,1)
    #plotGraphs(controller)

    state_labels = [f"Q_scalar",f"Q_x",f"Q_y",f"Q_z",f"X",f"Y",f"Z"]  # Example labels
    
    
    
    log_qv = logger_q_v_.FindLog(simulator.get_context())
    log_q = logger_q.FindLog(simulator.get_context())

    time = log_qv.sample_times()
    log_data_qv_x = log_qv.data()[4,:]
    log_data_q_x = log_q.data()[4,:]
    log_data_qv_y = log_qv.data()[5,:]
    log_data_q_y = log_q.data()[5,:]
    log_data_q_dot_x = log_q.data()[7+14,:]
    log_data_q_dot_y = log_q.data()[7+15,:]
    log_cylinder_x = log_q.data()[15,:]
    log_cylinder_y = log_q.data()[16,:]
    # use_logger(logger=logger_tau,simulator=simulator,legend=[f"tau_l",f"tau_r",f"tau_l",f"tau_r"])
    # use_logger(logger=logger_husky,simulator=simulator, legend=['salut'])
    # use_logger(logger=logger_cylinder,simulator=simulator, legend=['salut'])

    # plt.figure()
    # plt.plot(time,log_cylinder_x,label='cylinder_x')
    # plt.plot(time, log_cylinder_y,label='cylinder_y')
    # plt.legend()
    
    fig0,axes0=plt.subplots(4,1,figsize=(8,6))
    axes0[0].plot(vec_x,DSM_tau_list)
    axes0[0].set_title("DSM_tau")
    axes0[1].plot(vec_x,DSM_q_list)
    axes0[1].set_title("DSM_q")
    axes0[2].plot(vec_x,DSM_dq_list)
    axes0[2].set_title("DSM_qd")
    axes0[3].plot(vec_x,DSM_,'k--',label = 'DSM',linewidth=3)
    axes0[3].set_title("DSM")
    axes0[3].plot(vec_x,DSM_tau_list,'r',label = 'DSM_tau')
    axes0[3].plot(vec_x,DSM_q_list,'g', label = 'DSM position')
    #axes0[3].plot(vec_x,DSM_dq_list,'y',label = 'DSM velocity')
    axes0[3].legend()
    
    plt.tight_layout() 
    
    # fig1,axes1=plt.subplots(2,2)
    # constant_value = np.zeros_like(time)
    # constant_value[time >= 0] = despos_[4]  # Change to the x desired position after t=0
    # axes1[0][0].plot(time,log_data_qv_x,label = 'q_v ERG')
    # axes1[0][0].plot(time,log_data_q_x,label = 'robot position')
    # axes1[0][0].step(time,constant_value,where='post', color='r', linestyle='--', label='reference')
    # axes1[0][0].legend()
    # axes1[0][0].set_xlabel('time [s]')
    # axes1[0][0].set_title('Position comparaison')


    # constant_value_y = np.zeros_like(time)
    # constant_value_y[time >= 0] = despos_[5]  # Change to the x desired position after t=0
    # axes1[0][1].plot(time,log_data_qv_y,label = 'q_v ERG')
    # axes1[0][1].plot(time,log_data_q_y,label = 'robot position')
    # axes1[0][1].step(time,constant_value_y,where='post', color='r', linestyle='--', label='reference')
    # axes1[0][1].legend()
    # axes1[0][1].set_xlabel('time [s]')
    # axes1[0][1].set_title('Position comparaison')

    # axes1[1][0].set_title('Velocity of the robot in world frame')
    # axes1[1][0].plot(time,log_data_q_dot_x,label = 'V_x')
    # axes1[1][0].legend() 
    # axes1[1][0].set_xlabel('time [s]')

    # axes1[1][1].set_title('Velocity of the robot in world frame')
    # axes1[1][1].plot(time,log_data_q_dot_y,label = 'V_y')
    # axes1[1][1].legend() 
    # axes1[1][1].set_xlabel('time [s]')
    
    fig,axes3=plt.subplots(2,1,figsize=(8,6))
    axes3[0].plot(vec_x,navfield[:,4],label= "rho")
    axes3[0].plot(vec_x,rep_field[:,4],label= "rho_rep")
    axes3[0].plot(vec_x,att_field[:,4],label= "rho_att")
    axes3[0].legend()
    axes3[1].plot(time,log_data_q_x, label ='x_position')
    plt.legend()

    
    plt.show()
def plotGraphs(controller):
    #Speed plots
    fig0,axes0=plt.subplots(2,2)
    fig0.canvas.set_window_title('Speeds')    
    axes0[0][0].plot(controller.time_array, controller.Theta_array)
    axes0[0][0].set_title(f'theta')
    axes0[0][0].grid(which='both',linestyle='--')

    axes0[1][0].plot(controller.time_array, controller.x_dot_array)
    axes0[1][0].plot(controller.time_array, controller.x_dot_ref_array)
    axes0[1][0].set_title(f'X')
    axes0[1][0].grid(which='both',linestyle='--')

    axes0[0][1].plot(controller.time_array, controller.y_dot_array)
    axes0[0][1].set_title(f'Y')
    axes0[0][1].grid(which='both',linestyle='--')

    axes0[1][1].plot(controller.time_array, controller.w_z_array)
    axes0[1][1].plot(controller.time_array, controller.w_z_ref_array)
    axes0[1][1].set_title(f'w_z')
    axes0[1][1].grid(which='both',linestyle='--')
    plt.subplots_adjust(hspace=0.5)

    fig1,axes1=plt.subplots(2,2)
    fig1.canvas.set_window_title('Torques') 

    axes1[0][0].plot(controller.time_array, controller.tau_l_array)
    axes1[0][0].set_title(f'Left torque')
    axes1[0][0].grid(which='both',linestyle='--')

    axes1[0][1].plot(controller.time_array, controller.tau_r_array)
    axes1[0][1].set_title(f'Right torque')
    axes1[0][1].grid(which='both',linestyle='--')

    axes1[1][0].plot(controller.time_array,controller.x_error_array)
    axes1[1][0].set_title(f'Forward Error')
    axes1[1][0].grid(which='both',linestyle='--')


    axes1[1][1].plot(controller.time_array,controller.y_error_array)
    axes1[1][1].set_title(f'Lateral Error')
    axes1[1][1].grid(which='both',linestyle='--')
    plt.subplots_adjust(hspace=0.5)
    """
    fig2,axes2=plt.subplots(1,1)
    fig2.canvas.set_window_title('Position') 
    axes1[0][0].plot(controller.Y_pos_array,controller.Y_pos_array)
    axes1[0][0].set_title(f'Position')
    axes1[0][0].grid(which='both',linestyle='--')
    """
    plt.subplots_adjust(hspace=0.5)
    plt.figure()
    plt.plot(controller.X_pos_array,controller.Y_pos_array)
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')

    plt.show()


# Run the simulation with a specific time step. Try gradually increasing it!
sim_time_step=0.001
run_simulation(sim_time_step)


'''
12/03

issues with the code 

due to calc_dynamic --> the state doesn't update so not good computation of the trajectory prediction
'''
