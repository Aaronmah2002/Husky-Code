from trajectoryERG import ExplicitReferenceGovernor
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
from first_controller import Controller
T = 0.2  # Prediction horizon
dt = 1e-2
trajInit_=[0,0,0,0,1,-1,0]
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


######################################################################################################
#                             ########Define PD+G Controller as a LeafSystem #######   
######################################################################################################

    ## from the doc with the class in it 

#Check to see the coherence of the original dynamics eq
#HH = np.linalg.pinv(np.transpose(G) @ M_mat @ G) @ np.transpose(G) @ (E_mat @ np.array([[10],[-10],[10],[-10]]) - M_mat @ G_dot @ eta - (C_mat + g_mat) )
##################################################################################################
# Function to Create Simulation Scene



num_positions=11
num_velocities=10

####################################
#Explicite reference governor      #
####################################

class ERG(LeafSystem):
    def __init__(self):
        super().__init__()  # Don't forget to initialize the base class.

        # self._q_port = self.DeclareVectorInputPort(name="q", size=9)
        # self._dq_port = self.DeclareVectorInputPort(name="dq", size=9)

        self._state_port = self.DeclareVectorInputPort(name="state", size=21)
        
        self._tau_port = self.DeclareVectorInputPort(name="tau", size=4)
        self._qr_port = self.DeclareVectorInputPort(name="q_r", size=7)
        
        # self._qv_port = self.DeclareVectorInputPort(name="q_v", size=9)
        self.DeclareVectorOutputPort(name="q_v_filtered", size=7, calc=self.refrence)
        
        self.q_v_n = trajInit_ 

        self.erg = ExplicitReferenceGovernor(
          robust_delta_tau_=0.01, kappa_tau_=1.0,
          robust_delta_q_=0.1, kappa_q_=1.0, robust_delta_dq_=0.1, kappa_dq_=7.0,
          robust_delta_dp_EE_=0.01, kappa_dp_EE_=7.0, kappa_terminal_energy_=7.5)
        self.first_update = True
        
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/100,  # One millisecond time step.
            offset_sec=1.0,  # The first event is at time zero.
            update=self.refrence) # Call the Update method defined below.  #
        
    def refrence(self, context, output):
        # Evaluate the input ports
        state = self._state_port.Eval(context)
        q = state[:num_positions]
        dq = state[num_positions:]
        tau = self._tau_port.Eval(context)
        q_r = self._qr_port.Eval(context)
        csv_writing_file("tau_erg",tau)


        
        

        if self.first_update:
            self.q_v_ = trajInit_  # or an appropriate initial value    
            self.first_update = False
        self.q_v_ = self.erg.get_qv(q, dq, tau, q_r, self.q_v_)
        csv_writing_file("q_v_erg",self.q_v_)
        
        # Write into the output vector.
        output.SetFromVector(self.q_v_)


        


##################################################################################################
# Function to Create Simulation Scene
def create_sim_scene(sim_time_step):   
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    parser = Parser(plant)
    parser.AddModelsFromUrl("file://" + robot_path)
    parser.AddModelsFromUrl("file://" + scene_path)
    plant.Finalize()
    plant_context = plant.CreateDefaultContext()
    
    #Initial rotation angle(z axis) of the robot 
    init_angle_deg = -45; 
    rotation_angle = init_angle_deg/180*np.pi 

    # Set the initial position of the robot
    plant.SetDefaultPositions([np.cos(rotation_angle/2), 0.0, 0.0, np.sin(rotation_angle/2), 0, 0, 0.138, 0.0, 0.0, 0.0, 0.0]) #(quaternions), translations, wheel rotations
    
    # Add visualization to see the geometries in MeshCat
    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    # Add a PD controller to regulate the robot
    controller = builder.AddNamedSystem("PD controller", Controller(plant, plant_context))

    #Add an ERG on the top of the system 
    erg=builder.AddNamedSystem("erg",ERG())
    
    # Create a constant source for desired positions
    desired_rotation_dedg = 0
    desired_rotation_angle = init_angle_deg/180*np.pi 

    despos_ = [np.cos(desired_rotation_angle/2), 0.0, 0.0, np.sin(desired_rotation_angle/2),10,-8,0]
    
    des_pos = builder.AddNamedSystem("Desired position", ConstantVectorSource(despos_))
    
    builder.Connect(plant.get_state_output_port(),erg.GetInputPort("state"))
    builder.Connect(controller.GetOutputPort("tau_u"),erg.GetInputPort("tau"))
    builder.Connect(des_pos.get_output_port(), erg.GetInputPort("q_r"))


    # Connect systems: plant outputs to controller inputs, and vice versa
    builder.Connect(plant.get_state_output_port(), controller.GetInputPort("Current_state")) 
    builder.Connect(controller.GetOutputPort("tau_u"), plant.get_actuation_input_port())
    builder.Connect(erg.GetOutputPort("q_v_filtered"),controller.GetInputPort("Desired_state"))
    

    
    
    # Build and return the diagram
    diagram = builder.Build()
    diagram.set_name("diagram")
    diagram_context = diagram.CreateDefaultContext()
    return diagram, diagram_context,controller
diagram,diagram_context,controller = create_sim_scene(sim_time_step=0.001)
# Create a function to run the simulation scene and save the block diagram:
def run_simulation(sim_time_step):
    diagram,diagram_context,controller = create_sim_scene(sim_time_step)
    simulator = Simulator(diagram,diagram_context)
    
    simulator.Initialize()
    time.sleep(5); #Add 5s delay so the simulation is fluid
    simulator.set_target_realtime_rate(1.)
    '''
    # Save the block diagram as an image file
    svg_data = diagram.GetGraphvizString(max_depth=2)
    graph = pydot.graph_from_dot_data(svg_data)[0]
    image_path = "figures/block_diagram_husky.png"  # Change this path as needed
    graph.write_png(image_path)
    
    print(f"Block diagram saved as: {image_path}")
    '''
    # Run simulation and record for replays in MeshCat
    meshcat.StartRecording()
    simulator.AdvanceTo(20.0)  # Adjust this time as needed
    meshcat.PublishRecording() #For the replay
    #plotGraphs(controller)

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
run_simulation(sim_time_step=0.001)


'''
12/03

issues with the code 

due to calc_dynamic --> the state doesn't update so not good computation of the trajectory prediction
'''