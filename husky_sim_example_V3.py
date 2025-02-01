import os
import numpy as np
import pydot
import time
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

# Function to get the path relative to the script's directory
def get_relative_path(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, path))


# Start the visualizer and clean up previous instances
meshcat = StartMeshcat()
visualize = True # Bool to switch the viszualization and simulation
meshcat.Delete()
meshcat.DeleteAddedControls()

# Set the path to your robot model:
robot_path = get_relative_path("../../models/descriptions/robots//husky_description/urdf/husky.urdf")
scene_path = get_relative_path("../../models/objects&scenes/scenes/floor.sdf")


######################################################################################################
#                             ########Define PD+G Controller as a LeafSystem #######   
######################################################################################################
import math

class Controller(LeafSystem):
    def __init__(self, plant):
        super().__init__()

        #Creation of the arrays storing all the data for plotting
        self.init_data(self)

        # Declare input ports for desired and current states
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=21)
        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=7)

        self.step = 1
        # Store plant and context for dynamics calculations
        self.plant, self.plant_context_ad = plant, plant.CreateDefaultContext()

        # Declare discrete state and output port for control input (tau_u)
        state_index = self.DeclareDiscreteState(4)  # 4 state variables.
        self.DeclareStateOutputPort("tau_u", state_index)  # output: y=x.
        
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/1000,  # One millisecond time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.compute_tau_u) # Call the Update method defined below.
        
        
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/100,  # One millisecond time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.update_data)
    
    def init_data(self,plant):
        #Initialisation of the data for graphs 
        self.time = []
        self.tau_l_array = []
        self.tau_r_array = []
        self.x_dot = []
        self.y_dot = []
        self.w_z = []
        self.x_rel = []
        self.y_rel = []
        self.theta_array = []
        self.forward_error = []
        self.angular_error = []
        #Initialisation of the transformation matrices
        self.abs_to_rel = []
        self.rel_to_abs = []

        #Initialisation of the controller parameters
        self.Kp_ = [0.5]
        self.Kd_ = [0.6]

    def update_data(self, context, discrete_state):

        state = self._current_state_port.Eval(context)

        current_time = context.get_time()

        self.time.append(current_time)
        self.tau_l_array.append(np.round( self.tau_l, 2))
        self.tau_r_array.append(np.round( self.tau_r, 2))
        self.x_dot.append(np.round( self.rel_vel[0], 2))
        self.y_dot.append(np.round( self.rel_vel[1], 2))
        self.w_z.append(np.round( self.rel_vel[2], 2))
        self.theta_array.append(np.round( self.theta, 2))
        self.forward_error.append(self.x_error)
        self.angular_error.append(self.theta_error)
    def compute_tau_u(self, context, discrete_state):
        
        # Evaluate the input ports
        self.q_d = self._desired_state_port.Eval(context)
        self.q = self._current_state_port.Eval(context)

        #Computation of rotation angle
        self.theta = 2 * np.arctan2(self.q[3],self.q[0])
        
        #Computation of reltive to absolute trnaformation matrix
        self.rel_to_abs = np.array([
            [np.cos(self.theta), -np.sin(self.theta), 0],
            [np.sin(self.theta),np.cos(self.theta),0],
            [0,0,1]])
        
        #Computation of absolute to reltive tranformation matrix
        self.abs_to_rel = self.rel_to_abs.T
        
        #Necessary absolute velocities : X*, Y*, W_z
        self.abs_vel = np.array([
            [self.q[14]],
            [self.q[15]],
            [self.q[13]]
        ])
        #Necessary relative velocities : x*, x*, w_z
        self.rel_vel = self.abs_to_rel @ self.abs_vel

        #Error computation
        X_error = self.q_d[4] - self.q[4]
        Y_error = self.q_d[5] - self.q[5]

        self.theta_error = np.arctan2(Y_error, X_error) - self.theta
        self.x_error = X_error * np.cos(self.theta) + Y_error * np.sin(self.theta)

        x_dot_ref = 0.3 * self.x_error
        w_z_ref = 5 * self.theta_error - self.q[13]

        r = 0.165
        d = 0.613

        inv_kin_mat = np.array([
            [1/r, d/(2*r)],
            [1/r, -d/(2*r)],
        ])

        w_ref = inv_kin_mat @ np.array([[x_dot_ref],[w_z_ref]])

        self.tau_l = self.Kp_[0]*(w_ref[0] - self.q[17])
        self.tau_r = self.Kp_[0]*(w_ref[1] - self.q[18])
        self.tau = [self.tau_l, self.tau_r, self.tau_l, self.tau_r]
        """
        robot_rot_quaternion = self.q[0:4]
        robot_pos = self.q[4:7]
        robot_wheel_rot = self.q[7:11]
        robot_ang_velocity = self.q[11:14]
        robot_speed = self.q[14:17] #Absolute values
        robot_wheel_ang_velocity = self.q[17:21]
        """
        # Compute gravity forces for the current state
        self.plant_context_ad.SetDiscreteState(self.q)
        
        # Update the output port = state
        discrete_state.get_mutable_vector().SetFromVector(self.tau)


##################################################################################################
# Function to Create Simulation Scene
def create_sim_scene(sim_time_step):   
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    parser = Parser(plant)
    parser.AddModelsFromUrl("file://" + robot_path)
    parser.AddModelsFromUrl("file://" + scene_path)
    plant.Finalize()
    
    #Initial rotation angle(z axis) of the robot 
    init_angle_deg = -15; 
    rotation_angle = init_angle_deg/180*np.pi 

    # Set the initial position of the robot
    plant.SetDefaultPositions([np.cos(rotation_angle/2), 0.0, 0.0, np.sin(rotation_angle/2), 0, 0, 0.138, 0.0, 0.0, 0.0, 0.0]) #(quaternions), translations, wheel rotations
    
    # Add visualization to see the geometries in MeshCat
    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    # Add a PD controller to regulate the robot
    controller = builder.AddNamedSystem("PD controller", Controller(plant))
    
    # Create a constant source for desired positions
    despos_ = [0,0,0,0,10,10,0]
    des_pos = builder.AddNamedSystem("Desired position", ConstantVectorSource(despos_))
    
    # Connect systems: plant outputs to controller inputs, and vice versa
    builder.Connect(plant.get_state_output_port(), controller.GetInputPort("Current_state")) 
    builder.Connect(controller.GetOutputPort("tau_u"), plant.get_actuation_input_port())

    builder.Connect(des_pos.get_output_port(), controller.GetInputPort("Desired_state"))
    
    # Build and return the diagram
    diagram = builder.Build()
    return diagram, controller

# Create a function to run the simulation scene and save the block diagram:
def run_simulation(sim_time_step):
    diagram,controller = create_sim_scene(sim_time_step)
    simulator = Simulator(diagram)
    
    simulator.Initialize()
    time.sleep(5); #Add 5s delay so the simulation is fluid
    simulator.set_target_realtime_rate(1.)
    
    # Save the block diagram as an image file
    svg_data = diagram.GetGraphvizString(max_depth=2)
    graph = pydot.graph_from_dot_data(svg_data)[0]
    image_path = "figures/block_diagram_husky.png"  # Change this path as needed
    graph.write_png(image_path)
    print(f"Block diagram saved as: {image_path}")
    
    # Run simulation and record for replays in MeshCat
    meshcat.StartRecording()
    simulator.AdvanceTo(15.0)  # Adjust this time as needed
    meshcat.PublishRecording() #For the replay
    plotGraphs(controller)

def plotGraphs(controller):
    #Speed plots
    fig0,axes0=plt.subplots(2,2)
    fig0.canvas.set_window_title('Speeds')    
    axes0[0][0].plot(controller.time,controller.theta_array)
    axes0[0][0].set_title(f'theta')
    axes0[0][0].grid(which='both',linestyle='--')

    axes0[1][0].plot(controller.time,controller.x_dot)
    axes0[1][0].set_title(f'X')
    axes0[1][0].grid(which='both',linestyle='--')

    axes0[0][1].plot(controller.time,controller.y_dot)
    axes0[0][1].set_title(f'Y')
    axes0[0][1].grid(which='both',linestyle='--')

    axes0[1][1].plot(controller.time,controller.w_z)
    axes0[1][1].set_title(f'w_z')
    axes0[1][1].grid(which='both',linestyle='--')
    plt.subplots_adjust(hspace=0.5)

    fig1,axes1=plt.subplots(2,2)
    fig1.canvas.set_window_title('Torques') 

    axes1[0][0].plot(controller.time,controller.tau_l_array)
    axes1[0][0].set_title(f'Left torque')
    axes1[0][0].grid(which='both',linestyle='--')

    axes1[0][1].plot(controller.time,controller.tau_r_array)
    axes1[0][1].set_title(f'Right torque')
    axes1[0][1].grid(which='both',linestyle='--')

    axes1[1][0].plot(controller.time,controller.forward_error)
    axes1[1][0].set_title(f'Forward Error')
    axes1[1][0].grid(which='both',linestyle='--')


    axes1[1][1].plot(controller.time,controller.angular_error)
    axes1[1][1].set_title(f'Angular Error')
    axes1[1][1].grid(which='both',linestyle='--')
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Run the simulation with a specific time step. Try gradually increasing it!
run_simulation(sim_time_step=0.001)
