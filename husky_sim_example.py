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

        # Declare input ports for desired and current states
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=21)
        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=7)
        #desired state : quaternion + position
        self.Kp_ = [0.1,25]
        self.Kd_ = [0.1,5]
        self.step = 1
        # Store plant and context for dynamics calculations
        self.plant, self.plant_context_ad = plant, plant.CreateDefaultContext()
        
        #Print Time for printing with intervals
        self.last_print_time = 0.0

        # Declare discrete state and output port for control input (tau_u)
        state_index = self.DeclareDiscreteState(4)  # 4 state variables.
        self.DeclareStateOutputPort("tau_u", state_index)  # output: y=x.
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/1000,  # One millisecond time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.compute_tau_u) # Call the Update method defined below.
        
    
    def compute_tau_u(self, context, discrete_state):
        
        # Evaluate the input ports
        self.q_d = self._desired_state_port.Eval(context)
        self.q = self._current_state_port.Eval(context)

        robot_rot_quaternion = self.q[0:4]
        robot_pos = self.q[4:7]
        robot_wheel_rot = self.q[7:11]
        robot_ang_velocity = self.q[11:14]
        robot_speed = self.q[14:17] #Absolute values
        robot_wheel_ang_velocity = self.q[17:21]
        
        
        #Computation of rotational error
        rot_err =  np.arctan2((self.q_d[5] - self.q[5]) , (self.q_d[4] - self.q[4])) - np.arcsin(self.q[3])
        abs_ang_velocity = self.q[14]
        W = self.Kp_[1] * rot_err 
        
        #Computation of forward error
        dist_err = (self.q_d[4]-self.q[4])*np.cos(rot_err)
        abs_speed = (self.q[14]**2+self.q[15]**2)**(1/2)
        V = self.Kp_[0] * dist_err - self.Kd_[0] * abs_speed
	#Stops the torque when we are near the point but thi doen't relly work because of inertia
        
        V_1 = np.clip((V-W)/2, -15, 15)
        V_2 = np.clip((W+V)/2,-15,15)

        tau = [V_1, V_2, V_1,V_2]
        # Compute gravity forces for the current state
        self.plant_context_ad.SetDiscreteState(self.q)
        
        # Update the output port = state
        discrete_state.get_mutable_vector().SetFromVector(tau)
        #print state variables each 100ms
        currentPlantTime = context.get_time()
        print_time_step = 0.1; #time between each print for the states
        if(currentPlantTime - self.last_print_time > print_time_step):
            self.last_print_time = currentPlantTime
            print(tau)
            #print(currentPlantTime) to print the time
##################################################################################################
# Function to Create Simulation Scene
def create_sim_scene(sim_time_step):   
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    parser = Parser(plant)
    parser.AddModelsFromUrl("file://" + robot_path)
    parser.AddModelsFromUrl("file://" + scene_path)
    plant.Finalize()
    
    #rotation angle in degrees
    rotation_angle = 45/180*3.14 
    # Set the initial position of the robot
    plant.SetDefaultPositions([np.cos(rotation_angle/2), 0.0, 0.0, np.sin(rotation_angle/2), 0, 0, 0.138, 0.0, 0.0, 0.0, 0.0])
    #(quaternions), translations, wheel rotations
    
    # Add visualization to see the geometries in MeshCat
    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    # Add a PD controller to regulate the robot
    controller = builder.AddNamedSystem("PD controller", Controller(plant))
    
    # Create a constant source for desired positions
    despos_ = [0,0,0,1,1,1,0]
    des_pos = builder.AddNamedSystem("Desired position", ConstantVectorSource(despos_))
    
    # Connect systems: plant outputs to controller inputs, and vice versa
    builder.Connect(plant.get_state_output_port(), controller.GetInputPort("Current_state")) 
    builder.Connect(controller.GetOutputPort("tau_u"), plant.get_actuation_input_port())

    builder.Connect(des_pos.get_output_port(), controller.GetInputPort("Desired_state"))
    
    # Build and return the diagram
    diagram = builder.Build()
    return diagram

# Create a function to run the simulation scene and save the block diagram:
def run_simulation(sim_time_step):
    diagram = create_sim_scene(sim_time_step)
    
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
    simulator.AdvanceTo(40.0)  # Adjust this time as needed
    meshcat.PublishRecording() #For the replay
    
  

# Run the simulation with a specific time step. Try gradually increasing it!
run_simulation(sim_time_step=0.001)
