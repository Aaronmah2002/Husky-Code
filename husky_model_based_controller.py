import os
import numpy as np
import pydot
import time
import csv

from pydrake.geometry import Rgba

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
from pydrake.common.eigen_geometry import Quaternion


# Function to get the path relative to the script's directory
def get_relative_path(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, path))


# Start the visualizer and clean up previous instances
meshcat = StartMeshcat()
visualize = False # Bool to switch the viszualization and simulation
pathFollowing = False
Max_Sim_Time = 10.0


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
    def __init__(self, plant, plant_context):
        super().__init__()
        self.init_data(self)

        self.q_des = []
        self.q_dot_des = []
        self.q_ddot_des = []

        #Initialisation of the transformation matrices
        self.abs_to_rel = []
        self.rel_to_abs = []

        self.t_array = []
        #Generalized coordinates array
        self.gen_coord = []
        self.gen_coord_dot = []
    
        self.z_robot_1_array = []
        self.z_robot_2_array = []

        self.z_robot_dot_1_array = [0,0]
        self.z_robot_dot_2_array = [0,0]
        self.index = 2

        self.xi_array = [0]

        self.z_array = []

        self.K_p = np.diag([0, 0, 10, 50, 50, 0, 0, 0, 0, 0])
        self.K_d = np.diag([0, 0, 0, 100, 100, 0, 0, 0, 0, 0])

        # Declare input ports for desired and current states
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=21)
        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=7)

        self.step = 1
        # Store plant and context for dynamics calculations
        self.plant, self.plant_context = plant, plant_context

        # Declare discrete state and output port for control input (tau_u)
        state_index = self.DeclareDiscreteState(4)  # 4 state variables.
        self.DeclareStateOutputPort("tau_u", state_index)  # output: y=x.
        
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/1000,  # One millisecond time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.compute_tau_u) # Call the Update method defined below.
        
        
        '''self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/100,  # One millisecond time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.update_data)
        '''
        self.path_steps = 5000
        self.circ_path_x = []
        self.circ_path_y = []

        for i in range(self.path_steps+1):
            self.circ_path_x.append( 2* np.cos(i*2*np.pi/self.path_steps) )
            self.circ_path_y.append( 2* np.sin(i*2*np.pi/self.path_steps) )

        self.inf_path_x = []
        self.inf_path_y = []

        for i in range(self.path_steps+1):
            self.inf_path_x.append( 2* np.cos(i*2*np.pi/self.path_steps) )
            self.inf_path_y.append( 2* np.sin(i*4*np.pi/self.path_steps) )

    def init_data(self,plant):
        #Initialisation of the data for graphs for plotting

        #Arrays storing the time
        self.time_array = []

        #Arrays storing the torques applied to the wheels
        self.tau_l_array = []
        self.tau_r_array = []

        #Arrays storing the relative speeds of the robot
        self.x_dot_array = []
        self.y_dot_array = []
        self.w_z_array = []
        
        #Arrays storing the absolute position of the robot
        self.X_pos_array = []
        self.Y_pos_array = []
        self.Theta_array = []

        #Arrays storing the errors in x, y, theta (relative)
        self.y_error_array = []
        self.x_error_array = []
        self.angular_error = []

        self.x_dot_ref_array = []
        self.w_z_ref_array = []
    
    '''def update_data(self, context, discrete_state):

        state = self._current_state_port.Eval(context)

        current_time = context.get_time()

        self.time_array.append(current_time)

        self.tau_l_array.append(np.round( self.tau_l, 2))
        self.tau_r_array.append(np.round( self.tau_r, 2))

        self.x_dot_array.append(np.round( self.rel_vel[0], 2))
        self.y_dot_array.append(np.round( self.rel_vel[1], 2))
        self.w_z_array.append(np.round( self.rel_vel[2], 2))

        self.X_pos_array.append(np.round(state[4],2))
        self.Y_pos_array.append(np.round(state[5],2))
        self.Theta_array.append(np.round( self.theta, 2))

        self.x_error_array.append(self.x_error)
        self.y_error_array.append(self.y_error)
        self.angular_error.append(self.theta_error)

        self.x_dot_ref_array.append(self.x_dot_ref)
        self.w_z_ref_array.append(self.w_z_ref)
'''

    def compute_tau_u(self, context, discrete_state):
        M_mat = self.plant.CalcMassMatrix(self.plant_context) # 10x10 matrix
        C_mat = self.plant.CalcBiasTerm(self.plant_context) #1x10 matrix
        E_mat = self.plant.MakeActuationMatrix() #10x4 matrix
        g_mat = np.transpose(self.plant.CalcGravityGeneralizedForces(self.plant_context))
        #Order for the states : theta_x,theta_y,theta_z, x, y, z, alpha_1, alpha_2, alpha_3, alpha_4
        W = 0.555
        R = 0.165
        J_w = np.array([
            [1, 0, W/2],
            [1, 0, -W/2],
            [1, 0, W/2],
            [1, 0, -W/2]
        ])

        q = self._current_state_port.Eval(context)
        q_d = self._desired_state_port.Eval(context)



        self.theta = 2 * np.arctan2(q[3],q[0])
        
        #Computation of reltive to absolute trnaformation matrix
        self.rel_to_abs = np.array([
            [np.cos(self.theta), -np.sin(self.theta), 0],
            [np.sin(self.theta),np.cos(self.theta),0],
            [0,0,1]])
        
        #Computation of absolute to reltive tranformation matrix
        self.abs_to_rel = self.rel_to_abs.T
        
        #Necessary absolute velocities : X*, Y*, W_z
        self.abs_vel = np.array([
            [q[14]],
            [q[15]],
            [q[13]]
        ])

        
        rpy = RollPitchYaw(Quaternion(q[0], q[1], q[2], q[3]))
        robot_rot = np.array([rpy.roll_angle(), rpy.pitch_angle(), rpy.yaw_angle()])
        robot_pos = q[4:11]
        
        q_robot = np.concatenate((robot_rot,robot_pos))

        q_dot_robot = q[11:21]


        q_ddot_robot = self.plant.get_generalized_acceleration_output_port().Eval(self.plant_context);


        Phi = np.array([0,0,W/2,-np.sin(rpy.yaw_angle()),np.cos(rpy.yaw_angle()),0,0,0,0,0])

        if(pathFollowing):
            a = 0
        else:
            des_rpy = RollPitchYaw(Quaternion(q_d[0], q_d[1], q_d[2], q_d[3]))
            desired_rot = np.array([des_rpy.roll_angle(), des_rpy.pitch_angle(), des_rpy.yaw_angle()])
            desired_pos = q_d[4:7]
            desired_wheel = [0,0,0,0]

            self.q_des = np.concatenate((desired_rot, desired_pos, desired_wheel))
            self.q_dot_des = np.zeros(10)
            self.q_ddot_des = np.zeros(10)


        #print(f"M : {self.q_des} === {self.q_dot_des}\n\n")

        e = self.q_des - q_robot
        e_dot = self.q_dot_des - q_dot_robot

        q_ddot_ref = self.K_p @ np.transpose(e) + self.K_d @ np.transpose(e_dot) + np.transpose(self.q_ddot_des)
        
        a = J_w @ np.array([q_ddot_ref[3],q_ddot_ref[4],q_ddot_ref[2]])

        q_ddot_ref = np.concatenate((q_ddot_ref[0:6], a))

        tau = np.linalg.pinv(E_mat) @ (M_mat @ q_ddot_ref + C_mat + g_mat - np.transpose(Phi))
        tau = np.clip(tau, -20, 20)
        #print(f"Position : {q_ddot_robot} \n\n\n")
        print(f"Tau : {tau} \n\n\n")
        print(f"Qdd : {q_ddot_ref} \n\n\n")
        #Time
        t_reg = context.get_time()
        self.t_array.append(t_reg)
        
        # Update the output port = state
        discrete_state.get_mutable_vector().SetFromVector(tau)


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
    init_angle_deg = 0; 
    rotation_angle = 0/180*np.pi 

    # Set the initial position of the robot
    plant.SetDefaultPositions([np.cos(rotation_angle/2), 0.0, 0.0, np.sin(rotation_angle/2), 0, 0, 0.138, 0.0, 0.0, 0.0, 0.0]) #(quaternions), translations, wheel rotations
    
    # Add visualization to see the geometries in MeshCat
    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    # Add a PD controller to regulate the robot
    controller = builder.AddNamedSystem("PD controller", Controller(plant, plant_context))

    desired_rotation_angle = 0/180*np.pi 

    despos_ = [np.cos(desired_rotation_angle/2), 0.0, 0.0, np.sin(desired_rotation_angle/2),-5,0,0]

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
    simulator.AdvanceTo(Max_Sim_Time)  # Adjust this time as needed
    meshcat.PublishRecording() #For the replay
    plotGraphs(controller)

def plotGraphs(controller):
    #Speed plots
    fig0,axes0=plt.subplots(2,2)
    fig0.canvas.set_window_title('')  

    controller.z_robot_1_array.pop()
    axes0[0][0].plot(controller.t_array, controller.z_robot_1_array)
    axes0[0][0].set_title(f'z_1')
    axes0[0][0].grid(which='both',linestyle='--')

    controller.z_robot_2_array.pop()
    axes0[1][0].plot(controller.t_array, controller.z_robot_2_array)
    axes0[1][0].set_title(f'z_2')
    axes0[1][0].grid(which='both',linestyle='--')

    controller.z_robot_dot_1_array.pop()
    controller.z_robot_dot_1_array.pop()
    axes0[0][1].plot(controller.t_array, controller.z_robot_dot_1_array)
    axes0[0][1].set_title(f'z_dot_1')
    axes0[0][1].grid(which='both',linestyle='--')

    controller.z_robot_dot_2_array.pop()
    controller.z_robot_dot_2_array.pop()
    axes0[1][1].plot(controller.t_array, controller.z_robot_dot_2_array)
    axes0[1][1].set_title(f'z_dot_2')
    axes0[1][1].grid(which='both',linestyle='--')
    plt.subplots_adjust(hspace=0.5)

    plt.show()



def createCSV(array_1, array_2, array_3, array_4,title):

    # Example arrays with different lengths
    array1 = [1, 2, 3, 4]
    array2 = [5, 6]
    array3 = [9, 10, 11]
    array4 = [13, 14, 15, 16, 17]

    # Find the maximum length of the arrays
    max_length = max(len(array_1), len(array_2), len(array_3), len(array_4))

    # Function to pad arrays to the same length
    def pad_array(arr, length):
        return arr + [''] * (length - len(arr))

    # Pad all arrays
    array_1 = pad_array(array_1, max_length)
    array_2 = pad_array(array_2, max_length)
    array_3 = pad_array(array_3, max_length)
    array_4 = pad_array(array_4, max_length)

    # Combine arrays into rows
    rows = zip(array_1, array_2, array_3, array_4)

    # Write to CSV
    with open(title, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Robot position', 'Robot position', 'Reference', 'Reference'])  # Header
        writer.writerows(rows)

    print("CSV file 'output.csv' created successfully.")

# Run the simulation with a specific time step. Try gradually increasing it!
run_simulation(sim_time_step=0.00005)
