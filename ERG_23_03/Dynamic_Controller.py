import os
import numpy as np
import pydot
import time
import csv

from pydrake.geometry import Rgba
from pydrake.all import AbstractValue
from pydrake.all import ContactResults
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




# Set the path to your robot model:
robot_path = get_relative_path("../models/descriptions/robots//husky_description/urdf/husky.urdf")
scene_path = get_relative_path("../models/objects&scenes/scenes/floor.sdf")


######################################################################################################
#                             ########Define PD+G Controller as a LeafSystem #######   
######################################################################################################
import math

class Controller(LeafSystem):
    def __init__(self, plant, plant_context):
        super().__init__()
        #Creation of the arrays storing all the data for plotting
        self.init_data(self)
        self.case = 0
        self.f_x = [0,0,0,0]
        self.f_y = [0,0,0,0]
        self.F_abs = np.zeros((4, 3))
        self.F_abs1 = np.zeros((4, 3))
        self.F_abs2 = np.zeros((4, 3))


        self.v_prev = 0
        self.w_prev = 0
        # Declare input ports for desired and current states
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=21)
        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=7)
        self.contact_results_port = self.DeclareAbstractInputPort("contact_results", AbstractValue.Make(ContactResults()))
    
        self.DeclareVectorOutputPort(name="contact_forces", size=12, calc = self.compute_friction)
        #self._desired_state_port = self.DeclareVectorInputPort(name="Force_Contact")

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


    def compute_friction(self, context, discrete_state):
        
        discrete_state.set_value(self.F_abs.flatten)


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

        self.a_x_array = []
        self.w_z_dot_array = []

        self.a_x_ref = []
        self.a_x_real = []

        self.w_d_ref = []
        self.w_d_real = []


    def compute_tau_u(self, context, discrete_state):
        """
        robot_rot_quaternion = self.q[0:4]
        robot_pos = self.q[4:7]
        robot_wheel_rot = self.q[7:11]
        robot_ang_velocity = self.q[11:14]
        robot_speed = self.q[14:17] #Absolute values
        robot_wheel_ang_velocity = self.q[17:21]
        """
        current_time = context.get_time()
        self.time_array.append(current_time)


        self.q = self._current_state_port.Eval(context)
        self.q_d = self._desired_state_port.Eval(context)

        #Computation of rotation angle
        self.theta = 2 * np.arctan2(self.q[3],self.q[0])
        theta_d = 2 * np.arctan2(self.q_d[3],self.q_d[0])
        L = 0.670/2
        r = 0.165

        eta = np.array([
            [np.cos(self.theta) * self.q[14] + np.sin(self.theta) * self.q[15]],
            [self.q[13]]])

        x_icr = - float((-np.sin(self.theta) * self.q[14] + np.cos(self.theta) * self.q[15] ) / eta[1])
        if(abs(eta[1]) < 0.05):
            x_icr = 0

        #print(f"icr : {x_icr}")

        M_mat = self.plant.CalcMassMatrix(self.plant_context)[:10,:10] # 10x10 matrix
        C_mat = self.plant.CalcBiasTerm(self.plant_context).reshape(-1, 1)[:10] #1x10 matrix
        g_mat = self.plant.CalcGravityGeneralizedForces(self.plant_context)[:10].reshape(-1, 1)
        
        E_mat = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-L/r, L/r, -L/r, L/r],
            [1/r*np.cos(self.theta), 1/r*np.cos(self.theta), 1/r*np.cos(self.theta), 1/r*np.cos(self.theta)],
            [1/r*np.sin(self.theta), 1/r*np.sin(self.theta), 1/r*np.sin(self.theta), 1/r*np.sin(self.theta)],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

        G = np.array([
            [0, 0],
            [0, 0],
            [0, 1],
            [np.cos(self.theta), np.sin(self.theta) * 0.2],
            [np.sin(self.theta), -np.cos(self.theta) * 0.2],
            [0, 0],
            [1, -L],
            [1, L],
            [1, -L],
            [1, L]])

        G_dot = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [-np.sin(self.theta), np.cos(self.theta) * 0.2],
            [np.cos(self.theta), np.sin(self.theta) * 0.2],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]])


        u = np.array([
            [1],
            [1]]) #Pseudo acceleration vector (a_x, w_z dot)


        self.abs_to_rel = np.array([
            [np.cos(self.theta), np.sin(self.theta), 0],
            [-np.sin(self.theta),np.cos(self.theta),0],
            [0,0,1]])

        a=0.5
        b=0.5
        m = 125.5
        g = 9.81

        '''
        contact_results = self.contact_results_port.Eval(context)

        for i in range(contact_results.num_point_pair_contacts()):
            if(i<4):
                contact_info = contact_results.point_pair_contact_info(i) 
                self.contact_point = contact_info.contact_point().tolist()
                self.contact_force = contact_info.contact_force().tolist()
                self.F_abs[i] = self.contact_force
                self.f_x[i] = self.contact_force[0] #X (absolute) forces matrix F_L, F_R, B_L, B_R
                self.f_y[i] = self.contact_force[1] #Y (absolute) forces matrix F_L, F_R, B_L, B_R
        '''
        F_abs = np.transpose(self.F_abs) #Friction force in abolute coordinates 3x4 Matrix

        F_rel = self.abs_to_rel @ F_abs #Friction force in relative coordinates


    
        
        error = np.array([self.q_d[4]- [self.q[4]], [self.q_d[5] - self.q[5]], [np.sin(theta_d - self.theta)]])

        rel_error = self.abs_to_rel @ error
        K_p_theta_variable = 10
        #print(K_p_theta_variable)
        u = np.array([[4, 0, 0],[0, 60, 0]]) @ rel_error - np.array([[7, 0],[0, 70]]) @ eta 



        Rx = F_rel[0][0] + F_rel[0][1] + F_rel[0][2] + F_rel[0][3]
        Fy = F_rel[1][0] + F_rel[1][1] + F_rel[1][2] + F_rel[1][3]
        M_dyn = (F_rel[0][1] + F_rel[0][3] - F_rel[0][0] - F_rel[0][2])*0.26 + (F_rel[1][2] + F_rel[1][3] - F_rel[1][0] - F_rel[1][1]) * 0.29
        Rx = -Rx 
        Fy = -Fy 
        M_dyn = -M_dyn*0.7
        
        #print(M_dyn)
        
        #print(M_dyn)
        F_visc = np.array([
            [0],
            [0],
            [M_dyn],
            [Rx * np.cos(self.theta) - Fy * np.sin(self.theta)],
            [Rx * np.sin(self.theta) + Fy * np.cos(self.theta)],
            [0],
            [0],
            [0],
            [0],
            [0]])

        
        M2 = np.transpose(G) @ M_mat @ G
        M3 = np.transpose(G) @ M_mat @ G_dot

        tau = np.linalg.pinv(np.transpose(G) @ E_mat) @ (M2 @ u + M3 @ eta + np.transpose(G) @ (g_mat + F_visc + C_mat))
        
        #tau = np.transpose(tau)
        #tau = np.clip(tau, -20, 20)
        
        discrete_state.get_mutable_vector().SetFromVector(tau)

        self.tau_l_array.append(tau[0])
        self.tau_r_array.append(tau[1])
        self.X_pos_array.append(self.q[4])
        self.Y_pos_array.append(self.q[5])


        a =  (eta[0][0] - self.v_prev)/0.001
        w_d = (eta[1][0] - self.w_prev)/0.001

        self.v_prev = eta[0]
        self.w_prev = eta[1]
        u_real = np.array([a, w_d])

        self.a_x_real.append(float(u_real[0]))
        self.a_x_ref.append(u[0][0])

        self.w_d_real.append(float(u_real[1]))
        self.w_d_ref.append(u[1][0])

        self.y_error_array.append(rel_error[1])
        self.x_error_array.append(rel_error[0])

        self.x_dot_array.append(float(eta[0]))
        self.w_z_array.append(float(eta[1]))
       
