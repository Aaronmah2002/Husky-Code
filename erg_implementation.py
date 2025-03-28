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
T = 0.2  # Prediction horizon
dt = 1e-2
##reset files
files=["controller_q","q_v_erg","tau_erg","tau_pred","state_dyn","x_next","state_pred"]
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

        # Declare input ports for desired and current states
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=21)
        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=7)

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

    def compute_tau_u(self, context, discrete_state):
        """
        robot_rot_quaternion = self.q[0:4]
        robot_pos = self.q[4:7]
        robot_wheel_rot = self.q[7:11]
        robot_ang_velocity = self.q[11:14]
        robot_speed = self.q[14:17] #Absolute values
        robot_wheel_ang_velocity = self.q[17:21]
        """

        self.q = self._current_state_port.Eval(context)
        self.q_d = self._desired_state_port.Eval(context)

        #Computation of rotation angle
        self.theta = 2 * np.arctan2(self.q[3],self.q[0])
        theta_d = 2 * np.arctan2(self.q_d[3],self.q_d[0])
        L = 0.670/2
        r = 0.165

        M_mat = self.plant.CalcMassMatrix(self.plant_context) # 10x10 matrix
        C_mat = self.plant.CalcBiasTerm(self.plant_context).reshape(-1, 1) #1x10 matrix
        #E_mat = self.plant.MakeActuationMatrix() #10x4 matrix
        g_mat = self.plant.CalcGravityGeneralizedForces(self.plant_context).reshape(-1, 1)
        
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
            [np.cos(self.theta), 0],
            [np.sin(self.theta), 0],
            [0, 0],
            [1/r, -L/r],
            [1/r, L/r],
            [1/r, -L/r],
            [1/r, L/r]])

        G_dot = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [-np.sin(self.theta), 0],
            [np.cos(self.theta), 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]])

        eta = np.array([
            [np.cos(self.theta) * self.q[14] + np.sin(self.theta) * self.q[15]],
            [self.q[13]]])

        u = np.array([
            [1],
            [1]]) #Pseudo acceleration vector (a_x, w_z dot)


        self.abs_to_rel = np.array([
            [np.cos(self.theta), np.sin(self.theta), 0],
            [-np.sin(self.theta),np.cos(self.theta),0],
            [0,0,1]])

        a=0.5
        b=0.5
        m = 56.5
        g = 9.81

        y_dot = -np.sin(self.theta) * self.q[14] + np.cos(self.theta) * self.q[15]

        sgn_x1 = float(np.sign(eta[0] - L*eta[1]))
        sgn_x2 = float(np.sign(eta[0] + L*eta[1]))

        sgn_y1 = float(np.sign(y_dot + a*eta[1]))
        sgn_y3 = float(np.sign(y_dot - a*eta[1]))

        fr = 0.05
        mu = 0.5

        Rx = fr * (m * g / 2) * (sgn_x1 + sgn_x2)


        Fy = mu * (m * g / (a + b)) * (b * sgn_y1 + a * sgn_y3)


        Mr = mu * (a * b * m * g / (a + b)) * (sgn_y1 - sgn_y3) + fr * (L * m * g / 2) * (sgn_x2 - sgn_x1)
        
        F_mat = np.array([
            [0],
            [0],
            [Mr],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0]])

        error = np.array([self.q_d[4]- [self.q[4]], [self.q_d[5] - self.q[5]], [theta_d - self.theta]])
        rel_error = self.abs_to_rel @ error
        a_err = np.arctan2(error[0],error[1])
        error[1] = a_err
        u = np.array([[2, 0, 0],[0, 80, 0]]) @ rel_error - np.array([[4, 0],[0, 100]]) @ eta

        M2 = np.transpose(G) @ M_mat @ G
        M3 = np.transpose(G) @ M_mat @ G_dot
        tau = np.linalg.pinv(np.transpose(G) @ E_mat) @ (M2 @ u + M3 @ eta + np.transpose(G) @ (g_mat + C_mat + F_mat))
        
        print(f"{rel_error} \n\n\n")
        print(f"s : {F_mat} \n\n\n")
        print(f"t : {tau} \n\n\n")
        # Update the output port = state
        discrete_state.get_mutable_vector().SetFromVector(tau)
######################################################################################################
#                             ########Define Explicit Reference Governor as a LeafSystem #######   
######################################################################################################
num_positions=7
num_velocities=4
def calc_dynamics(x, u, plant_pred, plant_context):
    # assert diagram.IsDifferenceEquationSystem()[0], "must be a discrete-time system"
    """
    Calculate the next state given the current state x and control input u.
    
    Args:
        x: Current state vector.
        u: Control input vector.
    
    Returns:
        The next state vector.
    """
    plant_context.SetDiscreteState(x)
    plant_pred.get_actuation_input_port().FixValue(plant_context,u)
    state = plant_context.get_discrete_state()
    st=state.get_vector().value().flatten()
    csv_writing_file("state_dyn",st)
    #diagram_context.CalcForcedDiscreteVariableUpdate(plant_context, state)
    x_next = state.get_vector().value().flatten()
    csv_writing_file("x_next",x_next)
    return x_next
class ExplicitReferenceGovernor:
    def __init__(self, robust_delta_tau_, kappa_tau_, 
                 robust_delta_q_, kappa_q_, robust_delta_dq_, kappa_dq_, 
                 robust_delta_dp_EE_, kappa_dp_EE_, kappa_terminal_energy_):
        """
        Initialize the Explicit Reference Governor (ERG) with given parameters.
        
        Args:
            robust_delta_tau_ (float): Robustness parameter for joint torques.
            kappa_tau_ (float): Scaling parameter for joint torques.
            robust_delta_q_ (float): Robustness parameter for joint positions.
            kappa_q_ (float): Scaling parameter for joint positions.
            robust_delta_dq_ (float): Robustness parameter for joint velocities.
            kappa_dq_ (float): Scaling parameter for joint velocities.
            robust_delta_dp_EE_ (float): Robustness parameter for end-effector velocities.
            kappa_dp_EE_ (float): Scaling parameter for end-effector velocities.
            kappa_terminal_energy_ (float): Scaling parameter for terminal energy.
        """
        self.eta_ = 0.05 
        self.zeta_q_ = 0.15
        self.delta_q_ = 0.1 
        self.dt_ = 0.01  # Sampling time for the refrence governor
        self.num_positions=7

        num_positions=self.num_positions

        self.num_velocities=4 #wz,Vx,Vy,Vz (vx,vy,vz)
        num_velocities=self.num_velocities

        # Controller gains
        self.Kp_ = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        self.Kd_ = [3, 3, 3, 3]

        # Prediction parameters
        prediction_dt_ = dt  # Time step for predictions
        prediction_horizon_ = T  # Total prediction horizon
        self.num_pred_samples_ = int(prediction_horizon_ / prediction_dt_)

        # Robustness and scaling parameters
        self.robust_delta_tau_ = robust_delta_tau_
        self.kappa_tau_ = kappa_tau_
        self.robust_delta_q_ = robust_delta_q_
        self.kappa_q_ = kappa_q_
        self.robust_delta_dq_ = robust_delta_dq_
        self.kappa_dq_ = kappa_dq_
        self.robust_delta_dp_EE_ = robust_delta_dp_EE_
        self.kappa_dp_EE_ = kappa_dp_EE_
        self.kappa_terminal_energy_ = kappa_terminal_energy_

        # PLant for simulation
        self.plant_pred= MultibodyPlant(1e-3)
        arm = Parser(self.plant_pred).AddModelsFromUrl("file:///home/gheorghetb/drake_brubotics/models/descriptions/robots/husky_description/urdf/husky.urdf") 
        #self.plant_pred.set_contact_surface_representation(mesh_type)
        #self.plant_pred.set_contact_model(contact_model)
        # plant.SetPositionsAndVelocities(plant_context, x0 = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]))
        self.plant_pred.Finalize()
        self.plant_context = self.plant_pred.CreateDefaultContext()
        self.control_erg=Controller(self.plant_pred,self.plant_context)

        # Prediction lists for joint positions, velocities, and torques
        self.q_pred_list_ = np.zeros((num_positions, self.num_pred_samples_ + 1)) #7
        self.dq_pred_list_ = np.zeros((4, self.num_pred_samples_ + 1)) #2
        self.tau_pred_list_ = np.zeros((4, self.num_pred_samples_ + 1)) #4
        
        # Limits for joint angles, velocities, and torques
        ##self.limit_q_min_ = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) #useless
        ##self.limit_q_max_ = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]) #useless
        self.limit_tau_ = np.array([200,200,200,200]) #[20,20,20,20]
        self.limit_dq_ = np.array([10,10,10,10]) #[0,0,10]
        ##self.limit_dp_EE_ = [1.7, 2.5]  # Translation and rotation limits for the end effector #not for the moment


    def get_qv(self, q, dq, tau, q_r, q_v):
        """
        Compute the new reference joint positions using the navigation field and DSM.
        
        Args:
            q (np.array): Current joint positions.
            dq (np.array): Current joint velocities.
            tau (np.array): Current joint torques.
            q_r (np.array): Desired reference joint positions.
            q_v (np.array): Current applied reference joint positions.
        
        Returns:
            np.array: Updated reference joint positions.
        """
        rho_ = self.navigationField(q_r, q_v)
        #print(q_v)
        
        DSM_ = self.trajectoryBasedDSM(q, dq, tau, q_v)

        if DSM_ > 0:
          q_v_new = q_v + DSM_ * rho_ * self.dt_ 
        else:
          q_v_new = q + np.min([np.linalg.norm(DSM_ * rho_ * self.dt_), np.linalg.norm(q_r - q_v)]) * DSM_ * rho_ / max(np.linalg.norm(DSM_ * rho_), self.eta_)
        
        return q_v_new

    def navigationField(self, q_r, q_v): # if rho cst --> q_v doesn't update
        """
        Compute the navigation field based on attraction and repulsion forces.
        """
        rho_att = np.zeros(num_positions)
        rho_rep_q = np.zeros(num_positions)
        rho = np.zeros(num_positions)

        # Attraction field
        rho_att = (q_r - q_v) / max(np.linalg.norm(q_r - q_v), self.eta_)
        # print(f"norm rho_att = {np.linalg.norm(rho_att)}")
        # print(f"rho_att = \n {rho_att}")
        '''
        # Joint angle repulsion field (q)
        for i in range(7):
            rho_rep_q[i] = max((self.zeta_q_ - abs(q_v[i] - self.limit_q_min_[i])) / (self.zeta_q_ - self.delta_q_), 0.0) - \
                           max((self.zeta_q_ - abs(q_v[i] - self.limit_q_max_[i])) / (self.zeta_q_ - self.delta_q_), 0.0)
        # print(f"norm rho_rep_q = {np.linalg.norm(rho_rep_q)}")
        # print(f"rho_rep_q = \n {rho_rep_q}")
        '''
        # Total navigation field
        rho = rho_att #+ rho_rep_q
        csv_writing_file("navfield",rho)
        
        return rho

    def trajectoryBasedDSM(self, q, dq, tau, q_v):
      """
      Compute the Dynamic Safety Margin (DSM) based on trajectory predictions.
      """
      # Get trajectory predictions and save predicted q, dq, and tau in lists
      self.trajectoryPredictions(np.concatenate((q, dq,np.zeros(10))), tau, q_v)
      

      # Compute DSMs
      DSM_tau_ = self.dsmTau()
      #DSM_q_ = self.dsmQ()
      DSM_dq_ = self.dsmDq()
      #DSM_dp_EE_ = self.dsmDpEE()
      # DSM_terminal_energy_ = self.dsmTerminalEnergy(q_v)

      # Print DSMs
    #   print(f"DSM_tau_: {DSM_tau_}")
    #   print(f"DSM_q_: {DSM_q_}")
    #   print(f"DSM_dq_: {DSM_dq_}")
    #   print(f"DSM_dp_EE_: {DSM_dp_EE_}")
      # print(f"DSM_terminal_energy_: {DSM_terminal_energy_}")

      # Find the minimum among the DSMs
      DSM = min(DSM_tau_,DSM_dq_)
      #DSM = min(DSM,DSM_q_)
      #DSM = min(DSM,DSM_dp_EE_)
      # DSM = min(DSM,DSM_terminal_energy_)

      DSM = max(DSM, 0.0)
    #   print(f"DSM_final: {DSM}")
      
      return DSM
    

    def compute_tau_u(self, q_v):
        """
        robot_rot_quaternion = self.q[0:4]
        robot_pos = self.q[4:7]
        robot_wheel_rot = self.q[7:11]
        robot_ang_velocity = self.q[11:14]
        robot_speed = self.q[14:17] #Absolute values
        robot_wheel_ang_velocity = self.q[17:21]
        """
        
         #Initialisation of the transformation matrices
        self.abs_to_rel = []
        self.rel_to_abs = []
        #Initialisation of the controller parameters

        self.Kp_ = [3.2]
        
        #self.numposition=self.plant.num_positions()
        #self.numvelocities=self.plant.num_velocities()
        #plant.GetPositions()
        #plant.GetVelocities()
        
        

        #Computation of rotation angle
        self.theta = 2 * np.arctan2(self.q_v[3],self.q_v[0])
        
        #Computation of reltive to absolute trnaformation matrix
        self.rel_to_abs = np.array([
            [np.cos(self.theta), -np.sin(self.theta), 0],
            [np.sin(self.theta),np.cos(self.theta),0],
            [0,0,1]])
        
        #Computation of absolute to reltive tranformation matrix
        self.abs_to_rel = self.rel_to_abs.T
        
        #Necessary absolute velocities : X*, Y*, W_z
        self.abs_vel = np.array([
            [self.q_v[14]],
            [self.q_v[15]],
            [self.q_v[13]]
        ])

        r = 0.165 #Wheel radius
        d = 0.613 #Robot width (distance between wheels)

        inv_kin_mat = np.array([
            [1/r, d/(2*r)],
            [1/r, -d/(2*r)],
        ])


        #Necessary relative velocities : x*, x*, w_z
        self.rel_vel = self.abs_to_rel @ self.abs_vel

        #Error computation in absolutevalues
        X_error = self.q_r[4] - self.q_v[4]
        Y_error = self.q_r[5] - self.q_v[5]

        #Final angle for the robot in absolute/relative
        ref_angle = 2 * np.arctan2(self.q_r[3],self.q_r[0])

        #Angular error between the robot position and desired position
        self.theta_error = np.arctan2(Y_error, X_error) - self.theta

        #Y error between the robot position and desired position Absolute error X and Y-> relative error y
        self.y_error = -X_error * np.sin(self.theta) + Y_error * np.cos(self.theta)

        #X error between the robot position and desired position Absolute errors X and Y -> relative error x
        self.x_error = X_error * np.cos(self.theta) + Y_error * np.sin(self.theta)

        self.x_dot_ref = 1 * self.x_error - 0.2 * float(self.rel_vel[0])
        #self.x_dot_ref = np.clip(self.x_dot_ref, -1, 1)

        self.w_z_ref = 30 * self.y_error
        #self.w_z_ref = np.clip(self.w_z_ref, -2, 2)

        #=======================================
        #x_dot_ref = 1
        #w_z_ref = 1/2*np.sin(context.get_time()/2)

        #Computation of the reference speeds of the wheels
        w_ref = inv_kin_mat @ np.array([[self.x_dot_ref],[self.w_z_ref]])
        
        #Regulation of
        self.tau_l = 20*self.Kp_[0]*(w_ref[1] - self.q[17])
        self.tau_r = 20*self.Kp_[0]*(w_ref[0] - self.q[18])

        #self.tau_l = np.clip(self.tau_l , -20, 20)
        #self.tau_r = np.clip(self.tau_r , -20, 20)

        self.tau = np.array([self.tau_l, self.tau_r, self.tau_l, self.tau_r])
        csv_writing_file("tau_controller",self.tau)
        
        # Compute gravity forces for the current state
        # self.plant_context_ad.SetDiscreteState(self.q)
        
        
        #print('dimension of tau ',self.tau.shape()) #spoiler dime(tau)=4
    def trajectoryPredictions(self, state, tau, q_v):
      """
      Predict joint positions, velocities, and torques over the prediction horizon.
      """
      q_pred, dq_pred = state[:num_positions], state[7:11]
      
      tau_pred = tau
      # gravity_pred = np.zeros(plant.get_actuation_input_port().size())  # Placeholder for gravity prediction

      # Initialize lists to store predicted states
      q_pred_traj = [q_pred.copy()]
      dq_pred_traj = [dq_pred.copy()]

      self.q_pred_list_[:, 0] = q_pred
      self.dq_pred_list_[:, 0] = dq_pred
      self.tau_pred_list_[:, 0] = tau_pred

      # self.distance_tau_ = self.distanceTau(tau_pred)
      # self.distance_q_ = self.distanceQ(q_pred)
      # self.distance_dq_ = self.distanceDq(dq_pred)
      # self.distance_dp_EE_ = self.distanceDpEE(q_pred, dq_pred)

      for k in range(self.num_pred_samples_): #200
          # Compute tau_pred
          #gravity_pred = - self.plant_pred.CalcGravityGeneralizedForces(self.plant_context) # Compute gravity_pred for the current state
          tau_pred =   self.Kd_ * dq_pred #+ gravity_pred 
          #tau_pred=self.control_erg.compute_tau_u(self.plant_context,state)
          
          

          # Solve for x[k+1] using the computed tau_pred
          state_pred = calc_dynamics(np.concatenate((q_pred, dq_pred,np.zeros(10))), tau_pred, self.plant_pred, self.plant_context)  # Adjust this based on your calculation method
          #print(state_pred) only 7 elem ?????
          q_pred = state_pred[:num_positions]
          ##print(q_pred)
          dq_pred = state_pred[7:11]

          # # Store predicted states
          # q_pred_traj.append(q_pred.copy())
          # dq_pred_traj.append(dq_pred.copy())

          # Add q, dq, and tau to prediction list
          self.q_pred_list_[:, k + 1] = q_pred
          self.dq_pred_list_[:, k + 1] = dq_pred
          self.tau_pred_list_[:, k + 1] = tau_pred
          csv_writing_file("tau_pred",self.tau_pred_list_)
          csv_writing_file("state_pred",state_pred)
      # # Convert lists to arrays for plotting
      # q_pred_traj = np.array(q_pred_traj)
      # dq_pred_traj = np.array(dq_pred_traj)

      # # Define total prediction time T and time step size dt
      # dt = T / self.num_pred_samples_  # Time step size
      # time_steps = np.linspace(0, T, self.num_pred_samples_ + 1)  # Time steps array

      # # Plot predicted and desired states
      # plt.figure()
      # plt.plot(time_steps, q_pred_traj, label='Predicted q')
      # # plt.plot(time_steps, dq_pred_traj, label='Predicted dq')
      # plt.xlabel('Time')
      # plt.ylabel('Angle Position / Velocity')
      # plt.legend()
      # plt.show()


    def dsmTau(self):
        """
        Compute the DSM for joint torques.
        """
        for k in range(self.tau_pred_list_.shape[1]):  # number of prediction samples + 1
            tau_pred = self.tau_pred_list_[:, k]
            DSM_tau_temp = self.distanceTau(tau_pred) - self.robust_delta_tau_ #avoid violate limitation
            if k == 0:
                DSM_tau = DSM_tau_temp
            else:
                DSM_tau = min(DSM_tau, DSM_tau_temp)

        DSM_tau = self.kappa_tau_ * DSM_tau
        return DSM_tau #with kappa_tau=0.1 --> always small

    def dsmQ(self):
        """
        Compute the DSM for joint positions.
        """
        for k in range(self.q_pred_list_.shape[1]):  # number of prediction samples + 1
            q_pred = self.q_pred_list_[:, k]
            DSM_q_temp = self.distanceQ(q_pred) - self.robust_delta_q_
            if k == 0:
                DSM_q = DSM_q_temp
            else:
                DSM_q = min(DSM_q, DSM_q_temp)

        DSM_q = self.kappa_q_ * DSM_q
        return DSM_q

    def dsmDq(self):
        """
        Compute the DSM for joint velocities.
        """
        for k in range(self.dq_pred_list_.shape[1]):  # number of prediction samples + 1
            dotq_pred = self.dq_pred_list_[:, k]
            DSM_dotq_temp = self.distanceDq(dotq_pred) - self.robust_delta_dq_
            if k == 0:
                DSM_dotq = DSM_dotq_temp
            else:
                DSM_dotq = min(DSM_dotq, DSM_dotq_temp)

        DSM_dotq = self.kappa_dq_ * DSM_dotq
        return DSM_dotq

    def dsmDpEE(self):
        """
        Compute the DSM for end-effector velocities.
        """
        for k in range(self.q_pred_list_.shape[1]):  # number of prediction samples + 1
            q_pred = self.q_pred_list_[:, k]
            dotq_pred = self.dq_pred_list_[:, k]
            DSM_dotp_EE_temp = self.distanceDpEE(q_pred, dotq_pred) - self.robust_delta_dp_EE_            
            if k == 0:
                DSM_dotp_EE = DSM_dotp_EE_temp
            else:
                DSM_dotp_EE = min(DSM_dotp_EE, DSM_dotp_EE_temp)

        DSM_dotp_EE = self.kappa_dp_EE_ * DSM_dotp_EE
        return DSM_dotp_EE

    # def dsmTerminalEnergy(self, q_v):
    #     k = self.q_pred_list_.shape[1] - 1  # final prediction sample
    #     q_pred = self.q_pred_list_[:, k]
    #     dotq_pred = self.dq_pred_list_[:, k]
    #     distance_terminal_energy = self.distanceTerminalEnergy(q_pred, dotq_pred, q_v)
    #     DSM_terminal_energy = self.kappa_terminal_energy_ * distance_terminal_energy
    #     return DSM_terminal_energy


    def distanceTau(self, tau_pred):
        for i in range(4):
            tau_lowerlimit = tau_pred[i] - (-self.limit_tau_[i])
            tau_upperlimit = self.limit_tau_[i] - tau_pred[i]
            tau_distance_temp = min(tau_lowerlimit, tau_upperlimit)
            if i == 0:
                tau_distance = tau_distance_temp
            else:
                tau_distance = min(tau_distance, tau_distance_temp)
        return tau_distance

    def distanceQ(self, q_pred):
        for i in range(7):  # include all joints
            q_lowerlimit = q_pred[i] - self.limit_q_min_[i]
            q_upperlimit = self.limit_q_max_[i] - q_pred[i]
            q_distance_temp = min(q_lowerlimit, q_upperlimit)
            if i == 0:
                q_distance = q_distance_temp
            else:
                q_distance = min(q_distance, q_distance_temp)
        # print(f"q_distance: {q_distance}")
        return q_distance

    def distanceDq(self, dotq_pred):
        for i in range(4):
            distance_dotq_lowerlimit = dotq_pred[i] - (-self.limit_dq_[i])
            distance_dotq_upperlimit = self.limit_dq_[i] - dotq_pred[i]
            distance_dotq_temp = min(distance_dotq_lowerlimit, distance_dotq_upperlimit)
            if i == 0:
                distance_dotq = distance_dotq_temp
            else:
                distance_dotq = min(distance_dotq, distance_dotq_temp)
        return distance_dotq
      
    def distanceDpEE(self, q_pred, dotq_pred):
        endeffector_jacobian = np.zeros((6, num_positions))  # Example initialization
        dotp_EE = endeffector_jacobian @ dotq_pred

        for i in range(6):
            if i < 3:  # Translation
                distance_dotp_EE_lowerlimit = dotp_EE[i] - (-self.limit_dp_EE_[0])
                distance_dotp_EE_upperlimit = self.limit_dp_EE_[0] - dotp_EE[i]
            else:  # Rotation
                distance_dotp_EE_lowerlimit = dotp_EE[i] - (-self.limit_dp_EE_[1])
                distance_dotp_EE_upperlimit = self.limit_dp_EE_[1] - dotp_EE[i]
            distance_dotp_EE_temp = min(distance_dotp_EE_lowerlimit, distance_dotp_EE_upperlimit)
            if i == 0:
                distance_dotp_EE = distance_dotp_EE_temp
            else:
                distance_dotp_EE = min(distance_dotp_EE, distance_dotp_EE_temp)
        return distance_dotp_EE
    
    # def distanceTerminalEnergy(self, q_pred, dotq_pred, q_v):
    #     # Placeholder: Adjust this function based on your actual model
    #     m_total = 1.0  # Example mass total
    #     I_total = np.eye(3)  # Example inertia total
    #     F_x_Ctotal = np.zeros(3)  # Example center of mass force

    #     # Replace this with your actual computation of the mass matrix
    #     mass_matrix = np.eye(num_positions)  # Example initialization
    #     Kp_matrix = np.diag([1.0] * num_positions)  # Example Kp matrix

    #     terminal_energy = 0.5 * dotq_pred.T @ mass_matrix @ dotq_pred
    #     terminal_energy += 0.5 * (q_v - q_pred).T @ Kp_matrix @ (q_v - q_pred)

    #     terminal_energy_limit_ = 1.0  # Example limit
    #     distance_terminal_energy = terminal_energy_limit_ - terminal_energy

    #     return distance_terminal_energy
    
####################################
#                                  #
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
        trajInit_=[0,0,0,0,1,-1,0]
        self.q_v_n = trajInit_ 

        self.erg = ExplicitReferenceGovernor(
          robust_delta_tau_=0.01, kappa_tau_=0.10,
          robust_delta_q_=0.1, kappa_q_=15.0, robust_delta_dq_=0.1, kappa_dq_=7.0,
          robust_delta_dp_EE_=0.01, kappa_dp_EE_=7.0, kappa_terminal_energy_=7.5)
        
    def refrence(self, context, output):
        # Evaluate the input ports
        state = self._state_port.Eval(context)
        q = state[:num_positions]
        dq = state[12:16]
        tau = self._tau_port.Eval(context)
        q_r = self._qr_port.Eval(context)
        q_v = self.q_v_n
        csv_writing_file("tau_erg",tau)
        q_v_n = self.erg.get_qv(q, dq, tau, q_r, q_v)
        q_v_new = q_v_n
        print(q_v)
        #csv_writing_file("q_v_erg",q_v_new)
        #print(q_v_new)
        #print(q_v_new)
        # Write into the output vector.
        output.SetFromVector(q_v_new)


        


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

    despos_ = [np.cos(desired_rotation_angle/2), 0.0, 0.0, np.sin(desired_rotation_angle/2),5,0,0]
    
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
    simulator.AdvanceTo(3.0)  # Adjust this time as needed
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
