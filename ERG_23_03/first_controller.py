import numpy as np
from pydrake.all import *
class Controller(LeafSystem):
    def __init__(self, plant, plant_context):
        super().__init__()

        #Creation of the arrays storing all the data for plotting
        self.init_data(self)

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
        
        
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/1000,  # One millisecond time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.update_data)
    
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
    def update_data(self, context, discrete_state):

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
        self.plant.SetPositionsAndVelocities(self.plant_context, self.q)
        POSEE = self.plant.EvalBodyPoseInWorld(self.plant_context, self.plant.GetBodyByName("base_link"))
         #Initialisation of the transformation matrices
        self.abs_to_rel = []
        self.rel_to_abs = []
        #Initialisation of the controller parameters

        self.Kp_ = [3.2]
        # Evaluate the input ports
        self.q_d = self._desired_state_port.Eval(context)
        

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

        r = 0.165 #Wheel radius
        d = 0.613 #Robot width (distance between wheels)

        inv_kin_mat = np.array([
            [1/r, d/(2*r)],
            [1/r, -d/(2*r)],
        ])


        #Necessary relative velocities : x*, x*, w_z
        self.rel_vel = self.abs_to_rel @ self.abs_vel

        #Error computation in absolutevalues
        X_error = self.q_d[4] - self.q[4]
        Y_error = self.q_d[5] - self.q[5]

        #Final angle for the robot in absolute/relative
        ref_angle = 2 * np.arctan2(self.q_d[3],self.q_d[0])

        #Angular error between the robot position and desired position
        self.theta_error = np.arctan2(Y_error, X_error) - self.theta

        #Y error between the robot position and desired position Absolute error X and Y-> relative error y
        self.y_error = -X_error * np.sin(self.theta) + Y_error * np.cos(self.theta)

        #X error between the robot position and desired position Absolute errors X and Y -> relative error x
        self.x_error = X_error * np.cos(self.theta) + Y_error * np.sin(self.theta)

        self.x_dot_ref = 1 * self.x_error - 0.2 * float(self.rel_vel[0])
        self.x_dot_ref = np.clip(self.x_dot_ref, -1, 1)

        self.w_z_ref = 30 * self.y_error
        self.w_z_ref = np.clip(self.w_z_ref, -2, 2)

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

        self.tau = [self.tau_l, self.tau_r, self.tau_l, self.tau_r]
        # Compute gravity forces for the current state
        # self.plant_context_ad.SetDiscreteState(self.q)
        
        # Update the output port = state
        discrete_state.get_mutable_vector().SetFromVector(self.tau)
