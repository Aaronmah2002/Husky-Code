import numpy as np
from pydrake.all import *
######################################################################################################
#                             ########Define PD+G Controller as a LeafSystem #######   
######################################################################################################

class Controller(LeafSystem):
    def __init__(self, plant, plant_context):
        super().__init__()
        #Creation of the arrays storing all the data for plotting
        self.init_data(self)

        self.nearestpoint = np.array([100000,100000])
        self.first_point_selected = False

        self.lookahead_dist = 0.5
        self.nearestdist = 10000
        self.nearest_ind = 0

        self.v_prev = 0
        self.w_prev = 0
        # Declare input ports for desired and current states
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=21)
        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=7)

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
            [np.cos(self.theta), np.sin(self.theta) * x_icr],
            [np.sin(self.theta), -np.cos(self.theta) * x_icr],
            [0, 0],
            [1, -L],
            [1, L],
            [1, -L],
            [1, L]])

        G_dot = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [-np.sin(self.theta), np.cos(self.theta) * x_icr],
            [np.cos(self.theta), np.sin(self.theta) * x_icr],
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

        y_dot = -np.sin(self.theta) * self.q[14] + np.cos(self.theta) * self.q[15]

        sgn_x1 = float(np.sign(eta[0] - L*eta[1]))
        sgn_x2 = float(np.sign(eta[0] + L*eta[1]))

        sgn_y1 = float(np.sign(y_dot + a*eta[1]))
        sgn_y3 = float(np.sign(y_dot - a*eta[1]))

        fr = 0.05
        mu = 0.5

        Rx_stat = fr * (m * g / 2) * (sgn_x1 + sgn_x2) 

        Fy_stat = mu * (m * g / (a + b)) * (b * sgn_y1 + a * sgn_y3)

        Mr_stat = 5*mu * (a * b * m * g / (a + b)) * (sgn_y1 - sgn_y3) + fr * (L * m * g / 2) * (sgn_x2 - sgn_x1)
       
        #print(f"friciton : {Fy_stat}  \n\n {4*F_fric_approx}")
        Rx = Rx_stat * eta[0][0]
        Fy = Fy_stat * eta[0][0]
        Mr_dyn = Mr_stat * self.q[13]

        F_visc = np.array([
            [0],
            [0],
            [Mr_dyn],
            [Rx * np.cos(self.theta) - Fy * np.sin(self.theta)],
            [Rx * np.sin(self.theta) + Fy * np.cos(self.theta)],
            [0],
            [0],
            [0],
            [0],
            [0]])


        error = np.array([self.q_d[4]- [self.q[4]], [self.q_d[5] - self.q[5]], [theta_d - self.theta]])

        '''
        if(pathFollowing and p == 0):
            point_to_follow = self.nearestpoint
            if(not self.first_point_selected):
                self.first_point_selected = True
                for point in range(self.path_steps+1):
                    dist = np.sqrt( (self.circ_path_x[point]-self.q[4])**2 + (self.circ_path_y[point]-self.q[5])**2 )

                    if dist < self.nearestdist:
                        self.nearestdist = dist
                        self.nearest_ind = point
                        self.nearestpoint = np.array([self.circ_path_x[point],self.circ_path_y[point]])
                        point_to_follow = self.nearestpoint

            if(np.sqrt( (self.nearestpoint[0]-self.q[4])**2 + (self.nearestpoint[1]-self.q[5])**2 ) < self.lookahead_dist):
                self.nearest_ind = self.nearest_ind + 1
                self.nearest_ind = self.nearest_ind % self.path_steps
                #print(nearest_ind)
                self.nearestpoint = np.array([self.circ_path_x[self.nearest_ind],self.circ_path_y[self.nearest_ind]])
                point_to_follow = self.nearestpoint

            X_error = point_to_follow[0] - self.q[4]
            Y_error = point_to_follow[1] - self.q[5]
            error = np.array([[X_error],[Y_error],[0]])

        if(pathFollowing and p == 1):
            point_to_follow = self.nearestpoint
            if(not self.first_point_selected):
                self.first_point_selected = True
                for point in range(self.path_steps+1):
                    dist = np.sqrt( (self.inf_path_x[point]-self.q[4])**2 + (self.inf_path_y[point]-self.q[5])**2 )

                    if dist < self.nearestdist:
                        self.nearestdist = dist
                        self.nearest_ind = point
                        self.nearestpoint = np.array([self.inf_path_x[point],self.inf_path_y[point]])
                        point_to_follow = self.nearestpoint

            if(np.sqrt( (self.nearestpoint[0]-self.q[4])**2 + (self.nearestpoint[1]-self.q[5])**2 ) < self.lookahead_dist):
                self.nearest_ind = self.nearest_ind + 1
                self.nearest_ind = self.nearest_ind % self.path_steps
                #print(nearest_ind)
                self.nearestpoint = np.array([self.inf_path_x[self.nearest_ind],self.inf_path_y[self.nearest_ind]])
                point_to_follow = self.nearestpoint
            
            X_error = point_to_follow[0] - self.q[4]
            Y_error = point_to_follow[1] - self.q[5]
            error = np.array([[X_error],[Y_error],[0]])
        '''

        rel_error = self.abs_to_rel @ error

        a_err = np.arctan2(rel_error[1],rel_error[0])

        rel_error = self.abs_to_rel @ error

        #print(f"s : {rel_error}  {a_err}  {max(abs(rel_error[1]), abs(2*a_err))* np.sign(rel_error[1])}\n\n\n")

        #rel_error[1] = max(abs(rel_error[1]), abs(2*a_err))* np.sign(rel_error[1])

        K_p_theta_variable = 40 / (0.1 + np.exp(70*(rel_error[0][0]**2 + rel_error[1][0]**2)))*0

        u = np.array([[2, 0, 0],[0, 250, K_p_theta_variable]]) @ rel_error - np.array([[3, 0],[0, 200]]) @ eta
        #u = np.clip(u , -50, 50)
        ax = np.clip(u[0], -3, 3)
        w_z_dot = np.clip(u[1], -20,20)
        #u = [ax,w_z_dot]
        M2 = np.transpose(G) @ M_mat @ G
        M3 = np.transpose(G) @ M_mat @ G_dot
        tau = np.linalg.pinv(np.transpose(G) @ E_mat) @ (M2 @ u + M3 @ eta + np.transpose(G) @ (g_mat))
        
        tau = 2*tau
        #tau = np.clip(tau, -30, 30)

        
        #print(f"t : {sgn_y1 - sgn_y3} \n\n\n")
        # Update the output port = state
        discrete_state.get_mutable_vector().SetFromVector(tau)
