
import time
from pydrake.all import *



####################################
#     Create system diagram
####################################




def create_system_model(plant, scene_graph):
    """
    Add the Panda arm model to the plant and configure contact properties.
    
    Args:
        plant: The MultibodyPlant object to which the Panda arm model will be added.
        scene_graph: The SceneGraph object for visualization.
    
    Returns:
        Tuple containing the updated plant and scene_graph.
    """
    urdf = "file:///home/aaron/drake_brubotics/models/descriptions/robots/husky_description/urdf/husky.urdf"
    sdf = "file:///home/aaron/drake_brubotics/models/objects&scenes/scenes/floor.sdf"
    arm = Parser(plant).AddModelsFromUrl(urdf)
    floor = Parser(plant).AddModelsFromUrl(sdf)
    contact_model = ContactModel.kPoint  # Options: Hydroelastic, Point, or HydroelasticWithFallback
    discrete_solver = DiscreteContactApproximation.kSap # Options:kTamsi, kSap, kLagged, kSimilar
    #mesh_type = HydroelasticContactRepresentation.kTriangle  # Options: Triangle or Polygon    
    #plant.set_contact_surface_representation(mesh_type)
    plant.set_contact_model(contact_model)
    plant.set_discrete_contact_approximation(discrete_solver)
    plant.Finalize()
    return plant, scene_graph



######################################################################################################
#                         #########  explicit_reference_governor  ##########                       #
######################################################################################################
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
        # Plant Configuration parameters
        time_step = 0.01
        # PLant for simulation
        self.builder_pred = DiagramBuilder()
        self.plant_pred, scene_graph= AddMultibodyPlantSceneGraph(self.builder_pred, time_step)
        self.plant_pred, scene_graph = create_system_model(self.plant_pred, scene_graph) 

        # Finalize the diagram
        self.diagram = self.builder_pred.Build()                           
        self.diagram_context = self.diagram.CreateDefaultContext()    
        self.plant_context =  self.diagram.GetMutableSubsystemContext(self.plant_pred, self.diagram_context)
                
        self.eta_ = 0.005 
        self.zeta_q_ = 0.15 # range of influence for the repulsion field. 
        self.delta_q_ = 0.1 # threshold for when the repulsion effect starts to take place.
        self.dt_ = 0.01  # Sampling time for the refrence governor

        # Controller gains
        self.Kp_ = [120.0, 120.0, 120.0, 100.0]
        self.Kd_ = [8.0, 8.0, 8.0, 5.0]

        # Prediction parameters
        prediction_dt_ = time_step# 0.01  # Time step for predictions
        prediction_horizon_ = 0.2  # Total prediction horizon
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

        self.num_positions =  self.plant_pred.num_positions()
        self.num_velocities = self.plant_pred.num_velocities()
        self.num_tau = self.plant_pred.num_actuated_dofs()

        # Prediction lists for joint positions, velocities, and torques
        self.q_pred_list_ = np.zeros((self.num_positions, self.num_pred_samples_ + 1))
        self.dq_pred_list_ = np.zeros((self.num_velocities, self.num_pred_samples_ + 1))
        self.tau_pred_list_ = np.zeros((self.num_tau, self.num_pred_samples_ + 1))
        
        # Limits for joint angles, velocities, and torques
        #
        self.limit_q_min_ = np.array([-10, -10, -10, -10, -10, -3, -1])
        self.limit_q_max_ = np.array([10, 10, 10, 10, 10, 3, 3])
        self.limit_tau_ = np.array([20,20,20,20])
        self.limit_dq_ = np.array([2.1750, 2.1750, 2.1750, 2.1750])
        #self.limit_dp_EE_ = [1.7, 2.5]  # Translation and rotation limits for the end effector

        


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
        DSM_ = self.trajectoryBasedDSM(q, dq, tau, q_v)

        q_v_new = q_v + DSM_ * rho_ * self.dt_ 
        
        if DSM_ > 0:
          q_v_new = q_v + DSM_ * rho_ * self.dt_ 
        else:
          q_v_new = q_v + np.min([np.linalg.norm(DSM_ * rho_ * self.dt_), np.linalg.norm(q_r - q_v)]) * DSM_ * rho_ / max(np.linalg.norm(DSM_ * rho_), self.eta_)
        
        return q_v_new

    def navigationField(self, q_r, q_v):
        """
        Compute the navigation field based on attraction and repulsion forces.
        """
        rho_att = np.zeros(self.num_positions) 
        rho_rep_q = np.zeros(self.num_positions) 
        rho = np.zeros(self.num_positions) 

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
        return rho

    def trajectoryBasedDSM(self, q, dq, tau, q_v):
      """
      Compute the Dynamic Safety Margin (DSM) based on trajectory predictions.
      """
      # Get trajectory predictions and save predicted q, dq, and tau in lists
      start_time = time.time()
      self.trajectoryPredictions(np.concatenate((q, dq)), tau, q_v)
      #print(f"TIme taken to predict  x= {(time.time() - start_time)*1000} ms")

      # Compute DSMs
      DSM_tau_ = self.dsmTau()
      DSM_q_ = self.dsmQ()
      DSM_dq_ = self.dsmDq()

      # Find the minimum among the DSMs
      DSM = min(DSM_tau_,DSM_dq_)
      DSM = min(DSM,DSM_q_)
      # DSM = min(DSM,DSM_terminal_energy_)

      DSM = max(DSM, 0.0)
    # Print DSMs
    #   print(f"DSM_tau_: {DSM_tau_}")
    #   print(f"DSM_q_: {DSM_q_}")
    #   print(f"DSM_dq_: {DSM_dq_}")
    #   print(f"DSM_dp_EE_: {DSM_dp_EE_}")
    #   print(f"DSM_final: {DSM}")
      
      return DSM
    
    def compute_tau(self,plant,plant_context,q,q_d):  #q_d must be the applied ref computed by ERG
        """
        robot_rot_quaternion = self.q[0:4]
        robot_pos = self.q[4:7]
        robot_wheel_rot = self.q[7:11]
        robot_ang_velocity = self.q[11:14]
        robot_speed = self.q[14:17] #Absolute values
        robot_wheel_ang_velocity = self.q[17:21]
        """


        #Computation of rotation angle
        theta = 2 * np.arctan2(q[3],[0])
        theta_d = 2 * np.arctan2(q_d[3],q_d[0])
        L = 0.670/2
        r = 0.165

        M_mat = plant.CalcMassMatrix(plant_context) # 10x10 matrix
        C_mat = plant.CalcBiasTerm(plant_context).reshape(-1, 1) #1x10 matrix
        #E_mat = self.plant.MakeActuationMatrix() #10x4 matrix
        g_mat = plant.CalcGravityGeneralizedForces(plant_context).reshape(-1, 1)
        
        E_mat = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-L/r, L/r, -L/r, L/r],
            [1/r*np.cos(theta), 1/r*np.cos(theta), 1/r*np.cos(theta), 1/r*np.cos(theta)],
            [1/r*np.sin(theta), 1/r*np.sin(theta), 1/r*np.sin(theta), 1/r*np.sin(theta)],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

        G = np.array([
            [0, 0],
            [0, 0],
            [0, 1],
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, 0],
            [1/r, -L/r],
            [1/r, L/r],
            [1/r, -L/r],
            [1/r, L/r]])

        G_dot = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [-np.sin(theta), 0],
            [np.cos(theta), 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]])

        eta = np.array([
            [np.cos(theta) * q[14] + np.sin(theta) * q[15]],
            [q[13]]])

        u = np.array([
            [1],
            [1]]) #Pseudo acceleration vector (a_x, w_z dot)


        abs_to_rel = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta),np.cos(theta),0],
            [0,0,1]])

        a=0.5
        b=0.5
        m = 56.5
        g = 9.81

        y_dot = -np.sin(theta) * q[14] + np.cos(theta) * q[15]

        sgn_x1 = float(np.sign(eta[0] - L*eta[1]))
        sgn_x2 = float(np.sign(eta[0] + L*eta[1]))

        sgn_y1 = float(np.sign(y_dot + a*eta[1]))
        sgn_y3 = float(np.sign(y_dot - a*eta[1]))

        fr = 0.05
        mu = 0.5

        Rx_stat = fr * (m * g / 2) * (sgn_x1 + sgn_x2) 

        Fy_stat = mu * (m * g / (a + b)) * (b * sgn_y1 + a * sgn_y3)

        Mr_stat = mu * (a * b * m * g / (a + b)) * (sgn_y1 - sgn_y3) + fr * (L * m * g / 2) * (sgn_x2 - sgn_x1)
       
        Rx = Rx_stat * eta[0][0]
        Fy = Fy_stat * eta[0][0]
        Mr_dyn = Mr_stat * q[13]

        F_visc = np.array([
            [0],
            [0],
            [Mr_dyn],
            [Rx * np.cos(theta) - Fy * np.sin(theta)],
            [Rx * np.sin(theta) + Fy * np.cos(theta)],
            [0],
            [0],
            [0],
            [0],
            [0]])


        error = np.array([q_d[4]- [q[4]], [q_d[5] - q[5]], [theta_d - theta]])
        rel_error = abs_to_rel @ error

        a_err = np.arctan2(rel_error[1],rel_error[0])

        rel_error = abs_to_rel @ error

        #print(f"s : {rel_error}  {a_err}  {rel_error[1] * a_err}\n\n\n")

        #rel_error[1] = rel_error[1] * a_err

        K_p_theta_variable = 40 / (0.1 + np.exp(70*(rel_error[0][0]**2 + rel_error[1][0]**2)))*0

        u = np.array([[1.5, 0, 0],[0, 70, K_p_theta_variable]]) @ rel_error - np.array([[4, 0],[0, 100]]) @ eta


        
        Mr = np.abs(Mr_stat) * np.sign(u[1][0])
        if(np.abs(u[1]) < 0.1):
            Mr = 0
        F_stat = np.array([
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
        #print(F_stat)
        M2 = np.transpose(G) @ M_mat @ G
        M3 = np.transpose(G) @ M_mat @ G_dot
        tau = np.linalg.pinv(np.transpose(G) @ E_mat) @ (M2 @ u + M3 @ eta + np.transpose(G) @ (g_mat + C_mat + F_stat + F_visc))
        
        #print(f"{rel_error} \n\n\n")
        
        #print(f"t : {sgn_y1 - sgn_y3} \n\n\n")
        return tau 
    
    def compute_tau_u(self,q,q_d):
        """
        robot_rot_quaternion = self.q[0:4]
        robot_pos = self.q[4:7]
        robot_wheel_rot = self.q[7:11]
        robot_ang_velocity = self.q[11:14]
        robot_speed = self.q[14:17] #Absolute values
        robot_wheel_ang_velocity = self.q[17:21]
        """
        abs_to_rel = []
        rel_to_abs = []
        #Initialisation of the controller parameters

        Kp_ = [3.2]
        # Evaluate the input ports
        

        #Computation of rotation angle
        theta = 2 * np.arctan2(q[3],q[0])
        
        #Computation of reltive to absolute trnaformation matrix
        rel_to_abs = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),np.cos(theta),0],
            [0,0,1]])
        
        #Computation of absolute to reltive tranformation matrix
        abs_to_rel = rel_to_abs.T
        
        #Necessary absolute velocities : X*, Y*, W_z
        abs_vel = np.array([
            [q[14]],
            [q[15]],
            [q[13]]
        ])

        r = 0.165 #Wheel radius
        d = 0.613 #Robot width (distance between wheels)

        inv_kin_mat = np.array([
            [1/r, d/(2*r)],
            [1/r, -d/(2*r)],
        ])


        #Necessary relative velocities : x*, x*, w_z
        rel_vel = abs_to_rel @ abs_vel

        #Error computation in absolutevalues
        X_error = q_d[4] - q[4]
        Y_error = q_d[5] - q[5]

        #Final angle for the robot in absolute/relative
        ref_angle = 2 * np.arctan2(q_d[3],q_d[0])

        #Angular error between the robot position and desired position
        theta_error = np.arctan2(Y_error, X_error) - theta

        #Y error between the robot position and desired position Absolute error X and Y-> relative error y
        y_error = -X_error * np.sin(theta) + Y_error * np.cos(theta)

        #X error between the robot position and desired position Absolute errors X and Y -> relative error x
        x_error = X_error * np.cos(theta) + Y_error * np.sin(theta)

        x_dot_ref = 1 * x_error - 0.2 * float(rel_vel[0])

        w_z_ref = 30 * y_error

        #=======================================
        #x_dot_ref = 1
        #w_z_ref = 1/2*np.sin(context.get_time()/2)

        #Computation of the reference speeds of the wheels
        w_ref = inv_kin_mat @ np.array([[x_dot_ref],[w_z_ref]])
        
        #Regulation of
        tau_l = 20*Kp_[0]*(w_ref[1] - q[17])
        tau_r = 20*Kp_[0]*(w_ref[0] - q[18])

        #self.tau_l = np.clip(self.tau_l , -20, 20)
        #self.tau_r = np.clip(self.tau_r , -20, 20)

        tau = np.zeros(4)
        for i in range(4):
            if i%2==0:
                tau[i]=tau_l
            else:
                tau[i]=tau_r
        # Compute gravity forces for the current state
        # self.plant_context_ad.SetDiscreteState(self.q)
        
        # Update the output port = state
        return tau
    
    def trajectoryPredictions(self, state, tau, q_v):
        """
        Predict joint positions, velocities, and torques over the prediction horizon.
        """
        q_pred, dq_pred = state[:self.num_positions], state[self.num_positions:]
        tau_pred = tau

        # Initialize lists to store predicted states
        q_pred_traj = [q_pred.copy()]
        dq_pred_traj = [dq_pred.copy()]
        tau_pred_traj = [tau_pred.copy()]

        # Initialize lists to store predicted states
        self.q_pred_list_[:, 0] = q_pred
        self.dq_pred_list_[:, 0] = dq_pred
        self.tau_pred_list_[:, 0] = tau_pred
        
        for k in range(self.num_pred_samples_):
            
            # Compute tau_pred
            #tau_pred =  self.Kd_ * dq_pred[self.num_velocities-4:]  
            #tau_p = self.compute_tau(self.plant_pred,self.plant_context,state,q_v) 
            tau_pred=self.compute_tau_u(state,q_v) # works fine -> from the first stabilizing controller
            

            # Solve for x[k+1] using the computed tau_pred
            state_pred = self.calc_dynamics(np.concatenate((q_pred, dq_pred)), tau_pred)  # Adjust this based on your calculation method
            q_pred = state_pred[:self.num_positions]
            dq_pred = state_pred[self.num_positions:]

            # Store predicted states
            q_pred_traj.append(q_pred.copy())
            dq_pred_traj.append(dq_pred.copy())
            tau_pred_traj.append(tau_pred.copy())
            # print(q_v)

            # Add q, dq, and tau to prediction list
            self.q_pred_list_[:, k + 1] = q_pred
            self.dq_pred_list_[:, k + 1] = dq_pred
            self.tau_pred_list_[:, k + 1] = tau_pred
            # print(f"TIme taken to predict one sample = {(time.time() - start_time)*1000} ms")

        # Convert lists to arrays for plotting
        q_pred_traj = np.array(q_pred_traj)
        dq_pred_traj = np.array(dq_pred_traj)
        tau_pred_traj = np.array(tau_pred_traj)

        # # Define total prediction time T and time step size dt
        # dt = 0.2 / self.num_pred_samples_  # Time step size
        # time_steps = np.linspace(0, 0.2, self.num_pred_samples_ + 1)  # Time steps array

        # Assuming time_steps, q_pred_traj, dq_pred_traj, tau_pred_traj, q_desired_traj, dq_desired_traj, tau_desired_traj are defined

        # plt.figure()

        # Plot predicted states
        # plt.plot(time_steps, q_pred_traj, label='Predicted q', linestyle='-', color='b')
        # plt.plot(time_steps, dq_pred_traj, label='Predicted dq', linestyle='--', color='b')
        # plt.plot(time_steps, tau_pred_traj, label='Predicted tau', linestyle=':', color='b')

        # plt.xlabel('Time')
        # plt.ylabel('Angle Position / Velocity / Torque')
        # plt.legend()
        # plt.title('Predicted vs Desired States')
        # plt.show()

        # Plot predicted states vs desired states for each joint
        # num_joints = self.num_positions
        # fig, axs = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints))

        # for i in range(num_joints):
        #     axs[i].plot(time_steps, q_pred_traj[:, i], label='Predicted q', linestyle='-', color='b')
        #     axs[i].plot(time_steps, q_v[i] * np.ones_like(time_steps), label='Desired q', linestyle='--', color='r')
        #     axs[i].set_xlabel('Time')
        #     axs[i].set_ylabel(f'Joint {i+1} Position')
        #     axs[i].legend()
        #     axs[i].set_title(f'Predicted vs Desired Position for Joint {i+1}')

        # plt.tight_layout()
        # plt.show()

    # Calculate system dynamics
    '''
    TamsiSolver uses the Transition-Aware Modified Semi-Implicit (TAMSI) method, [Castro et al., 2019], 
    to solve the equations below for mechanical systems in contact with regularized friction:
                q̇ = N(q) v
    (1)  M(q) v̇ = τ + Jₙᵀ(q) fₙ(q, v) + Jₜᵀ(q) fₜ(q, v)

    where:
    - v ∈ ℝⁿᵛ: Vector of generalized velocities
    - M(q) ∈ ℝⁿᵛˣⁿᵛ: Mass matrix
    - Jₙ(q) ∈ ℝⁿᶜˣⁿᵛ : Jacobian of normal separation velocities
    - Jₜ(q) ∈ ℝ²ⁿᶜˣⁿᵛ: Jacobian of tangent velocities
    - fₙ ∈ ℝⁿᶜ: Vector of normal contact forces
    - fₜ ∈ ℝ²ⁿᶜ: Vector of tangent friction forces
    - τ ∈ ℝⁿᵛ: Vector of generalized forces containing all other applied forces (e.g., Coriolis, gyroscopic terms, actuator forces, etc.) but contact forces.

    This solver assumes a compliant law for the normal forces fₙ(q, v) and therefore the functional dependence of fₙ(q, v) with q and v is stated explicitly.

    Since TamsiSolver uses regularized friction, we explicitly emphasize the functional dependence of fₜ(q, v) with the generalized velocities. 
    The functional dependence of fₜ(q, v) with the generalized positions stems from its direct dependence with the normal forces fₙ(q, v).
    '''
    def calc_dynamics(self, x, u):
        #assert self.diagram.IsDifferenceEquationSystem()[0], "must be a discrete-time system"
        """
        Calculate the next state given the current state x and control input u.
        
        Args:
            x: Current state vector.
            u: Control input vector.
        
        Returns:
            The next state vector.
        """
        # Autodiff copy of the system for computing dynamics gradients
        self.plant_context.SetDiscreteState(x)
        self.plant_pred.get_actuation_input_port().FixValue(self.plant_context, u)
        state = self.diagram_context.get_discrete_state()
        self.diagram.CalcForcedDiscreteVariableUpdate(self.diagram_context, state)
        x_next = state.get_vector().value().flatten()
        # print(x_next)
        return x_next

    def dsmTau(self):
        """
        Compute the DSM for joint torques.
        """
        for k in range(self.tau_pred_list_.shape[1]):  # number of prediction samples + 1
            tau_pred = self.tau_pred_list_[:, k]
            DSM_tau_temp = self.distanceTau(tau_pred) - self.robust_delta_tau_
            if k == 0:
                DSM_tau = DSM_tau_temp
            else:
                DSM_tau = min(DSM_tau, DSM_tau_temp)

        DSM_tau = self.kappa_tau_ * DSM_tau
        return DSM_tau

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
        # wrong because taking the first 4
        for i in range(4):
            distance_dotq_lowerlimit = dotq_pred[i] - (-self.limit_dq_[i])
            distance_dotq_upperlimit = self.limit_dq_[i] - dotq_pred[i]
            distance_dotq_temp = min(distance_dotq_lowerlimit, distance_dotq_upperlimit)
            if i == 0:
                distance_dotq = distance_dotq_temp
            else:
                distance_dotq = min(distance_dotq, distance_dotq_temp)
        return distance_dotq
      
    def distanceDpEE(self, q_pred, dotq_pred): ### Fix
        endeffector_jacobian = np.zeros((6, self.num_positions))  # Example initialization
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