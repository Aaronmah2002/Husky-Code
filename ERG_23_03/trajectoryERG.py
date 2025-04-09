
import time
import csv
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
    obs = "file:///home/aaron/drake_brubotics/models/objects&scenes/objects/cylinder.sdf"
    arm = Parser(plant).AddModelsFromUrl(urdf)
    floor = Parser(plant).AddModelsFromUrl(sdf)
    #obstacle = Parser(plant).AddModelsFromUrl(obs)
    contact_model = ContactModel.kHydroelasticWithFallback  # Options: Hydroelastic, Point, or HydroelasticWithFallback
    discrete_solver = DiscreteContactApproximation.kTamsi # Options:kTamsi, kSap, kLagged, kSimilar
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
    def __init__(self,time_step, robust_delta_tau_, kappa_tau_, 
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


        # Prediction parameters
        prediction_dt_ = time_step# 0.01  # Time step for predictions
        prediction_horizon_ = 0.02  # Total prediction horizon
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

        # self.num_positions =  self.plant_pred.num_positions()
        # self.num_velocities = self.plant_pred.num_velocities()
        self.num_tau = self.plant_pred.num_actuated_dofs()


        
        husky_instance = self.plant_pred.GetModelInstanceByName("husky")
        self.num_positions = self.plant_pred.num_positions(husky_instance)
        self.num_velocities = self.plant_pred.num_velocities(husky_instance)

        # obs_instance = self.plant_pred.GetModelInstanceByName("cylinder")
        # self.obsp = self.plant_pred.num_positions(obs_instance)
        # self.obsv = self.plant_pred.num_velocities(obs_instance)

        # self.obsvecpos=np.zeros(self.obsp)
        # self.obsvecvel=np.zeros(self.obsv)

        self.f_x = [0,0,0,0]
        self.f_y = [0,0,0,0]
        self.F_abs = np.zeros((4, 3))
        # Prediction lists for joint positions, velocities, and torques
        self.q_pred_list_ = np.zeros((self.num_positions, self.num_pred_samples_ + 1))
        self.dq_pred_list_ = np.zeros((self.num_velocities, self.num_pred_samples_ + 1))
        self.tau_pred_list_ = np.zeros((self.num_tau, self.num_pred_samples_ + 1))



        

        # Limits for joint angles, velocities, and torques
        #
        self.limit_q_min_ = np.array([-10, -10]) #only limiting XY on the grid
        self.limit_q_max_ = np.array([10, 10])
        self.limit_tau_ = np.array([16.04,16.04,16.04,16.04])
        self.limit_dq_ = np.array([2, 2, 2, 2])
        #self.limit_dp_EE_ = [1.7, 2.5]  # Translation and rotation limits for the end effector


        self.DSM_tau_list = []
        self.DSM_q_list = []
        self.DSM_list = []
        self.DSM_qd_list = []
        self.navfield_list= []
        self.rep_field_list = []
        self.att_field_list = []
        self.epsilon = 1
        


    def get_qv(self, q, dq, tau, q_r, q_v,q_o):
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
        rho_ = self.navigationField(q_r, q_v,q_o)
        self.navfield_list.append(rho_)

        DSM_ = self.trajectoryBasedDSM(q, dq, tau, q_v)

        q_v_new = q_v + DSM_ * rho_ * self.dt_ 
        
        if DSM_ > 0:
          q_v_new = q_v + DSM_ * rho_ * self.dt_ 
        else:
          q_v_new = q_v + np.min([np.linalg.norm(DSM_ * rho_ * self.dt_), np.linalg.norm(q_r - q_v)]) * DSM_ * rho_ / max(np.linalg.norm(DSM_ * rho_), self.eta_)
        
        return q_v_new
    
    def barrierfct(self,d,epsilon):
        if d <= epsilon :
            b = 1/d**2*(epsilon - d)
        else :
            b=0
        return b
    def navigationField(self, q_r, q_v, q_o): #obs_1_position is a 7q vector
        """
        Compute the navigation field based on attraction and repulsion forces.
        """
        rho_att = np.zeros(self.num_positions) 
        rho_rep_q = np.zeros(self.num_positions) #attention  position of obs only 7 so concat with [0,0,0,0]
        rho = np.zeros(self.num_positions) 


        # Attraction field
        rho_att = (q_r - q_v) / max(np.linalg.norm(q_r - q_v), self.eta_)
        # print(f"norm rho_att = {np.linalg.norm(rho_att)}")
        # print(f"rho_att = \n {rho_att}")
        #print(rho_rep_q)
        distance = q_v[:7] - q_o[:7]
        rho_rep_q_1 = distance/np.linalg.norm(distance)* self.barrierfct(d=np.linalg.norm(distance),epsilon=self.epsilon)
        '''
        # Joint angle repulsion field (q)
        for i in range(7):
            rho_rep_q[i] = max((self.zeta_q_ - abs(q_v[i] - self.limit_q_min_[i])) / (self.zeta_q_ - self.delta_q_), 0.0) - \
                           max((self.zeta_q_ - abs(q_v[i] - self.limit_q_max_[i])) / (self.zeta_q_ - self.delta_q_), 0.0)
        # print(f"norm rho_rep_q = {np.linalg.norm(rho_rep_q)}")
        # print(f"rho_rep_q = \n {rho_rep_q}")
        '''


        rho_rep_q= rho_rep_q_1
        self.rep_field_list.append(rho_rep_q)
        self.att_field_list.append(rho_att)
        # print('rho_rep_concat ',rho_rep_q)
        # print('rho_att ',rho_att)
        # Total navigation field
        rho = rho_att + rho_rep_q
        return rho


    def compute_tau_u(self,q,q_d):

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
    def compute_tau_dyn(self,state,q_v):

        #Computation of rotation angle
        theta = 2 * np.arctan2(state[3],state[0])
        theta_d = 2 * np.arctan2(q_v[3],q_v[0])
        L = 0.670/2
        r = 0.165

        eta = np.array([
            [np.cos(theta) * state[14] + np.sin(theta) * state[15]],
            [state[13]]])

        M_mat = self.plant_pred.CalcMassMatrix(self.plant_context)[:10,:10] # 10x10 matrix
        C_mat = self.plant_pred.CalcBiasTerm(self.plant_context).reshape(-1, 1)[:10] #1x10 matrix
        g_mat = self.plant_pred.CalcGravityGeneralizedForces(self.plant_context)[:10].reshape(-1, 1)
        
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
            [np.cos(theta), np.sin(theta) * 0.2],
            [np.sin(theta), -np.cos(theta) * 0.2],
            [0, 0],
            [1, -L],
            [1, L],
            [1, -L],
            [1, L]])

        G_dot = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [-np.sin(theta), np.cos(theta) * 0.2],
            [np.cos(theta), np.sin(theta) * 0.2],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]])


        u = np.array([
            [1],
            [1]]) #Pseudo acceleration vector (a_x, w_z dot)


        abs_to_rel = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta),np.cos(theta),0],
            [0,0,1]])

        a=0.5
        b=0.5
        m = 125.5
        g = 9.81

        """
        contact_results = self.contact_results_port.Eval(self.plant_context)
        
        for i in range(contact_results.num_point_pair_contacts()):
            if(i<4):
                contact_info = contact_results.point_pair_contact_info(i) 
                contact_point = contact_info.contact_point().tolist()
                contact_force = contact_info.contact_force().tolist()
                self.F_abs[i] = contact_force
                self.f_x[i] = contact_force[0] #X (absolute) forces matrix F_L, F_R, B_L, B_R
                self.f_y[i] = contact_force[1] #Y (absolute) forces matrix F_L, F_R, B_L, B_R
        """
        F_abs = np.transpose(self.F_abs) #Friction force in abolute coordinates 3x4 Matrix

        F_rel = abs_to_rel @ F_abs #Friction force in relative coordinates
        
        error = np.array([[q_v[4]- state[4]], [q_v[5] - state[5]], [np.sin(theta_d - theta)]])

        rel_error = abs_to_rel @ error

        u = np.array([[4, 0, 0],[0, 60, 0]]) @ rel_error - np.array([[7, 0],[0, 70]]) @ eta 


        Rx = F_rel[0][0] + F_rel[0][1] + F_rel[0][2] + F_rel[0][3]
        Fy = F_rel[1][0] + F_rel[1][1] + F_rel[1][2] + F_rel[1][3]
        M_dyn = (F_rel[0][1] + F_rel[0][3] - F_rel[0][0] - F_rel[0][2])*0.26 + (F_rel[1][2] + F_rel[1][3] - F_rel[1][0] - F_rel[1][1]) * 0.29
        Rx = -Rx 
        Fy = -Fy 
        M_dyn = -M_dyn*0.7
        
        F_visc = np.array([
            [0],
            [0],
            [M_dyn],
            [Rx * np.cos(theta) - Fy * np.sin(theta)],
            [Rx * np.sin(theta) + Fy * np.cos(theta)],
            [0],
            [0],
            [0],
            [0],
            [0]])

        
        M2 = np.transpose(G) @ M_mat @ G
        M3 = np.transpose(G) @ M_mat @ G_dot

        tau = np.linalg.pinv(np.transpose(G) @ E_mat) @ (M2 @ u + M3 @ eta + np.transpose(G) @ (g_mat + F_visc + C_mat))
        tau = np.transpose(tau)
        return tau[0]

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
      DSM = min(DSM_tau_,DSM_q_)
      DSM = min(DSM,DSM_q_)
      # DSM = min(DSM,DSM_terminal_energy_)

      DSM = max(DSM, 0.0)
      self.DSM_list.append(DSM)
      #print(DSM)
      #print(f"DSM_final: {DSM}")
    # Print DSMs
    #   print(f"DSM_tau_: {DSM_tau_}")
    #   print(f"DSM_q_: {DSM_q_}")
    #   print(f"DSM_dq_: {DSM_dq_}")
    
      
      return DSM
    
    
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
            #tau_pred1=self.compute_tau_u(state,q_v) # works fine -> from the first stabilizing controller
            tau_pred=self.compute_tau_dyn(state,q_v)

            # Solve for x[k+1] using the computed tau_pred
            state_pred = self.calc_dynamics(np.concatenate((q_pred,dq_pred)), tau_pred)  # Adjusted to match the state size and don't take into account the cylinder state
            q_pred = state_pred[:self.num_positions]
            dq_pred = state_pred[:self.num_velocities]

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
        self.DSM_tau_list.append(DSM_tau)
        return DSM_tau

    def dsmQ(self):
        """
        Compute the DSM for joint positions.
        """
        ##for k in range(self.q_pred_list_.shape[1]):  # number of prediction samples + 1
        for k in range(2):
            q_pred = self.q_pred_list_[3:5, k]  #[3:5] just taking the XY position in the q_pred 11 elements vector of positions
            DSM_q_temp = self.distanceQ(q_pred) - self.robust_delta_q_
            if k == 0:
                DSM_q = DSM_q_temp
            else:
                DSM_q = min(DSM_q, DSM_q_temp)

        DSM_q = self.kappa_q_ * DSM_q
        self.DSM_q_list.append(DSM_q)
        #print(DSM_q)
        return DSM_q

    def dsmDq(self):
        """
        Compute the DSM for joint velocities.
        """

        for k in range(self.dq_pred_list_.shape[0]):  # number of prediction samples + 1
            dotq_pred = self.dq_pred_list_[:, k]
            DSM_dotq_temp = self.distanceDq(dotq_pred) - self.robust_delta_dq_
            if k == 0:
                DSM_dotq = DSM_dotq_temp
            else:
                DSM_dotq = min(DSM_dotq, DSM_dotq_temp)

        DSM_dotq = self.kappa_dq_ * DSM_dotq
        self.DSM_qd_list.append(DSM_dotq)
        return DSM_dotq

    
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
        ##for i in range(7):  # include all joints
        for i in range(2): #so juste in order to impose constraints on X and Y
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
        dotq_pred=dotq_pred[2:6]
        for i in range(4):
            distance_dotq_lowerlimit = dotq_pred[i] - (-self.limit_dq_[i])
            distance_dotq_upperlimit = self.limit_dq_[i] - dotq_pred[i]
            distance_dotq_temp = min(distance_dotq_lowerlimit, distance_dotq_upperlimit)
            if i == 0:
                distance_dotq = distance_dotq_temp
            else:
                distance_dotq = min(distance_dotq, distance_dotq_temp)
        return distance_dotq
