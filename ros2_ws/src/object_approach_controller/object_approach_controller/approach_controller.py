import casadi as ca
from casadi import Callback
import mujoco as mj
import mujoco.viewer
# import pinocchio as pin
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import String

"""
Node that subscribes to teleop joint angles and computes collision-free joint angles.
"""
class ApproachController(Node):

    def __init__(self, mj_model, mj_data):
        super().__init__('teleop_joint_angle_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        
        self.mj_model = mj_model
        self.mj_data = mj_data      
        mj.mj_forward(self.mj_model, self.mj_data)

        self.subscription  # prevent unused variable warning
        self.teleop_joint_angles = np.zeros(mj_model.nq)

        # Get relevant IDs for collision detection
        right_index_distal_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_GEOM, "right_index_distal_geom")
        right_thumb_distal_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_GEOM, "right_thumb_distal_geom")
        obj_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_GEOM, "obj1_geom")

        # print(right_index_distal_id)
        # print(right_thumb_distal_id)
        # print(obj_id)

 
        # Launch MJCF viewer
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            print("Simulation started. Close the window to stop.")

            while viewer.is_running():
                mj.mj_step(self.mj_model, self.mj_data)
                # print(self.get_signed_distance(right_index_distal_id, obj_id)) 
                dist = self.get_signed_distance(right_index_distal_id, obj_id)
                print(f"Dist: {dist:8.4f}m", end="\r")

                collision_free_joint_angles = self.get_collision_free_joint_angles(self.mj_data.qpos[:self.mj_model.nq])
                # print(f"\rCollision Free Joint Angles: {collision_free_joint_angles}", end="", flush=True)
                q_str = np.array2string(collision_free_joint_angles, 
                        formatter={'float_kind':lambda x: f"{x:7.3f}"},
                        separator=', ')

                print(f"\rCollision Free Joint Angles: {q_str}", end="", flush=True)
                viewer.sync()



    def get_signed_distance(self, geom1_id, geom2_id):
        dist_max = 1000.0
        fromto = np.zeros(6, dtype=np.float64)


        dist = mj.mj_geomDistance(self.mj_model, self.mj_data, geom1_id, geom2_id, dist_max, fromto)
        return dist

    def listener_callback(self, msg):
        """_summary_

        Args:
            joint_angles (ndarray): joint angles parsed from teleop
        """
        self.teleop_joint_angles = msg.data.joint_angles

    def get_collision_free_joint_angles(self, q_ref):
        """
        Solve for a joint configuration that prevents collision between finger bodies and object.
        
        :param self: Description
        :param q_ref: Description
        """
        assert len(q_ref) == self.mj_model.nq

        right_index_distal_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_GEOM, "right_index_distal_geom")
        right_thumb_distal_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_GEOM, "right_thumb_distal_geom")
        obj_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_GEOM, "obj1_geom")


        # Opti variables
        opti = ca.Opti()

        nq = self.mj_model.nq

        # q = ca.SX.sym("q", nq)
        q = opti.variable(nq)
        # q_ref = np.zeros(nq) # TODO: set initial guess from teleop instead of hardcode

        # qpos = np.array([-0.06, -0.5381 -0.872871 -0.9371 -0.971053 0.690403 0.902978 0.940307 -0.0828766 -0.750108 2.01172e-13 -4.2624e-05 3.59805e-16 3.43339e-17 1])

        Q_diag = np.ones(nq) 
        Q = ca.diag(Q_diag) 

        cost = ca.bilin(Q, q-q_ref, q-q_ref)
        opti.minimize(cost)

        # opti.subject_to(opti.bounded(self.mj_model.jnt_range[:,0], q, self.mj_model.jnt_range[:,1]))
        call_opts = {"enable_fd": True}
        dist_func = SignedDistanceConstraint("dist_right_index_obj", self.mj_model, self.mj_data, right_index_distal_id, obj_id, call_opts)
        eps = 1e-1
        opti.subject_to(dist_func(q) >= eps) # Constraint to prevent SDF from exceeding certain size


        opts = {
            "print_time": False,
            "ipopt": {
                "jacobian_approximation": "finite-difference-values",
                "hessian_approximation": "limited-memory",
                "print_level": 0,
                "sb": "yes",          
                "max_iter": 1000,
            },
            "ad_weight_sp": 0
        }
        opti.solver('ipopt', opts)

        try:
            sol = opti.solve()
            optimal_q = sol.value(q)
            # print("Collision free IK solution found!")
        except Exception as e:
            print(f"Solver failed to find collision free IK solution. {e}")
            # optimal_q = opti.debug.value(q)
            # print(optimal_q)

        return optimal_q

class SignedDistanceConstraint(ca.Callback):
    def __init__(self, name, model, data, geom_id1, geom_id2, opts={}):
        Callback.__init__(self)
        self.model = model
        self.data = data
        self.geom_id1 = geom_id1
        self.geom_id2 = geom_id2
        self.fromto = np.zeros(6, dtype=np.float64)
        
        self.construct(name, opts)
    
    def get_n_in(self): return 1
    def get_n_out(self): return 1

    # Required: Tell CasADi the shape of input (nq x 1)
    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.model.nq, 1)

    # Required: Tell CasADi the shape of output (1 x 1)
    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(1, 1)

    def eval(self, arg):
        q = np.array(arg[0]).flatten()

        self.data.qpos[:self.model.nq]  = q
        mj.mj_forward(self.model, self.data)

        dist = mj.mj_geomDistance(self.model, self.data, self.geom_id1, self.geom_id2, 1000.0, self.fromto)
        return [dist]
    
    def has_jacobian(self): 
        return False

def main(args=None):
    rclpy.init(args=args)


    model = mj.MjModel.from_xml_path('models/planar_two_finger_manipulator.xml')
    data = mj.MjData(model)
    minimal_subscriber = ApproachController(model, data)


    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()